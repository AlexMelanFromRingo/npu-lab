# 06. Жизненный цикл QNN runtime в коде

Тут мы привязываемся к актуальному коду в `android-app/`. Если читаешь это позже и код разошёлся с документом — код первичен.

## Слои нашего приложения

```
┌────────────────────────────────────────────────────────────┐
│ UI (Compose, screens/)                                     │
├────────────────────────────────────────────────────────────┤
│ Pipelines:  StableDiffusionPipeline, BenchmarkRunner       │
├────────────────────────────────────────────────────────────┤
│ Kotlin runtime wrapper:  QnnRuntime, QnnModel              │
├────────────────────────────────────────────────────────────┤
│ JNI:  NpuLabNative (Kotlin) <-> qnn_jni.cpp                │
├────────────────────────────────────────────────────────────┤
│ C++ wrapper:  qnn_runtime.cpp                              │
├────────────────────────────────────────────────────────────┤
│ QNN System API (dlopen):                                   │
│   libQnnSystem.so + libQnnHtp.so + libQnnHtpV81Stub.so     │
├────────────────────────────────────────────────────────────┤
│ HTP DSP-сторона (в APK!):   libQnnHtpV81Skel.so + V81.so   │
└────────────────────────────────────────────────────────────┘
```

## Жизненный цикл одного inference

### 1. Открываем backend

```kotlin
val rt = QnnRuntime(NpuLabNative.Backend.HTP, htpArch = 79)
```

Что происходит ниже:
1. JNI: `initBackend(0, 79)` → C++: `QnnRuntime::Create(Backend::HTP, 79)`
2. `dlopen("libQnnHtp.so", RTLD_NOW)` — динамическая загрузка backend lib
3. `dlsym("QnnInterface_getProviders")` — достаём указатель на entry point
4. Через него получаем vtable `QnnInterface_t*` (это структура с указателями на ~50 функций)
5. `iface->backendCreate(...)` — создаём backend handle
6. Возвращаем opaque `uint64_t` handle через JNI в Kotlin

### 2. Грузим граф

```kotlin
val model = rt.loadModel("/sdcard/.../models/sd15/unet.bin")
```

→ `iface->contextCreateFromBinary(backend, device, perfInfra, blob, size, &ctx, signal)` — десериализует граф из памяти + связывает с backend. Это самая дорогая операция: для UNet SD 1.5 (~1.3 ГБ) занимает ~3-5 секунд первый раз. После — кэшируется в page cache OS.

→ `iface->graphRetrieve(ctx, "model", &graph)` — достаём указатель на скомпилированный граф.

→ Нам также нужны схемы входов/выходов (имена тензоров, dtype, shape). Это даёт `QnnSystemContext_getBinaryInfoFromBinary` из `libQnnSystem.so` — мы вызываем её перед `contextCreateFromBinary`. Она возвращает `QnnSystemContext_BinaryInfo_t` с массивом `QnnSystemContext_GraphInfo_t`, каждый из которых содержит список `Qnn_Tensor_t` (имя, dtype, dimensions). Бинарь читается дважды (один раз для introspection, второй — для самого контекста); для больших моделей это ~100 мс оверхеда на загрузку.

### 3. Готовим тензоры

QNN ожидает structures вида `Qnn_Tensor_t` с указателями на хост-память. Память должна быть **выровнена**, и для лучшей производительности — это **ION/dma-buf** (общий с DSP, без копирования). У нас в первой версии используется `ByteBuffer.allocateDirect` — это просто malloc'd память, выровненная на cacheline. Это работает, но добавляет одну копию (CPU→DSP DMA), которой можно избежать.

```kotlin
val input = ByteBuffer.allocateDirect(numel * 2).order(ByteOrder.nativeOrder())
for (v in latents) input.putShort(floatToHalf(v))  // FP32 → FP16 на нашей стороне
```

### 4. Запуск

```kotlin
val executionUs = model.execute(arrayOf(input1, input2, ...), arrayOf(output1, ...))
```

→ `iface->graphExecute(graph, inputs, n_inputs, outputs, n_outputs, profile, signal)` — синхронный вызов, возвращается когда DSP закончил. Под капотом FastRPC передаёт сигнал на DSP, ждёт ответа.

`executionUs` — wall-time от вызова до возврата, в микросекундах. Реальная DSP-side latency меньше — разница идёт на RPC и копирования.

### 5. Чтение результата

ByteBuffer'ы заполнены DSP'ом. Если выход FP16 — конвертируем в FP32 на CPU:
```kotlin
val out = FloatArray(numel)
buf.rewind()
for (i in 0 until numel) out[i] = halfToFloat(buf.short)
```

### 6. Освобождение

```kotlin
model.close()      // QnnContext_free
rt.close()         // QnnBackend_free + dlclose
```

## Полный пример: один прогон MobileNet-V3

```kotlin
fun classify(bitmap: Bitmap, modelPath: String): IntArray {
    QnnRuntime(NpuLabNative.Backend.HTP).use { rt ->
        rt.loadModel(modelPath).use { model ->
            // Подготовка входа: bitmap → 1x3x224x224 NCHW в uint8 (модель квантована)
            val input = ByteBuffer.allocateDirect(3 * 224 * 224)
            for (y in 0 until 224) for (x in 0 until 224) {
                val p = bitmap.getPixel(x, y)
                input.put((Color.red(p)).toByte())
                // ... три плоскости подряд для NCHW
            }
            input.rewind()
            val outputs = model.allocOutputs()
            model.execute(arrayOf(input), outputs)
            // outputs[0] — [1, 1000] логиты ImageNet
            return topK(outputs[0].asFloatBuffer(), k = 5)
        }
    }
}
```

## Многократное использование

Для SD 1.5 UNet вызывается 20 раз (по числу шагов), плюс дважды на шаг (cond + uncond для CFG). Загружать `.bin` каждый раз — медленно. В `StableDiffusionPipeline` модель грузится один раз в конструкторе и закрывается в `close()`. ViewModel держит ссылку на pipeline пока активен экран.

## Однопоточность

QNN graph execution **не безопасен в многих потоках**. Если хочешь параллельных инференсов — нужно делать несколько контекстов (caveat: каждый займёт VTCM и DSP при «горячем» состоянии, реальной параллельности на одном чипе не получишь). Для нашего случая всё последовательно: один поток в `Dispatchers.IO`, ViewModel выставляет состояние в UI через StateFlow.

## Дальше

→ [07-benchmarking.md](07-benchmarking.md) — методика сравнения с RTX 4080S.
