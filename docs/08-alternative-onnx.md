# 08. Альтернативный путь — ONNX Runtime + QNN Execution Provider

Запомни этот документ как «версия 2 эксперимента». Сейчас приложение собрано на чистом QNN SDK + AI Hub. Альтернатива — **ONNX Runtime Mobile с QNN Execution Provider** — даёт почти ту же производительность, но с более портативным кодом и без регистрации на QPM.

## Почему стоит вернуться

| Свойство                          | QNN SDK (наш путь)         | ORT + QNN EP                |
|-----------------------------------|----------------------------|------------------------------|
| Зависимость от Qualcomm SDK       | Нужен SDK + аккаунт QPM   | Только pip install onnxruntime-qnn |
| Зависимость от AI Hub             | Любая модель → AI Hub      | Любая ONNX модель → onnxruntime |
| Switch между бэкендами            | Перезагрузка контекста    | Один параметр в `SessionOptions` |
| Скорость на HTP                   | Reference (100%)           | 90–95% от QNN SDK            |
| Размер runtime                    | Несколько .so по 5–20 МБ  | Один .so ~30 МБ (всё внутри) |
| API в коде                        | C, ~30 функций            | C++/Java, привычный          |
| Поддержка CPU/CUDA EP             | Нет                        | Да — тот же код запускается на десктопе |
| Тренировка интуиции               | Глубже понимаешь HTP      | Меньше «черных ящиков»       |

Ключевой выигрыш ORT — это **одинаковый код для телефона и десктопа**. Можно прогнать ту же `.onnx` через CUDA EP на 4080S, через CPU EP на любом ноутбуке, через QNN EP на S26 Ultra — и получить *сравнимые* числа на одной и той же модели.

## Как бы выглядел тот же проект

### build.gradle.kts

Заменить наш custom CMake / JNI на:

```kotlin
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    // Замени на onnxruntime-android-qnn если будет отдельный артефакт; на момент
    // написания QNN EP включён в основной onnxruntime-android начиная с 1.17.
}
```

Никакого `externalNativeBuild`, никакого `jniLibs/arm64-v8a/libQnn*.so` руками. Всё прячется внутри AAR-а.

### Подключение QNN EP

```kotlin
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.QnnExecutionProvider

fun runOnNpu(modelBytes: ByteArray, inputs: Map<String, OnnxTensor>): Map<String, OnnxTensor> {
    val env = OrtEnvironment.getEnvironment()
    val opts = OrtSession.SessionOptions().apply {
        // EP-specific options для QNN
        val qnnOpts = mapOf(
            "backend_path" to "libQnnHtp.so",
            "htp_performance_mode" to "burst",        // или sustained_high_performance
            "htp_graph_finalization_optimization_mode" to "3",  // OptimizeForInference
            "rpc_control_latency" to "100",
            "soc_model" to "60",                       // SM8850 в QNN reference
        )
        addQnn(qnnOpts)
    }
    return env.createSession(modelBytes, opts).use { session ->
        session.run(inputs).also { /* read outputs */ }
    }
}
```

### Конвертация моделей

Не нужен AI Hub — можно работать с любыми `.onnx`:

```python
# На PC, один раз:
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# 1. Получаем ONNX модель (например из HF):
import torch
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
torch.onnx.export(unet, ..., "unet.onnx")

# 2. Квантуем (нужно, чтобы QNN EP положил граф на HTP):
quantize_dynamic("unet.onnx", "unet_quant.onnx", weight_type=QuantType.QInt8)

# 3. Готово — этот .onnx адресует и HTP, и CUDA, и CPU без изменений
```

### Один EP — один код

```kotlin
enum class TargetEP { QNN_HTP, CPU, /* на десктопе ещё CUDA, ROCm, DirectML */ }

fun makeSession(model: ByteArray, target: TargetEP): OrtSession {
    val opts = OrtSession.SessionOptions()
    when (target) {
        TargetEP.QNN_HTP -> opts.addQnn(mapOf("backend_path" to "libQnnHtp.so"))
        TargetEP.CPU -> { /* default */ }
    }
    return OrtEnvironment.getEnvironment().createSession(model, opts)
}
```

Тот же `OrtSession.run` для всех EP. ORT сам решает, какие узлы графа лягут на NPU, какие на CPU (если есть unsupported ops).

## Чего не хватает в ORT + QNN EP

- **Меньший контроль над VTCM**: QNN EP сам решает, что кэшировать. На пограничных моделях это даёт хуже latency, чем ручной QNN.
- **Один граф = одна сессия**: для пайплайнов из 3 моделей (SD) — три сессии, и между ними нужно копирование через CPU. В чистом QNN можно цепочкой через ION-buffers без отрыва от DSP.
- **Метаданные**: profiling от QNN EP отдаёт меньше деталей, чем `qnn-net-run --profiling_level=detailed`.

## Когда возвращаться к этой ветке

Хороший момент — когда:
- надоест возиться с тремя `.so` файлами и зависимостью от QPM;
- появится желание прогнать ту же модель на 4080S через ORT CUDA EP для честного сравнения архитектур;
- нужно будет шарить пайплайн между Android и десктопом (например, CLI для дебага).

Сейчас же мы остались на чистом QNN, чтобы потрогать «настоящий» SDK и понять, как именно вызывается NPU — это знание не пропадёт даже после миграции на ORT.

→ Дальше: [09-troubleshooting.md](09-troubleshooting.md)

## Реализовано в приложении (Studio → ONNX)

Этот путь теперь встроен: вкладка **Studio → ONNX** грузит любой `.onnx` и гоняет
его на NPU через `onnxruntime-android-qnn` (ORT + QNN EP) — **на устройстве, без ПК**.

- AAR (`com.microsoft.onnxruntime:onnxruntime-android-qnn:1.26.0`) несёт только
  `libonnxruntime.so` (~7.5 МБ), **без своих QNN-библиотек** — берёт наши 2.46
  через `backend_path=<nativeLibDir>/libQnnHtp.so`. Поэтому второй QNN-стек не
  конфликтует с рукописным пайплайном (SD/Whisper/zoo).
- Бэкенды: HTP (NPU) / GPU / CPU — через `backend_path` (libQnnHtp/Gpu/Cpu.so).
- `ep.context_enable=1` → ORT кэширует скомпилированный контекст рядом с моделью
  (первый запуск компилирует на устройстве, дальше быстро).
- Неподдержанные QNN-операторы автоматически уезжают на CPU (ORT partitioning).
- Код: `OrtNpuRunner.kt` (Java ORT API: `SessionOptions.addQnn(...)`),
  `OnnxScreen`/`OnnxViewModel`. Это и есть «компиляция любой модели под NPU,
  встроенная в APK» — то, чего нельзя сделать через x86-only `qairt-converter`.

**Замечание про SD:** твои SD-компоненты в ONNX (text_encoder/vae из чекпойнта)
пойдут именно сюда — импортируешь .onnx и запускаешь, без ручного повторения
конвенций AI Hub. Полный SD-пайплайн (3 модели + sampler + токенайзер) поверх
ORT — следующий шаг.
