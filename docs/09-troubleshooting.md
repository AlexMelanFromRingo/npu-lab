# 09. Когда что-то не работает

Самые частые грабли в порядке частоты.

## Сборка APK

### `JNI: cannot find QnnInterface.h`

Не подхватился `qnn.sdk.root`. Проверь `android-app/local.properties`:
```
qnn.sdk.root=/полный/путь/до/qairt/2.46.0.260424
```

Гарантируй, что внутри есть `include/QNN/QnnInterface.h`. Перебилди — на каждом запуске CMake перечитывает.

### `libnpulab.so not found at runtime`

Скорее всего CMake собрал её для `armeabi-v7a` или `x86_64`, а мы фильтруем только `arm64-v8a`. Чек:
```
unzip -l app/build/outputs/apk/debug/app-debug.apk | grep libnpulab
```

Должно быть `lib/arm64-v8a/libnpulab.so`. Если есть только `lib/armeabi-v7a/` — посмотри `abiFilters` в `app/build.gradle.kts`.

### `org.gradle.api.GradleException: SDK location not found`

Создай `android-app/local.properties` со строкой:
```
sdk.dir=/home/alex_melan/Android/sdk
```

(У нас этот файл создаётся скриптом setup-android-sdk.sh, но если запускаешь Gradle с другой машины — пропиши вручную.)

## Запуск на устройстве

### `dlopen("libQnnHtp.so") failed: cannot find library`

`.so` файлы не положили в `app/src/main/jniLibs/arm64-v8a/`. Скопируй из QNN SDK:
```
cp ~/qnn-sdk/qairt/2.46.0.260424/lib/aarch64-android/libQnn*.so \
   android-app/app/src/main/jniLibs/arm64-v8a/
```

### `dlopen succeeded, but QnnInterface_getProviders returned 0 interfaces`

Версия libQnn*.so не совпадает с версией заголовков, под которые компилировались. Не миксуй SDK версии — копируй .so из той же папки, что используется на компиляции.

### `contextCreateFromBinary returned 4002 (QNN_CONTEXT_ERROR_INVALID_BINARY)`

Бинарник скомпилирован под другую HTP арку (например v79 вместо v81). Проверь на ПК,
под что собран файл:
```bash
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-utility \
    --context_binary models/sd15/unet.bin --json_file /tmp/unet.json
grep -o '"socModel": [0-9]*\|"dspArch": [0-9]*' /tmp/unet.json
```
Для S26 Ultra должно быть `socModel: 87` (SM8850) и `dspArch: 81`. Если нет —
перекачай модели (`fetch-models.py` тянет slug `qualcomm_snapdragon_8_elite_gen5`)
или перекомпилируй на AI Hub, указав правильный device.

### `contextCreateFromBinary returned 14001 (DEVICE)` / NPU не инициализируется

С версии 0.2.0 приложение само прикладывает к тексту ошибки хвост внутреннего
лога QNN (`--- QNN log tail ---`), а на экране Device есть кнопка
**NPU self-test** — полный отчёт с матрицей инициализации. Смотри их в первую
очередь.

**Случай, реально пойманный на S26 Ultra (решён в 0.3.0):**
```
Failed in loading stub: dlopen failed: library "libcdsprpc.so" not found:
needed by .../lib/arm64/libQnnHtpV81Stub.so in namespace clns-9
```
`libcdsprpc.so` — vendor-драйвер FastRPC. С Android 12 (targetSdk 31+) линкер
отдаёт приложению vendor-библиотеки **только** если они задекларированы в
манифесте:
```xml
<uses-native-library android:name="libcdsprpc.so" android:required="false" />
<uses-native-library android:name="libadsprpc.so" android:required="false" />
<uses-native-library android:name="libOpenCL.so"  android:required="false" />
```
Это не «Samsung заблокировал NPU», а штатный механизм Android (vendor public
native libraries): библиотека входит в `/vendor/etc/public.libraries.txt` на
Qualcomm-устройствах, декларация в манифесте открывает к ней доступ. Ровно так
делает официальный ChatApp из qualcomm/ai-hub-apps.

Если 14001 остаётся, классические причины — DSP-загрузчик не нашёл
`libQnnHtpV81Skel.so`:

1. `ADSP_LIBRARY_PATH` должен указывать на `nativeLibraryDir` **до** первого
   обращения к QNN — это делает `QnnRuntimeLibs.setup()` из
   `Application.onCreate()`. В logcat ищи строку `ADSP_LIBRARY_PATH=...`.
2. Skel должен реально лежать в `nativeLibraryDir`. Это требует
   `packaging { jniLibs.useLegacyPackaging = true }` в `build.gradle.kts` и
   наличия `libQnnHtpV81Skel.so` + `libQnnHtpV81.so` в `jniLibs/arm64-v8a/`.
   `QnnRuntimeLibs.setup()` пишет в logcat ошибку, если файлов нет.
3. Версии skel и `libQnnHtp.so` должны быть из одного релиза SDK.

### GPU: `rc=5002 (INVALID_HANDLE)` / CPU: `rc=5005 (CREATE_FROM_BINARY)` — а HTP работает

**Это ожидаемо, а не баг.** Context binary — формат, скомпилированный под
конкретный бэкенд. Все бинарники AI Hub с public S3 собраны под **HTP**:
GPU- и CPU-бэкенды QNN физически не могут их десериализовать. Колонки
GPU/CPU в Benchmark с такими моделями всегда будут падать с понятным
сообщением (мы его дописываем в текст ошибки).

Хочешь честное сравнение NPU vs GPU vs CPU на телефоне — нужен другой формат
той же модели: TFLite (GPU delegate / XNNPACK) или ONNX Runtime (QNN EP vs
CPU EP). AI Hub публикует те же модели в TFLite/ONNX вариантах. Это
зафиксировано как альтернативный путь в README.

### `graphExecute returned 6005 (QNN_GRAPH_ERROR_INVALID_TENSOR)`

Тензоры в `graphExecute` обязаны нести **тот же `id`**, который им назначил
бэкенд при создании контекста (см. QnnGraph.h). Наш рантайм читает id из
`QnnSystemContext_getBinaryInfo` и прокидывает их в каждый вызов — если будешь
менять `qnn_runtime.cpp::Execute`, не потеряй `in_tensors[i].v2.id`. Хостовый
тест `scripts/run-host-introspect-test.sh` проверяет, что у всех тензоров в
наших бинарниках id ненулевые.

### App запускается, но кнопка Generate выдаёт «QNN SDK not compiled in»

Это код-стаб (см. `qnn_runtime.cpp` под `#ifndef NPULAB_HAVE_QNN`). Значит CMake собрал без `-DNPULAB_HAVE_QNN=1` — то есть `qnn.sdk.root` не подхватился. Возврат к началу секции «Сборка».

### Графика рисуется чёрными квадратами / blown-out цветами

Проверь масштаб входа VAE. В AI Hub-бинарнике деление на SD scaling factor
(`z / 0.18215`) **уже зашито внутрь графа** — приложение обязано подавать
сырые латенты. Если поделить ещё раз снаружи, картинка станет серой/блёклой
(вход выйдет за квант-диапазон ±11 и заклампится). Это закреплено тестом
`BinaryMetadataContractTest.vae expects RAW latents`.

### Картинка «шумная»/мыльная при корректном пайплайне

1. Расписание шагов должно быть «leading» (`[951, 901, …, 1]` для 20 шагов),
   а не linspace от 999: вход `timestep` у UNet квантован со scale≈0.01477 —
   максимум представимого ≈968, и t=999 молча клампится на первом же шаге.
2. Формулы DPM++ 2M используют VP-сигмы `sqrt(1-ᾱ)`. Инвариант «идеальная
   модель остаётся на точной траектории» закреплён в `DpmSolverTest`.

### HTP падает после ~50 прогонов

Возможна термальная защита Samsung — телефон троттлит HTP при > 45°C. Дай телефону отдохнуть пару минут, перепрогон без зарядки + в прохладе.

## Сравнение с RTX 4080S

### Картинки на NPU и GPU выглядят по-разному

Ожидаемо. NPU работает в w8a16, GPU в FP16. Разница не должна быть «другой образ», но небольшие отличия в текстуре/цвете нормальны. Если хочется битово сравнить — сконвертируй ту же квантизованную ONNX в обоих направлениях.

### NPU «странно» быстрый на маленьких моделях

Перепроверь warmup. Первый прогон включает компиляцию JIT'ом HTP firmware'а, что добавляет 0.5–1 сек. После 2-3 итераций — стабильное число.

### Бенчмарк NPU нестабильный или медленнее ожидаемого

По умолчанию DSP живёт под DCVS (динамическое управление частотами) и меряешь
ты не сеть, а power management. Наш рантайм при инициализации HTP включает
burst-режим: DCVS off + максимальные voltage corners + RPC polling 9999 мкс
(см. `SetupHtpBurstMode()` в `qnn_runtime.cpp`, рецепт из htp_backend.html в
доках SDK). В logcat должно появиться `HTP perf: burst mode active`. Если
вместо этого warning — бенчмарк всё равно отработает, но цифры будут плавать.

### NPU «странно» медленный на больших моделях

Скорее всего VTCM-spill: модель не помещается в 16 МБ on-chip памяти и активации сбрасываются в DDR. Лечится — меньшая модель или другая стратегия батчинга. Профайл через `qnn-net-run --profiling_level=detailed` на ПК даст разбивку по слоям.

## Где смотреть логи

- На устройстве:
  ```
  adb logcat -s NpuLab:V NpuLabJNI:V QnnRuntime:V QnnRuntimeLibs:V QNN:V
  ```
  (`QNN` — это внутренние сообщения самого QNN-стека: мы подключаем его логгер
  к logcat через `logCreate(NpulabQnnLogCallback, VERBOSE)`; именно там видно
  настоящую причину ошибок вида `rc=14001`.)
- Hexagon-side stderr (DSP пишет туда):
  ```
  adb shell 'cat /vendor/dsp/cdsp/logs/qnn.log 2>/dev/null'
  ```
  (доступно не на всех устройствах без root)
- Profile detail из QNN: создать `Qnn_Profile_Handle_t` в `graphExecute` и потом распечатать через `QnnProfile_getEvents`. Не сделано в нашей текущей реализации — добавь, если будешь развивать.
