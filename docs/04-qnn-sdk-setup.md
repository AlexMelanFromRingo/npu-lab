# 04. Установка Qualcomm AI Engine Direct (QNN) SDK

QNN SDK — это набор заголовков (`include/QNN/*.h`) и shared library под Android (`lib/aarch64-android/libQnn*.so`), которые приложение dlopen'ит на устройстве. SDK раздаёт Qualcomm бесплатно.

## Получаем SDK

С 2025 года SDK живёт на софт-центре Qualcomm и **не требует QPM** (раньше требовал):

1. https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community — здесь раздают community-версию без NDA.
2. Регистрация через Qualcomm ID (можно Google / GitHub / корпоративная почта).
3. Скачать актуальный zip — на момент написания это `v2.46.0.260424.zip` (~1.7 ГБ).
4. Распаковать:
   ```bash
   mkdir -p ~/qnn-sdk && cd ~/qnn-sdk
   unzip ~/Downloads/v2.46.0.260424.zip
   ```

Внутри получишь:
```
qairt/
└── 2.46.0.260424/
    ├── include/QNN/                ← заголовки
    │   ├── QnnInterface.h
    │   ├── QnnContext.h
    │   ├── QnnGraph.h
    │   ├── QnnBackend.h
    │   ├── QnnTensor.h
    │   ├── QnnTypes.h
    │   ├── System/QnnSystemInterface.h        ← для introspection .bin файлов
    │   ├── System/QnnSystemContext.h
    │   ├── HTP/                    ← HTP-specific конфиги
    │   ├── DSP/  GPU/  CPU/
    │   └── ...
    ├── lib/
    │   ├── aarch64-android/        ← runtime .so для устройства (наш target)
    │   │   ├── libQnnSystem.so
    │   │   ├── libQnnHtp.so
    │   │   ├── libQnnHtpV68Stub.so .. V81Stub.so   ← по одному на каждую HTP арку
    │   │   ├── libQnnHtpPrepare.so (87 МБ)         ← JIT компиляция для DLC формата
    │   │   ├── libQnnCpu.so
    │   │   ├── libQnnGpu.so
    │   │   └── ...
    │   └── hexagon-v81/unsigned/   ← .so для DSP-стороны (skel + HTP core).
    │       │                          Это 32-битные Hexagon (DSP6) ELF, не ARM!
    │       ├── libQnnHtpV81Skel.so
    │       └── libQnnHtpV81.so
    ├── bin/                        ← qnn-net-run, qnn-context-binary-utility
    ├── docs/                       ← полная API-документация
    └── examples/                   ← официальные примеры
```

## Подключаем SDK к Android-проекту

### 1. Прописать путь в local.properties

В `android-app/local.properties`:
```
sdk.dir=/home/alex_melan/Android/sdk
qnn.sdk.root=/home/alex_melan/qnn-sdk/qairt/2.46.0.260424
```

Этот файл подхватывает `app/build.gradle.kts` и передаёт путь в CMake через `-DQNN_SDK_ROOT=...`. CMake включает заголовки и определяет `NPULAB_HAVE_QNN=1`.

Если флаг не задан, проект собирается со stub-режимом и в рантайме выкидывает понятную ошибку.

### 2. Скопировать runtime библиотеки в APK

Это нужно сделать один раз. Обрати внимание: **Hexagon-библиотеки (skel) кладутся
в тот же jniLibs/arm64-v8a**, хотя они не ARM — так делает канонический
ChatApp из github.com/qualcomm/ai-hub-apps, и вместе с
`packaging { jniLibs.useLegacyPackaging = true }` инсталлятор раскладывает их в
`nativeLibraryDir` обычными файлами, куда их и находит DSP-загрузчик через
`ADSP_LIBRARY_PATH`:

```bash
cd /home/alex_melan/npu_experiments/android-app
mkdir -p app/src/main/jniLibs/arm64-v8a

QNN=$HOME/qnn-sdk/qairt/2.46.0.260424/lib
# ARM-сторона (наш процесс)
cp "$QNN/aarch64-android/libQnnSystem.so"               app/src/main/jniLibs/arm64-v8a/
cp "$QNN/aarch64-android/libQnnHtp.so"                  app/src/main/jniLibs/arm64-v8a/
cp "$QNN/aarch64-android/libQnnHtpV81Stub.so"           app/src/main/jniLibs/arm64-v8a/
cp "$QNN/aarch64-android/libQnnHtpV81CalculatorStub.so" app/src/main/jniLibs/arm64-v8a/
cp "$QNN/aarch64-android/libQnnHtpProfilingReader.so"   app/src/main/jniLibs/arm64-v8a/
cp "$QNN/aarch64-android/libQnnHtpNetRunExtensions.so"  app/src/main/jniLibs/arm64-v8a/
cp "$QNN/aarch64-android/libQnnCpu.so"                  app/src/main/jniLibs/arm64-v8a/
cp "$QNN/aarch64-android/libQnnGpu.so"                  app/src/main/jniLibs/arm64-v8a/
# DSP-сторона (Hexagon ELF — едут в тот же каталог!)
cp "$QNN/hexagon-v81/unsigned/libQnnHtpV81Skel.so"      app/src/main/jniLibs/arm64-v8a/
cp "$QNN/hexagon-v81/unsigned/libQnnHtpV81.so"          app/src/main/jniLibs/arm64-v8a/
```

В `build.gradle.kts` для Hexagon-файлов добавлен `keepDebugSymbols` — иначе AGP
попытается прогнать их через llvm-strip, который понимает только ARM.

**libQnnHtpPrepare.so не копируем** — он 87 МБ и нужен только для JIT-компиляции DLC-формата на устройстве. Мы работаем с уже скомпилированными context binaries — `Prepare` для них не нужен. Если ты будешь работать с DLC моделями (которые есть в публичном S3 для ESRGAN/MobileNet) — придётся его добавить, и APK подрастёт до ~166 МБ.

### 3. Почему мы возим skel с собой, а не берём вендорский

На S26 Ultra в `/vendor/dsp/cdsp/` действительно лежит какой-то
`libQnnHtpV81Skel.so` — но это версия QAIRT, с которой Samsung собирал
прошивку, и она почти наверняка **не совпадает** с нашей 2.46. Skel и host-side
`libQnnHtp.so` должны быть из одного релиза, иначе ловишь от загадочных ошибок
инициализации до тихих падений. Поэтому наш skel едет в APK, а
`ADSP_LIBRARY_PATH` начинается с `nativeLibraryDir` и только потом fallback'и
на вендорские пути (`/vendor/dsp/cdsp;...`).

## Версия QNN vs версия SoC

QNN SDK содержит stubs для нескольких HTP арок одновременно:

| Snapdragon чип               | HTP arch | Стаб-библиотека             |
|------------------------------|----------|-----------------------------|
| 8 Elite Gen 5 (SM8850) 🎯     | **v81**  | libQnnHtpV81Stub.so         |
| 8 Elite (SM8750)             | v79      | libQnnHtpV79Stub.so         |
| 8 Gen 3 (SM8650)             | v75      | libQnnHtpV75Stub.so         |
| 8 Gen 2 (SM8550)             | v73      | libQnnHtpV73Stub.so         |
| 8(+) Gen 1 (SM8450/SM8475)   | v69      | libQnnHtpV69Stub.so         |

Проверено по реальным context binaries: `qnn-context-binary-utility` для наших
моделей с public S3 показывает `socModel=87` (SM8850) и `dspArch=81`.

Бери последнюю версию SDK для своего чипа — она обратно совместима со старыми архитектурами. 2.46 поддерживает все из таблицы.

## Sanity check (опционально)

В `bin/aarch64-android/` лежит `qnn-net-run` — официальный консольный инструмент Qualcomm. Запустить его на устройстве:
```bash
adb push $QNN_SDK_ROOT/bin/aarch64-android/qnn-net-run /data/local/tmp/
adb shell chmod +x /data/local/tmp/qnn-net-run
adb shell /data/local/tmp/qnn-net-run --help
```

Если help отображается — QNN SDK в принципе живой на твоём устройстве, и проблема (если будет) точно в нашем коде.

## Если у тебя нет аккаунта на Software Center

Альтернатива: **ONNX Runtime Mobile с QNN Execution Provider**. Не требует регистрации, библиотеки идут вместе с ORT. Подробности — [08-alternative-onnx.md](08-alternative-onnx.md).

→ Дальше: [05-using-ai-hub.md](05-using-ai-hub.md) — где брать модели (включая **public путь без AI Hub аккаунта**).
