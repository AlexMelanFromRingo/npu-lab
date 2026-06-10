# NPU Lab

Android-приложение для запуска нейросетей на **Hexagon NPU** (HTP) в Snapdragon 8 Elite Gen 5 (Samsung Galaxy S26 Ultra) + RU-документация и скрипты сравнения с RTX 4080 SUPER.

```
                ┌──────────────────────────────┐
                │       NPU Lab .apk           │
                │                              │
   ┌────────────┼─────────┐    ┌───────────────┼────┐
   │  Generate  │         │    │  Benchmark    │    │      Device
   │ Stable     │         │    │  NPU vs GPU   │    │      info
   │ Diffusion  │         │    │  vs CPU       │    │
   │ 1.5 → 512² │         │    │               │    │
   └────────────┘         └────└───────────────┘────┘
            │                          │
            │           QNN runtime    │
            │ (HTP / Adreno / Kryo)    │
            └──────────┬───────────────┘
                       │
                  Hexagon NPU v81
```

> 📖 **Документация про NPU для новичков:** [docs/00-readme.md](docs/00-readme.md) — 9 разделов про то, что такое NPU, как устроен Hexagon, как с ним общаться из кода, и как корректно сравнивать с десктопным GPU.

## Что внутри

- `android-app/` — Kotlin + Jetpack Compose + Material 3 приложение
  - экраны: **Generate** (Stable Diffusion 1.5), **Speech** (Whisper Base: запись с микрофона → текст на NPU, язык авто), **Benchmark** (модели каталога + свои `.bin` из `models/custom/`), **Models**, **Device** (+ NPU self-test), **Account**
  - JNI-обёртка над **QNN SDK 2.46** (Qualcomm AI Engine Direct), HTP burst-режим (DCVS off) для честных бенчмарков
  - DPM++ 2M sampler (leading-расписание, формулы сверены с diffusers), полный byte-level CLIP BPE tokenizer (кириллица/эмодзи работают, бит-в-бит с HF), Whisper log-mel frontend (golden-тест против HF) и autoregressive-декодер с KV-кэшами — всё на Kotlin
  - ViewModel + StateFlow обвязка между UI и pipeline
- `docs/` — детальная RU-документация о Hexagon NPU (10 разделов, включая [свои модели](docs/10-custom-models.md))
- `scripts/` — установка SDK, скачивание моделей (БЕЗ AI Hub аккаунта по дефолту), **компиляция своих моделей** (`compile-model.py`: AI Hub или полностью локально), PC-бенчмарк, хостовый тест моделей
- `models/` — место, куда складываются скачанные модели (не в git); `models/custom/` — свои скомпилированные `.bin`
- `tools/host_introspect/` — C++ тест: код интроспекции из APK гоняется на ПК против реальных .bin через x86-библиотеки QNN SDK

## Быстрый старт

```bash
# 1. Android SDK (если ещё не стоит)
./scripts/setup-android-sdk.sh

# 2. QNN SDK уже распакован в ~/qnn-sdk/qairt/2.46.0.260424/
#    и прописан в android-app/local.properties

# 3. Модели — БЕЗ аккаунта, прямо с public S3 от Qualcomm
python scripts/fetch-models.py          # SD 1.5 + Whisper Base

# 4. Сборка APK
cd android-app && ./gradlew :app:assembleDebug

# 5. Установка на телефон
adb install -r app/build/outputs/apk/debug/app-debug.apk

# 6. Модели на телефон
cd .. && adb push models /sdcard/Android/data/io.melan.npulab/files/

# 7. Запустить приложение, открыть Generate, ввести prompt, нажать кнопку.
```

## Сборка release APK

```bash
cd android-app
./gradlew :app:assembleRelease
# →  app/build/outputs/apk/release/app-release.apk  (~57 МБ: R8 + QNN-библиотеки,
#     включая Hexagon-skel и 90 МБ libQnnHtpPrepare.so в сжатом виде)
```

Подпись берётся из `local.properties` (release.store.* / release.key.*). Дефолтный
keystore `android-app/release.keystore` сгенерирован локально, в git не уходит.
Если переменных нет — release собирается **без подписи** (с предупреждением в логе),
устанавливать такой APK Android откажется.

## Аккаунт AI Hub в приложении

Во вкладке **Account** можно сохранить один или несколько API-токенов:
- Каждый хранится в EncryptedSharedPreferences (AES-256-GCM, ключ из Android Keystore — переживает рестарт устройства, не уходит ни в backup, ни в логи).
- Один аккаунт активен, остальные — про запас. Переключение в один клик.
- Кнопка **Verify** делает `GET /api/v1/users/auth/user/` с заголовком `Authorization: token <T>` на `workbench.aihub.qualcomm.com` и показывает email/username, который вернул сервер. На неправильный токен — красная плашка с причиной.
- Удаление через диалог подтверждения.

Сейчас токен используется только для Verify; runtime-загрузка моделей с AI Hub в приложении — TODO. Скрипт `scripts/fetch-models.py` на PC по-прежнему работает через `qai-hub configure` (свой `~/.qai_hub/client.ini`).

## Public S3 vs AI Hub аккаунт

Скрипт `fetch-models.py` по умолчанию тянет всё с **публичного S3 от Qualcomm** — никакой регистрации, никакого API ключа. Это работает для:
- ✅ Stable Diffusion 1.5 (w8a16, 3 части)
- ✅ Whisper Base (float)
- ✅ ~150 других моделей из каталога https://huggingface.co/qualcomm

AI Hub аккаунт нужен только если:
- Хочется конкретно Real-ESRGAN или MobileNet-V3 в формате context binary (на S3 они лежат только как DLC).
- Хочется скомпилировать свою модель.

В таком случае:
```bash
pip install qai-hub qai-hub-models
qai-hub configure --api_token <твой_токен>  # https://aihub.qualcomm.com → Profile → API token
python scripts/fetch-models.py --source aihub --models esrgan mobilenet
```

Подробности: [docs/05-using-ai-hub.md](docs/05-using-ai-hub.md).

## Tech stack

- **Android**: Kotlin 2.0, Jetpack Compose, Material 3, Compose Navigation 2.8, AGP 8.7, SDK 35, NDK 26
- **Native**: C++17, CMake 3.22, dlopen
- **NPU**: Qualcomm QNN SDK 2.46.0
- **Models**: Qualcomm public S3 mirror (SD 1.5 w8a16, Whisper Base float). Опционально — Qualcomm AI Hub.
- **Python (host-side)**: stdlib only для public mirror'а; `qai-hub` для AI Hub пути

## Что компилируется без QNN SDK

Приложение собирается со stub-режимом, если в `local.properties` не указан `qnn.sdk.root`. UI работает, экран Device показывает характеристики телефона, кнопки Generate/Benchmark выкидывают понятную ошибку при попытке инференса. Это удобно для проверки UI без полного setup'а.

## Состояние проекта

| Компонент                       | Статус | Примечания                                        |
|---------------------------------|--------|---------------------------------------------------|
| Android SDK setup script        | ✅      | `scripts/setup-android-sdk.sh`                    |
| Gradle проект + Compose + Material 3 | ✅ | собирается gradlew :app:assembleDebug             |
| JNI слой (NpuLabNative)         | ✅      | context→runtime маршрутизация для нескольких бэкендов, проверка direct-буферов |
| C++ QNN runtime (2.46 API)      | ✅      | `qnn_runtime.cpp` — `getBinaryInfo` + V2 тензоры **с обязательными id**, HTP burst-режим |
| Деплой Hexagon-библиотек        | ✅      | skel в `jniLibs` + `ADSP_LIBRARY_PATH=nativeLibraryDir` (канон ChatApp) |
| DPM++ 2M scheduler              | ✅      | VP-сигмы как в diffusers, leading-расписание `[951…1]` |
| CLIP BPE tokenizer              | ✅      | полный byte-level BPE: пунктуация, кириллица, эмодзи — бит-в-бит с HF |
| StableDiffusionPipeline         | ✅      | входы UNet по именам; VAE получает сырые латенты (деление на 0.18215 уже в графе) |
| BenchmarkRunner                 | ✅      | warmup + 8 итераций, median/p95/min/max           |
| GenerateViewModel / Screen      | ✅      | StateFlow, прогресс по шагам                      |
| BenchmarkViewModel / Screen     | ✅      | прогресс по ячейкам model×backend                 |
| DeviceInfoScreen                | ✅      | SoC, HTP arch, libQnn*.so присутствие, RAM, CPU   |
| Public S3 download (no auth)    | ✅      | `scripts/fetch-models.py` default mode            |
| AI Hub fallback                 | ✅      | `scripts/fetch-models.py --source aihub`          |
| PC-side benchmark (CUDA)        | ✅      | `scripts/benchmark-pc.py`                         |
| Whisper Speech-to-Text UI       | ✅      | экран Speech: микрофон → log-mel (golden-тест vs HF) → encoder → decoder c KV-кэшами, язык авто |
| Свои модели на NPU              | ✅      | `scripts/compile-model.py` (AI Hub / локально, проверено end-to-end) + автоподхват `models/custom/*.bin` в Benchmark |
| NPU self-test                   | ✅      | Device → кнопка: матрица deviceCreate × createFromBinary + полный QNN-лог, копируется без adb |
| Юнит-тесты (JVM)                | ✅      | 38 тестов: токенизаторы vs HF-фикстуры, DPM++ инварианты, log-mel golden, fp16, квантование, контракт схемы против реальных .bin |
| Хостовый интроспекция-тест (C++)| ✅      | `scripts/run-host-introspect-test.sh` — код из APK против реальных .bin через x86 libQnnSystem |
| Документация (RU, 10 разделов)  | ✅      | `docs/`                                           |

## Тесты

```bash
# JVM-тесты (токенизатор, scheduler, квантование, контракт метаданных моделей)
cd android-app && ./gradlew :app:testDebugUnitTest

# Хостовый C++-тест: интроспекция реальных .bin тем же кодом, что в APK
scripts/run-host-introspect-test.sh
```

Контракт-тесты читают `app/src/test/resources/binmeta/*.json` — несмодифицированные
дампы `qnn-context-binary-utility` с реальных моделей (socModel=87/SM8850,
dspArch=81). Фикстуры токенизатора сгенерированы HF `transformers.CLIPTokenizer`.

## Известные ограничения

- **На реальном устройстве пока не запускалось** (билд-машина без телефона). Но
  все предположения о моделях теперь проверены на ПК против самих .bin-файлов:
  имена/порядок/типы тензоров, ненулевые id (контракт `graphExecute`),
  квант-диапазоны timestep/latent. Если что-то всё же упадёт — см.
  [docs/09-troubleshooting.md](docs/09-troubleshooting.md) и
  `adb logcat -s NpuLab:V QnnRuntime:V NpuLabJNI:V QnnRuntimeLibs:V QNN:V`.
- **DLC модели** требуют `libQnnHtpPrepare.so` — она уже в jniLibs (90 МБ), но
  для ESRGAN/MobileNet всё равно удобнее собрать context binary через AI Hub
  (см. [docs/04-qnn-sdk-setup.md](docs/04-qnn-sdk-setup.md)). Если DLC не нужны —
  можно удалить Prepare из jniLibs и срезать ~90 МБ с APK.
- **Whisper** подключён только в Benchmark (encoder); полноценный ASR-экран с
  KV-кэшем декодера — TODO.

## Альтернативный путь — ONNX Runtime + QNN EP

Зафиксирован в [docs/08-alternative-onnx.md](docs/08-alternative-onnx.md). Более портативный путь — тот же ONNX-граф работает и на NPU телефона (QNN EP), и на RTX 4080S (CUDA EP). Сейчас на чистом QNN, чтобы понять железо изнутри; миграция на ORT — следующий разумный шаг.

## Benchmark: что реально сравнивается

Колонки GPU/CPU в Benchmark с моделями из public S3 **падают by design**:
context binary компилируется под конкретный бэкенд, и все бинарники AI Hub —
HTP-only. На устройстве сравниваем HTP с разными моделями, а «NPU vs GPU vs
CPU» в одном форвард-проходе требует TFLite/ONNX-вариант модели (см.
альтернативы ниже). Сравнение с RTX 4080S живёт в `scripts/benchmark-pc.py`.

## Другие альтернативы, к которым можно вернуться

- **NPU vs GPU vs CPU на устройстве: TFLite или ONNX Runtime.** AI Hub
  публикует те же модели в TFLite (GPU delegate / XNNPACK) и ONNX (QNN EP /
  CPU EP). Подключение TFLite-варианта Whisper/MobileNet в Benchmark даст
  честное трёхстороннее сравнение — бэкенды QNN GPU/CPU не читают HTP context
  binaries.

- **Sampler: Euler вместо DPM++ 2M.** Референс Qualcomm
  (`qai_hub_models/models/_shared/stable_diffusion/app.py`) использует
  `EulerDiscreteScheduler` — именно на его траекториях калибровалась
  квантизация UNet. Наш DPM++ 2M математически корректен (сверен с diffusers,
  закреплён тестами) и обычно даёт лучше качество за 20 шагов, но если на
  устройстве проявятся артефакты квантования — попробовать Euler первым делом.
- **Деплой Hexagon-библиотек через assets.** Текущий путь — jniLibs +
  `useLegacyPackaging` (канон ChatApp). Альтернатива, если когда-нибудь
  захочется `extractNativeLibs=false` (APK без распаковки): носить skel в
  assets и при первом старте копировать в `filesDir` через `AssetManager.open()`
  (потоковое копирование; `openFd()` не работает — assets сжаты), направив туда
  `ADSP_LIBRARY_PATH`.

## Лицензия

MIT для нашего кода. Модели с public S3 / AI Hub имеют свои лицензии (см. карточку модели на HuggingFace).
