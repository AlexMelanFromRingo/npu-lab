# NPU Lab

Run real neural networks — **Stable Diffusion 1.5**, **Whisper** (Base / Small /
Large-v3-Turbo), and **your own models** — directly on the **Hexagon NPU (HTP)**
of a Snapdragon 8 Elite Gen 5 phone (Samsung Galaxy S26 Ultra), through
Qualcomm AI Engine Direct (QNN), with no cloud and no NNAPI middleman.

Built as a hands-on lab to compare a phone NPU against a desktop GPU
(RTX 4080 SUPER) on identical workloads — and battle-tested on real hardware.

```
 Kotlin / Compose UI            C++ runtime                Hexagon NPU (cDSP)
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ Generate (SD 1.5)    │   │ qnn_runtime.cpp      │   │ libQnnHtpV81Skel.so  │
│ Speech (Whisper ×3)  │──▶│  dlopen libQnnHtp.so │──▶│  + libQnnHtpV81.so   │
│ Benchmark (+custom)  │   │  context binaries    │   │  (shipped in the APK)│
│ Device (self-test)   │   │  burst-mode DCVS     │   │  via FastRPC         │
└──────────────────────┘   └──────────────────────┘   └──────────────────────┘
```

## Measured on a Galaxy S26 Ultra (SM8850, Hexagon V81)

| Workload | Result |
|---|---|
| Whisper Base encoder (30 s audio) | **~24.6 ms** median, ±0.8 ms |
| Whisper Base autoregressive decode | **185–222 tok/s** |
| Speech-to-text languages | RU / EN / UK / DE verified, 99 supported, auto-detect |

## What's inside

- **`android-app/`** — Kotlin 2.0 + Jetpack Compose + Material 3
  - **Generate** — Stable Diffusion 1.5 (w8a16): CLIP byte-level BPE tokenizer,
    selectable sampler — EulerDiscrete (the scheduler the AI Hub binaries were
    calibrated against, default) or DPM-Solver++ 2M (verified against
    diffusers) — UNet ×2 per step with CFG, VAE; marshalling in Kotlin,
    inference on HTP.
  - **Speech** — Whisper on the NPU: log-mel frontend (1:1 port of HF
    `WhisperFeatureExtractor`, golden-tested), fp16 KV-cache ping-pong decoder,
    forced `[SOT, lang, transcribe, notimestamps]` prompt, per-chunk language
    auto-detect, recordings up to 2 min (30 s windows). Model flavor selector:
    Base / Small / Large-v3-Turbo (80- and 128-mel, v2/v3 vocab handled).
  - **Benchmark** — 37-model catalog (SD, Whisper ×4, and a 31-model zoo of
    classification / detection / depth / segmentation / super-resolution
    networks as device-agnostic float DLCs) + any custom `.bin`/`.dlc` dropped
    into `models/custom/`, warmup + median/p95. **DLC models run on HTP, GPU
    and CPU** — composed and prepared on device per backend, so the three-way
    comparison is real.
  - **Device** — SoC info plus a one-tap **NPU self-test**: every deviceCreate
    flavor × context load, platform info as QNN sees it, full internal log —
    copyable, no adb required.
  - **Models** — in-app downloads from Qualcomm's public S3 (no account).
- **`scripts/`**
  - `fetch-models.py` — pull prebuilt context binaries (SD 1.5, Whisper ×3) from
    public S3, no registration.
  - `compile-model.py` — **your ONNX/TorchScript → HTP context binary** for
    SM8850: cloud route (AI Hub) or fully local route (qairt-converter +
    qnn-context-binary-generator; auto-provisions Python 3.10 via uv).
  - `run-host-introspect-test.sh` — runs the APK's real introspection code
    against the real `.bin` files on your PC through x86 QNN libraries.
  - `benchmark-pc.py` — the RTX side of the comparison.
- **`docs/`** — a 10-part course on the Hexagon NPU (in Russian): what an NPU
  is, how HTP talks FastRPC, SDK setup, AI Hub, benchmarking methodology,
  troubleshooting with real error codes, custom-model compilation.
- **`tools/host_introspect/`** — C++ host test sharing the exact
  `binary_info_walk.h` the APK ships.

## Quick start

```bash
# 0. Android SDK + QNN SDK (see docs/04-qnn-sdk-setup.md)
./scripts/setup-android-sdk.sh
#    QAIRT SDK → ~/qnn-sdk/qairt/<version>/, path goes into android-app/local.properties

# 1. QNN runtime libraries into the APK (not committed — Qualcomm SDK files)
./scripts/copy-qnn-libs.sh

# 2. Models from Qualcomm's public S3 (no account needed)
python scripts/fetch-models.py                    # SD 1.5 + Whisper Base
python scripts/fetch-models.py --models whisper_small whisper_turbo   # optional

# 3. Build & install
cd android-app && ./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb push ../models /sdcard/Android/data/io.melan.npulab/files/

# …or skip adb entirely: install the APK any way you like and download
# models from the in-app Models tab.
```

### Your own model on the NPU

```bash
python scripts/compile-model.py my_model.onnx              # AI Hub (fp16)
python scripts/compile-model.py my_model.onnx --quantize w8a16
python scripts/compile-model.py my_model.onnx --source local   # offline
adb push models/custom /sdcard/Android/data/io.melan.npulab/files/models/
# → the model appears as a Benchmark chip automatically
```

Constraints (static shapes, HTP op coverage, SoC binding):
[docs/10-custom-models.md](docs/10-custom-models.md).

## Tests

```bash
cd android-app && ./gradlew :app:testDebugUnitTest   # 46 JVM tests
scripts/run-host-introspect-test.sh                  # C++ vs real binaries
```

The JVM suite pins the moving parts to ground truth: tokenizers byte-exact
against HF fixtures (incl. Cyrillic and emoji), the DPM++ solver against an
exact-posterior invariant, the log-mel frontend against golden output of the
HF extractor (80 and 128 mel), quantization helpers against the real
scale/offset values, and the whole tensor schema against unmodified
`qnn-context-binary-utility` dumps of the production binaries.

## The pitfalls this repo already paid for

Documented in detail in [ARCHITECTURE.md](ARCHITECTURE.md) and
[docs/09-troubleshooting.md](docs/09-troubleshooting.md); the short list:

1. **`<uses-native-library libcdsprpc.so>`** — without this manifest entry
   (Android 12+) the FastRPC driver is invisible to your linker namespace and
   every HTP call dies with `rc=14001`. This single line cost the most debugging.
2. **Tensors must carry their backend-assigned `id`** in `graphExecute` —
   rebuilding `Qnn_Tensor_t` with `id=0` fails with `INVALID_TENSOR`.
3. **Hexagon `.so` files ride inside `jniLibs/arm64-v8a`** (they're DSP6 ELFs,
   not ARM!) with `useLegacyPackaging=true` and a `keepDebugSymbols` guard so
   AGP doesn't strip them. Asset-extraction schemes break on compressed assets.
4. **Context binaries are backend- and SoC-specific** — an HTP binary will not
   load on the QNN GPU/CPU backends, and a V79 binary won't run on V81. The
   cross-backend format is **DLC** (composed + prepared on device per backend
   via `systemDlcComposeGraphs`) — that's what the model zoo uses.
5. **AI Hub SD 1.5 quirks**: the VAE divides by 0.18215 *inside* the graph
   (don't pre-divide), and the UNet's quantized `timestep` grid tops out at
   ≈968 (use the *leading* schedule, never linspace's t=999).
6. **HTP idles under DCVS** — set the documented burst power config or your
   benchmark measures power management instead of the network.

## Status & limitations

- Verified on a real Galaxy S26 Ultra: SD 1.5 generation, Whisper STT (4
  languages incl. auto-detect), benchmarks, custom-model loading.
- QNN GPU/CPU backends can't deserialize HTP context binaries (by design) —
  use the DLC zoo models for three-way HTP/GPU/CPU benchmarks.
- Whisper streaming (real-time) and SD img2img are not implemented.

## Docs in Russian

The full beginner-friendly NPU course lives in [docs/](docs/00-readme.md),
and the original Russian README is preserved as [README.ru.md](README.ru.md).

## License

MIT for everything in this repo. Qualcomm QNN SDK libraries and AI Hub model
binaries are **not** included and come under Qualcomm's own license terms —
`scripts/copy-qnn-libs.sh` and `scripts/fetch-models.py` pull them from your
own SDK install / Qualcomm's public mirror.
