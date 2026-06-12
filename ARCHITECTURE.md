# Architecture

How NPU Lab talks to the Hexagon NPU, why each layer looks the way it does,
and the field notes from making it run on a real Galaxy S26 Ultra.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ UI (Jetpack Compose, Material 3)                                        │
│   GenerateScreen   TranscribeScreen   BenchmarkScreen   DeviceInfoScreen│
│        │ StateFlow        │                  │                │         │
│   GenerateVM        TranscribeVM        BenchmarkVM       (self-test)   │
├────────┼──────────────────┼──────────────────┼────────────────┼─────────┤
│ Pipelines (Kotlin)        │                  │                │         │
│   StableDiffusionPipeline WhisperPipeline   BenchmarkRunner             │
│   ├ ClipTokenizer         ├ WhisperTokenizer (decode-only)              │
│   ├ DpmSolverMultistep    ├ LogMelFrontend (DFT+mel, coroutines)        │
│   └ quant/layout helpers  └ Fp16 + KV-cache ping-pong                   │
├─────────────────────────────────────────────────────────────────────────┤
│ QnnRuntime.kt (Kotlin façade) ── NpuLabNative (JNI) ── qnn_jni.cpp      │
├─────────────────────────────────────────────────────────────────────────┤
│ qnn_runtime.cpp (C++17)                                                 │
│   dlopen libQnnSystem.so / libQnnHtp.so|Gpu|Cpu                         │
│   introspect (binary_info_walk.h) → create context → graphExecute      │
│   HTP burst power config · QNN log ring buffer · self-test matrix      │
├─────────────────────────────────────────────────────────────────────────┤
│ QNN host libs (aarch64, in APK)        Hexagon libs (DSP6 ELF, in APK) │
│   libQnnHtp.so, libQnnHtpV81Stub.so ──FastRPC──▶ libQnnHtpV81Skel.so   │
│   libQnnSystem.so, libQnnGpu/Cpu.so   (libcdsprpc.so, /dev/fastrpc)    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1. Model format: QNN context binaries

Everything the app runs is a **serialized QNN context binary** (`.bin`):
a graph already compiled for one specific backend and SoC. We use the ones
Qualcomm AI Hub publishes on its public S3 mirror (compiled for
`socModel=87` / SM8850, `dspArch=81` / Hexagon V81), or ones you compile
yourself with `scripts/compile-model.py`.

Key properties that shape the code:

- **Backend-specific.** An HTP binary cannot be deserialized by the QNN
  GPU/CPU backends (`rc=5005`). The Benchmark screen explains this instead of
  pretending the comparison is possible.
- **SoC/arch-specific.** V79 binaries don't run on V81. The self-test and
  `qnn-context-binary-utility` both surface what a binary was built for.
- **Self-describing.** `QnnSystemContext_getBinaryInfo` exposes graph names,
  tensor names/shapes/dtypes/quant params **and tensor ids**. The runtime
  trusts the binary, not hardcoded constants — which is what lets Whisper
  Tiny / Small / Large-v3-Turbo run through the same pipeline unchanged.

The second supported format is **DLC** (`.dlc`) — a device-agnostic graph
container. `LoadDlc()` creates an empty context on the selected backend, has
`libQnnSystem` compose the graphs into it (`systemDlcCreateFromFile` →
`systemDlcComposeGraphs`), then finalizes (= online prepare; HTP pulls in
`libQnnHtpPrepare.so`). Slower to load than a context binary, but one float
DLC runs on HTP *and* GPU *and* CPU — the model zoo and the three-way
benchmark are built on this.

## 2. Native runtime (`app/src/main/cpp/`)

`qnn_runtime.cpp` owns one backend instance per `QnnRuntime` object:

1. `dlopen("libQnnSystem.so")` → introspection interface.
2. `dlopen("libQnnHtp.so" | Gpu | Cpu)` → `QnnInterface_getProviders`.
3. `logCreate(callback, VERBOSE)` — QNN's internal log goes to logcat tag
   `QNN` **and** into a ring buffer; WARN/ERROR tail is appended to every
   error message so screenshots are self-diagnosing (no adb needed).
4. `backendCreate`, then `deviceCreate` with an attempt chain:
   explicit-unsigned-PD → nullptr-config (canonical) → custom SOC/ARCH
   (offline-prepare style, last resort). First success wins; failures log why.
5. HTP only: **burst power config** via `deviceGetInfrastructure` →
   `createPowerConfigId` → `setPowerConfig` (DCVS off, max voltage corners,
   RPC polling 9999 µs) — without it benchmarks measure power management.
6. `contextCreateFromBinary` per model; `graphRetrieve` by name from
   introspection; `graphExecute` with `Qnn_TensorV2_t` structs that **carry
   the introspected tensor ids**.

JNI (`qnn_jni.cpp`) keeps a registry of runtimes and routes context-scoped
calls by ownership (`HasContext`), because several backends can be alive at
once during a benchmark. Buffers are direct `ByteBuffer`s; sizes are verified
against `byteSize()` before every execute.

`RunNpuSelfTest()` is a standalone bring-up matrix (every deviceCreate flavor
× context load + `deviceGetPlatformInfo` + full-verbosity log capture) used by
the Device screen — it found the real cause of every on-device failure so far.

## 3. Deployment of QNN libraries

The APK carries, in `jniLibs/arm64-v8a/`:

- aarch64 host libs: `libQnnSystem.so`, `libQnnHtp.so`, `libQnnHtpV81Stub.so`,
  `libQnnGpu.so`, `libQnnCpu.so`, `libQnnHtpPrepare.so` (DLC only), …
- **Hexagon (DSP6!) libs in the same folder**: `libQnnHtpV81Skel.so`,
  `libQnnHtpV81.so`. This mirrors Qualcomm's ChatApp. Three gradle/manifest
  switches make it work:
  - `packaging { jniLibs.useLegacyPackaging = true }` — installer extracts
    them as real files into `nativeLibraryDir` (the DSP loader needs paths,
    not APK entries);
  - `keepDebugSymbols += "**/libQnnHtpV81*.so"` — AGP's strip step only
    understands ARM ELFs;
  - `<uses-native-library libcdsprpc.so/libadsprpc.so/libOpenCL.so>` —
    on targetSdk 31+ vendor libraries are invisible to the app's linker
    namespace unless declared. Missing `libcdsprpc.so` produces
    `dlopen … not found in namespace clns-N` inside the QNN stub and a
    misleading `rc=14001` at the API surface.

`QnnRuntimeLibs.setup()` (Application.onCreate, **before** any
`System.loadLibrary`) points `ADSP_LIBRARY_PATH`, `CDSP_LIBRARY_PATH` and
`DSP_LIBRARY_PATH` at `nativeLibraryDir` plus the stock vendor fallbacks.

These libraries are **not committed** — run `scripts/copy-qnn-libs.sh` after
installing the QAIRT SDK.

## 4. Stable Diffusion pipeline

Three context binaries (text_encoder, unet, vae_decoder) + CLIP tokenizer.
All schema assumptions are pinned by tests against `qnn-context-binary-utility`
dumps of the real binaries (`app/src/test/resources/binmeta/`).

The non-obvious, verified-the-hard-way details:

- **Quantization convention**: `real = scale × (q + offset)` with *negative*
  offsets (e.g. UNet latent: scale 2.42e-4, offset −33983 → range ≈ ±8).
  `quantizeUfixed16`/`dequantizeUfixed16` in Kotlin implement exactly this.
- **The VAE divides by 0.18215 internally.** AI Hub bakes
  `z / scaling_factor` into the exported graph. Feeding pre-divided latents
  washes the image out — the input quant range (±11) is sized for raw latents.
- **Timestep grid tops out at ≈968** (`scale·65535`). diffusers' default
  linspace schedule starts at t=999 → silent clamp on the structurally most
  important step. We use the *leading* schedule `[951, 901, …, 1]`.
- **DPM-Solver++ 2M in VP parametrization** (`σ = √(1−ᾱ)`, not Karras σ/α).
  A property test drives the solver with a perfect-epsilon model and asserts
  the trajectory stays on `x_t = α_t·x₀ + σ_t·ε` exactly.
- Layout is **NHWC** on the wire; the scheduler works in CHW and transposes
  at the quantize/dequantize boundary.
- UNet inputs are resolved **by name** (timestep / latent / text_emb), not by
  position.

## 5. Whisper pipeline

Mirrors Qualcomm's `HfWhisperApp` reference loop, generalized over variants:

- **Frontend**: 1:1 Kotlin port of HF `WhisperFeatureExtractor` — periodic
  hann(400), hop 160, reflect padding, 3001→3000 frames, slaney mel filterbank
  (shipped as binary assets generated by transformers — 80 *and* 128 bins),
  `log10 → clamp(max−8) → (x+4)/4`. Golden-tested against the real extractor
  to <2e-3. The 400-point DFT is a precomputed-table matmul parallelized with
  coroutines (no FFT dependency).
- **Encoder**: `input_features [1, nMels, 3000]` fp16 → per-layer cross K/V
  caches. `nMels` is read from the encoder binary (80 = tiny…medium,
  128 = large-v3 family).
- **Decoder loop** (geometry from introspection: layer count from
  `k_cache_self_*` names, window from `attention_mask` shape):
  - prompt is **forced** to `[SOT, lang, transcribe, notimestamps]`. Greedy
    decoding from a bare SOT loves to drift into `<|en|>` or `<|translate|>`
    on short audio (Russian speech came out as English before this fix);
  - `lang` is user-pinned or detected at step 0 by argmax **restricted to the
    language-token range** — the official `detect_language` method;
  - attention mask `[1,1,1,maskLen]` initialized to −100.0 (fp16 0xD640!),
    one slot un-masked per step from the right edge;
  - self K/V caches **ping-pong**: step n's output buffers become step n+1's
    input buffers — zero copies, zero conversions;
  - v2 vs v3 vocab differences (`<|transcribe|>` 50359 vs 50360, language
    range end) are encoded in `WhisperVariant`.
- **Long audio**: split into 30 s windows (hard Whisper context), transcribed
  sequentially, text concatenated; per-chunk language detection.
- **Tokenizer**: decode-only — reverse vocab + inverse GPT-2 byte table.
  No merges needed for decoding.

## 6. Tokenizers

`ClipTokenizer` is a full byte-level BPE encoder (GPT-2 byte↔unicode table,
CLIP pre-tokenizer regex, `</w>` word-end convention, merge ranks) —
byte-exact with HF `CLIPTokenizer` on every fixture, including Cyrillic,
emoji and truncation behavior. `WhisperTokenizer` reuses the same byte table
in reverse. Fixtures are generated by `transformers` and live in test
resources, so regressions are caught against ground truth, not intuition.

## 7. Testing strategy

Three rings, none requiring the device:

1. **JVM unit tests (42)** — pure logic: tokenizers vs HF fixtures, DPM++
   exact-posterior invariant, fp16 conversion bit patterns, quantization
   round-trips with production scale/offset values, mel frontend vs golden
   HF output (80/128), chunking, catalog consistency (zip member names ↔
   extract maps — this caught `vae.bin` ≠ `vae_decoder.bin`).
2. **Schema contract tests** — unmodified `qnn-context-binary-utility` JSON
   dumps of the real `.bin` files, asserting graph names, tensor names/order/
   dtypes/dims, nonzero ids, quant ranges (timestep ≤968 documented, VAE
   expects raw latents).
3. **Host C++ test** (`tools/host_introspect`) — the APK's actual
   `binary_info_walk.h` runs on x86 against the real binaries through the
   SDK's x86 `libQnnSystem.so`.

On-device, the **NPU self-test** (Device tab) covers the last mile: namespace
visibility, FastRPC bring-up, per-flavor deviceCreate, real context loads,
with the full QNN log attached to a copyable report.

## 8. Known limitations / recorded alternatives

- QNN GPU/CPU backends can't read HTP context binaries → no on-device
  three-way benchmark with this format. Alternative (recorded, not built):
  TFLite GPU-delegate / ONNX Runtime QNN-EP builds of the same models.
- ONNX Runtime + QNN EP as a portable replacement for the raw QNN layer:
  `docs/08-alternative-onnx.md`.
- Euler sampler (what AI Hub calibrated SD against) vs our DPM++ 2M: if
  quantization artifacts ever show up, try Euler first.
- Whisper streaming, SD img2img, in-app AI Hub job submission: not built.

## 9. Repo layout

```
android-app/            Kotlin + C++ app (see §2–§6)
docs/                   RU course: 00–10 (NPU basics → custom models)
scripts/                fetch/compile/copy/benchmark/self-test scripts
tools/host_introspect/  host-side C++ test against real binaries
models/                 (gitignored) downloaded/compiled context binaries
README.ru.md            original Russian README
```
