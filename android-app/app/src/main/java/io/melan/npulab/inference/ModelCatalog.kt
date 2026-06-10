package io.melan.npulab.inference

import android.content.Context
import java.io.File

/**
 * Models are NOT shipped inside the APK — they are too large (~1.5 GB for SD 1.5).
 * Instead, the user runs `scripts/fetch-models.py` on their PC to download
 * pre-compiled QNN context binaries from Qualcomm AI Hub, then `adb push`es
 * them to /sdcard/Android/data/io.melan.npulab/files/models/ on the device.
 *
 * This catalog lists every model the app knows about, and ModelStore checks
 * which ones are actually present on the device.
 */
enum class ModelKind {
    STABLE_DIFFUSION_1_5,   // text→image, multi-stage pipeline
    SD_TURBO,               // 1-4 step fast variant
    WHISPER_BASE,           // speech recognition, 74M
    WHISPER_SMALL,          // speech recognition, 244M — better RU/UK quality
    WHISPER_LARGE_V3_TURBO, // speech recognition, 809M — best quality, v3 vocab
    REAL_ESRGAN_X4,         // 4× super-resolution
    MOBILENET_V3,           // image classification (cheap baseline)
    CUSTOM,                 // user-compiled context binary from models/custom/
}

data class ModelAsset(
    val kind: ModelKind,
    val displayName: String,
    val description: String,
    /** File names expected inside the models directory, relative to root. */
    val expectedFiles: List<String>,
    /** Approximate inference cost per single forward pass, for scheduling. */
    val approxMs: Int,
    /** Source for in-app install, or null if only available via PC compile pipeline. */
    val installSource: InstallSource? = null,
)

/**
 * Where the app pulls the .bin files from when the user taps Install in the
 * Catalog screen. We currently support a single download case: a zip on the
 * public Qualcomm S3 mirror with extractMap mapping zip entries → on-device
 * relative paths (under getExternalFilesDir/models/).
 */
data class InstallSource(
    /** Direct zip URL on Qualcomm public S3 bucket. */
    val zipUrl: String,
    /** Map: zip member filename suffix → on-device destination relative path. */
    val extractMap: Map<String, String>,
    /** Optional extra files to fetch one-by-one (e.g. HF tokenizer json/txt). */
    val auxFiles: List<AuxFile> = emptyList(),
    /** Estimated total download size in MiB, for UI hint before fetch. */
    val approxMib: Int,
)

data class AuxFile(val url: String, val destRelative: String)

object ModelCatalog {
    private const val SD15_ZIP = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/" +
        "qai-hub-models/models/stable_diffusion_v1_5/releases/v0.54.0/" +
        "stable_diffusion_v1_5-qnn_context_binary-w8a16-qualcomm_snapdragon_8_elite_gen5.zip"
    private const val WHISPER_ZIP = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/" +
        "qai-hub-models/models/whisper_base/releases/v0.54.0/" +
        "whisper_base-qnn_context_binary-float-qualcomm_snapdragon_8_elite_gen5.zip"
    private const val WHISPER_SMALL_ZIP = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/" +
        "qai-hub-models/models/whisper_small/releases/v0.54.0/" +
        "whisper_small-qnn_context_binary-float-qualcomm_snapdragon_8_elite_gen5.zip"
    private const val WHISPER_TURBO_ZIP = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/" +
        "qai-hub-models/models/whisper_large_v3_turbo/releases/v0.54.0/" +
        "whisper_large_v3_turbo-qnn_context_binary-float-qualcomm_snapdragon_8_elite_gen5.zip"
    private const val HF_CLIP = "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main"

    val all: List<ModelAsset> = listOf(
        ModelAsset(
            kind = ModelKind.STABLE_DIFFUSION_1_5,
            displayName = "Stable Diffusion 1.5",
            description = "512×512 text→image, ~20 DPM++ steps, 3-model chain. Generate tab.",
            expectedFiles = listOf(
                "sd15/text_encoder.bin",
                "sd15/unet.bin",
                "sd15/vae_decoder.bin",
                "sd15/tokenizer/vocab.json",
                "sd15/tokenizer/merges.txt",
            ),
            approxMs = 12_000,
            installSource = InstallSource(
                zipUrl = SD15_ZIP,
                // Zip member names verified against the real v0.54.0 archive:
                // text_encoder.bin / unet.bin / vae.bin (NOT vae_decoder.bin!)
                // + metadata.json which we skip. Extra key kept in case a
                // future release renames vae.bin back.
                extractMap = mapOf(
                    "text_encoder.bin" to "sd15/text_encoder.bin",
                    "unet.bin" to "sd15/unet.bin",
                    "vae.bin" to "sd15/vae_decoder.bin",
                    "vae_decoder.bin" to "sd15/vae_decoder.bin",
                ),
                auxFiles = listOf(
                    AuxFile("$HF_CLIP/vocab.json", "sd15/tokenizer/vocab.json"),
                    AuxFile("$HF_CLIP/merges.txt", "sd15/tokenizer/merges.txt"),
                ),
                approxMib = 680,
            ),
        ),
        ModelAsset(
            kind = ModelKind.SD_TURBO,
            displayName = "SD Turbo",
            description = "Distilled 1–4 step variant, ~3 s per image.",
            expectedFiles = listOf(
                "sd_turbo/text_encoder.bin",
                "sd_turbo/unet.bin",
                "sd_turbo/vae_decoder.bin",
                "sd_turbo/tokenizer/vocab.json",
                "sd_turbo/tokenizer/merges.txt",
            ),
            approxMs = 3_000,
            installSource = null,  // not on public S3 yet — needs AI Hub compile
        ),
        ModelAsset(
            kind = ModelKind.WHISPER_BASE,
            displayName = "Whisper Base",
            description = "Speech recognition, 74M params. Fastest; fair quality. Speech tab.",
            expectedFiles = listOf(
                "whisper/encoder.bin",
                "whisper/decoder.bin",
                "whisper/tokenizer/vocab.json",
            ),
            approxMs = 1_500,
            installSource = InstallSource(
                zipUrl = WHISPER_ZIP,
                extractMap = mapOf(
                    "encoder.bin" to "whisper/encoder.bin",
                    "decoder.bin" to "whisper/decoder.bin",
                ),
                auxFiles = listOf(
                    AuxFile(
                        "https://huggingface.co/openai/whisper-base/resolve/main/vocab.json",
                        "whisper/tokenizer/vocab.json",
                    ),
                ),
                approxMib = 175,
            ),
        ),
        ModelAsset(
            kind = ModelKind.WHISPER_SMALL,
            displayName = "Whisper Small",
            description = "Speech recognition, 244M params. Much better RU/UK quality. Speech tab.",
            expectedFiles = listOf(
                "whisper_small/encoder.bin",
                "whisper_small/decoder.bin",
                "whisper_small/tokenizer/vocab.json",
            ),
            approxMs = 4_000,
            installSource = InstallSource(
                zipUrl = WHISPER_SMALL_ZIP,
                extractMap = mapOf(
                    "encoder.bin" to "whisper_small/encoder.bin",
                    "decoder.bin" to "whisper_small/decoder.bin",
                ),
                auxFiles = listOf(
                    AuxFile(
                        "https://huggingface.co/openai/whisper-small/resolve/main/vocab.json",
                        "whisper_small/tokenizer/vocab.json",
                    ),
                ),
                approxMib = 546,
            ),
        ),
        ModelAsset(
            kind = ModelKind.WHISPER_LARGE_V3_TURBO,
            displayName = "Whisper Large-v3 Turbo",
            description = "Speech recognition, 809M params, 128-mel. Best quality; ~1.9 GB. Speech tab.",
            expectedFiles = listOf(
                "whisper_large_v3_turbo/encoder.bin",
                "whisper_large_v3_turbo/decoder.bin",
                "whisper_large_v3_turbo/tokenizer/vocab.json",
            ),
            approxMs = 12_000,
            installSource = InstallSource(
                zipUrl = WHISPER_TURBO_ZIP,
                extractMap = mapOf(
                    "encoder.bin" to "whisper_large_v3_turbo/encoder.bin",
                    "decoder.bin" to "whisper_large_v3_turbo/decoder.bin",
                ),
                auxFiles = listOf(
                    AuxFile(
                        "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/vocab.json",
                        "whisper_large_v3_turbo/tokenizer/vocab.json",
                    ),
                ),
                approxMib = 1923,
            ),
        ),
        ModelAsset(
            kind = ModelKind.REAL_ESRGAN_X4,
            displayName = "Real-ESRGAN x4",
            description = "4× super-resolution 256→1024. Public S3 has DLC only — needs AI Hub.",
            expectedFiles = listOf("real_esrgan_x4.bin"),
            approxMs = 600,
            installSource = null,
        ),
        ModelAsset(
            kind = ModelKind.MOBILENET_V3,
            displayName = "MobileNet-V3 Large",
            description = "ImageNet-1k classification. Public S3 has DLC only — needs AI Hub.",
            expectedFiles = listOf("mobilenet_v3.bin"),
            approxMs = 8,
            installSource = null,
        ),
    )

    fun byKind(kind: ModelKind): ModelAsset = all.first { it.kind == kind }
}

class ModelStore(private val ctx: Context) {
    val root: File = File(ctx.getExternalFilesDir(null), "models").also { it.mkdirs() }

    fun pathOf(relative: String): File = File(root, relative)

    fun isInstalled(asset: ModelAsset): Boolean =
        asset.expectedFiles.all { pathOf(it).exists() }

    fun installedKinds(): Set<ModelKind> =
        ModelCatalog.all.filter { isInstalled(it) }.map { it.kind }.toSet()

    fun missingFilesFor(asset: ModelAsset): List<String> =
        asset.expectedFiles.filterNot { pathOf(it).exists() }

    /**
     * User-compiled context binaries dropped into models/custom/ (via
     * scripts/compile-model.py + adb push, or any other way). Each .bin
     * becomes a benchmarkable [ModelAsset] — no catalog entry needed.
     */
    fun customAssets(): List<ModelAsset> = scanCustomBins(File(root, "custom"))
}

/** Pure scanner — separated from ModelStore so JVM tests can cover it. */
internal fun scanCustomBins(customDir: File): List<ModelAsset> =
    customDir.listFiles { f -> f.isFile && f.name.endsWith(".bin") && f.length() > 0 }
        ?.sortedBy { it.name.lowercase() }
        ?.map { f ->
            ModelAsset(
                kind = ModelKind.CUSTOM,
                displayName = f.name.removeSuffix(".bin"),
                description = "Custom context binary from models/custom/",
                expectedFiles = listOf("custom/${f.name}"),
                approxMs = 100,
            )
        }
        ?: emptyList()
