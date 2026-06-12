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
    WHISPER_TINY,           // speech recognition, 39M — fastest
    WHISPER_BASE,           // speech recognition, 74M
    WHISPER_SMALL,          // speech recognition, 244M — better RU/UK quality
    WHISPER_LARGE_V3_TURBO, // speech recognition, 809M — best quality, v3 vocab
    ZOO,                    // model-zoo entry: float DLC, runs on HTP/GPU/CPU
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
    private const val WHISPER_TINY_ZIP = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/" +
        "qai-hub-models/models/whisper_tiny/releases/v0.54.0/" +
        "whisper_tiny-qnn_context_binary-float-qualcomm_snapdragon_8_elite_gen5.zip"
    private const val WHISPER_SMALL_ZIP = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/" +
        "qai-hub-models/models/whisper_small/releases/v0.54.0/" +
        "whisper_small-qnn_context_binary-float-qualcomm_snapdragon_8_elite_gen5.zip"
    private const val WHISPER_TURBO_ZIP = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/" +
        "qai-hub-models/models/whisper_large_v3_turbo/releases/v0.54.0/" +
        "whisper_large_v3_turbo-qnn_context_binary-float-qualcomm_snapdragon_8_elite_gen5.zip"
    private const val HF_CLIP = "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main"

    /**
     * Model zoo: device-agnostic float DLCs from the public S3 mirror. One
     * file per model, composed on-device for whichever backend you pick —
     * these are the entries that make the GPU/CPU benchmark buttons real.
     * Availability verified against the v0.54.0 release bucket.
     */
    private data class Zoo(
        val id: String,
        val name: String,
        val task: String,
        val mib: Int,
        val ms: Int,
        /** dlc file names inside the zip; default = single "<id>.dlc". */
        val members: List<String> = emptyList(),
    )

    private val zooEntries = listOf(
        Zoo("mobilenet_v2", "MobileNet-V2", "ImageNet classification", 12, 4),
        Zoo("mobilenet_v3_large", "MobileNet-V3 Large", "ImageNet classification", 21, 4),
        Zoo("squeezenet1_1", "SqueezeNet 1.1", "ImageNet classification", 4, 3),
        Zoo("shufflenet_v2", "ShuffleNet-V2", "ImageNet classification", 4, 3),
        Zoo("mnasnet05", "MnasNet-0.5", "ImageNet classification", 7, 3),
        Zoo("efficientnet_b0", "EfficientNet-B0", "ImageNet classification", 18, 5),
        Zoo("googlenet", "GoogLeNet", "ImageNet classification", 23, 5),
        Zoo("resnet18", "ResNet-18", "ImageNet classification", 41, 6),
        Zoo("resnet50", "ResNet-50", "ImageNet classification", 90, 9),
        Zoo("inception_v3", "Inception-V3", "ImageNet classification", 84, 10),
        Zoo("convnext_tiny", "ConvNeXt-Tiny", "ImageNet classification", 101, 15),
        Zoo("swin_tiny", "Swin-Tiny", "ImageNet classification (transformer)", 100, 20),
        Zoo("face_det_lite", "FaceDetLite", "Face detection", 3, 3),
        Zoo("foot_track_net", "FootTrackNet", "Person/foot detection", 9, 4),
        Zoo("posenet_mobilenet", "PoseNet", "Pose estimation", 11, 5),
        Zoo("litehrnet", "LiteHRNet", "Pose estimation", 4, 6),
        Zoo("mediapipe_face", "MediaPipe Face", "Face detection + landmarks", 3, 3,
            members = listOf("face_detector.dlc", "face_landmark_detector.dlc")),
        Zoo("mediapipe_hand", "MediaPipe Hand", "Hand detection + landmarks", 10, 4,
            members = listOf("hand_detector.dlc", "hand_landmark_detector.dlc")),
        Zoo("midas", "MiDaS", "Monocular depth", 58, 15),
        Zoo("depth_anything_v2", "Depth Anything V2", "Monocular depth", 87, 30),
        Zoo("deeplabv3_plus_mobilenet", "DeepLabV3+", "Semantic segmentation", 20, 12),
        Zoo("ffnet_40s", "FFNet-40S", "Semantic segmentation (Cityscapes)", 49, 15),
        Zoo("fastsam_s", "FastSAM-S", "Segment anything", 41, 25),
        Zoo("unet_segmentation", "U-Net", "Segmentation", 109, 30),
        Zoo("sesr_m5", "SESR-M5", "Super-resolution ×4", 1, 4),
        Zoo("xlsr", "XLSR", "Super-resolution ×4", 1, 4),
        Zoo("quicksrnetlarge", "QuickSRNet Large", "Super-resolution ×4", 1, 5),
        Zoo("real_esrgan_general_x4v3", "Real-ESRGAN general x4v3", "Super-resolution ×4", 4, 12),
        Zoo("real_esrgan_x4plus", "Real-ESRGAN x4plus", "Super-resolution ×4", 59, 70),
        Zoo("esrgan", "ESRGAN", "Super-resolution ×4", 59, 70),
        Zoo("aotgan", "AOT-GAN", "Image inpainting", 53, 45),
    )

    private val zoo: List<ModelAsset> = zooEntries.map { e ->
        val members = e.members.ifEmpty { listOf("${e.id}.dlc") }
        ModelAsset(
            kind = ModelKind.ZOO,
            displayName = e.name,
            description = "${e.task} · float DLC · runs on HTP / GPU / CPU",
            expectedFiles = members.map { "zoo/$it" },
            approxMs = e.ms,
            installSource = InstallSource(
                zipUrl = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/" +
                    "qai-hub-models/models/${e.id}/releases/v0.54.0/${e.id}-qnn_dlc-float.zip",
                extractMap = members.associateWith { "zoo/$it" },
                approxMib = e.mib,
            ),
        )
    }


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
            kind = ModelKind.WHISPER_TINY,
            displayName = "Whisper Tiny",
            description = "Speech recognition, 39M params. Fastest, lowest quality. Speech tab.",
            expectedFiles = listOf(
                "whisper_tiny/encoder.bin",
                "whisper_tiny/decoder.bin",
                "whisper_tiny/tokenizer/vocab.json",
            ),
            approxMs = 800,
            installSource = InstallSource(
                zipUrl = WHISPER_TINY_ZIP,
                extractMap = mapOf(
                    "encoder.bin" to "whisper_tiny/encoder.bin",
                    "decoder.bin" to "whisper_tiny/decoder.bin",
                ),
                auxFiles = listOf(
                    AuxFile(
                        "https://huggingface.co/openai/whisper-tiny/resolve/main/vocab.json",
                        "whisper_tiny/tokenizer/vocab.json",
                    ),
                ),
                approxMib = 101,
            ),
        ),
    ) + zoo

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
    customDir.listFiles { f ->
        f.isFile && (f.name.endsWith(".bin") || f.name.endsWith(".dlc")) && f.length() > 0
    }
        ?.sortedBy { it.name.lowercase() }
        ?.map { f ->
            ModelAsset(
                kind = ModelKind.CUSTOM,
                displayName = f.name.removeSuffix(".bin").removeSuffix(".dlc"),
                description = if (f.name.endsWith(".dlc"))
                    "Custom DLC from models/custom/ (runs on HTP/GPU/CPU)"
                else
                    "Custom context binary from models/custom/ (HTP only)",
                expectedFiles = listOf("custom/${f.name}"),
                approxMs = 100,
            )
        }
        ?: emptyList()
