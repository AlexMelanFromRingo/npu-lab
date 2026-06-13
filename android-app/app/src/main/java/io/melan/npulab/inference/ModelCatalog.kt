package io.melan.npulab.inference

import android.content.Context
import java.io.File

/**
 * Models are NOT shipped inside the APK — too large (~680 MB for SD 1.5).
 * They are downloaded on-device from the Models tab, or pushed manually into
 * Android/data/io.melan.npulab/files/models/. This catalog lists everything
 * the app knows about; [ModelStore] checks what is actually present.
 */
enum class ModelKind {
    STABLE_DIFFUSION_1_5,
    SD_TURBO,
    WHISPER_TINY,
    WHISPER_BASE,
    WHISPER_SMALL,
    WHISPER_LARGE_V3_TURBO,
    ZOO,                    // model-zoo entry: float DLC, runs on HTP/GPU/CPU
    CUSTOM,                 // user-supplied model from models/custom/
}

/** What a model does — drives which screen/pipeline can run it. */
enum class ModelCategory {
    TEXT_TO_IMAGE,
    SPEECH,
    CLASSIFICATION,
    DETECTION,
    POSE,
    DEPTH,
    SEGMENTATION,
    SUPER_RESOLUTION,
    INPAINTING,
    OTHER,
}

data class ModelAsset(
    /** Stable unique id — used as map/list key. NOT [kind] (zoo shares ZOO). */
    val id: String,
    val kind: ModelKind,
    val category: ModelCategory,
    val displayName: String,
    val description: String,
    /** File names expected inside the models directory, relative to root. */
    val expectedFiles: List<String>,
    /** Approximate inference cost per single forward pass, for scheduling. */
    val approxMs: Int,
    /** Source for in-app install, or null if only available via PC pipeline. */
    val installSource: InstallSource? = null,
) {
    /** The primary model file (.bin or .dlc) used by Benchmark/Vision. */
    val primaryModelFile: String?
        get() = expectedFiles.firstOrNull { it.endsWith(".bin") || it.endsWith(".dlc") }

    val isDlc: Boolean get() = primaryModelFile?.endsWith(".dlc") == true
}

/**
 * Download recipe for the Models tab: a zip on the public Qualcomm S3 mirror
 * (extractMap maps zip-member suffix → on-device relative path) plus optional
 * one-by-one aux files (HF tokenizers etc.).
 */
data class InstallSource(
    val zipUrl: String,
    val extractMap: Map<String, String>,
    val auxFiles: List<AuxFile> = emptyList(),
    val approxMib: Int,
)

data class AuxFile(val url: String, val destRelative: String)

object ModelCatalog {
    private const val S3 = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models"
    private const val REL = "v0.54.0"
    private const val HF_CLIP = "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main"

    private fun ctxBin(model: String, quant: String) =
        "$S3/$model/releases/$REL/$model-qnn_context_binary-$quant-qualcomm_snapdragon_8_elite_gen5.zip"

    private fun whisper(
        id: String, kind: ModelKind, dir: String, name: String, desc: String,
        hf: String, mib: Int, ms: Int,
    ) = ModelAsset(
        id = id, kind = kind, category = ModelCategory.SPEECH,
        displayName = name, description = desc,
        expectedFiles = listOf("$dir/encoder.bin", "$dir/decoder.bin", "$dir/tokenizer/vocab.json"),
        approxMs = ms,
        installSource = InstallSource(
            zipUrl = ctxBin(id, "float"),
            extractMap = mapOf("encoder.bin" to "$dir/encoder.bin", "decoder.bin" to "$dir/decoder.bin"),
            auxFiles = listOf(AuxFile("$hf/resolve/main/vocab.json", "$dir/tokenizer/vocab.json")),
            approxMib = mib,
        ),
    )

    /* ---------------------------- zoo (float DLCs) ---------------------------- */

    private data class Zoo(
        val id: String, val name: String, val task: String,
        val category: ModelCategory, val mib: Int, val ms: Int,
        /** dlc file names inside the zip; default = single "<id>.dlc". */
        val members: List<String> = emptyList(),
        /** true for ImageNet classifiers — pull labels.txt for the Vision tab. */
        val labels: Boolean = false,
    )

    private val zooEntries = listOf(
        Zoo("mobilenet_v2", "MobileNet-V2", "ImageNet classification", ModelCategory.CLASSIFICATION, 12, 4, labels = true),
        Zoo("mobilenet_v3_large", "MobileNet-V3 Large", "ImageNet classification", ModelCategory.CLASSIFICATION, 21, 4, labels = true),
        Zoo("squeezenet1_1", "SqueezeNet 1.1", "ImageNet classification", ModelCategory.CLASSIFICATION, 4, 3, labels = true),
        Zoo("shufflenet_v2", "ShuffleNet-V2", "ImageNet classification", ModelCategory.CLASSIFICATION, 4, 3, labels = true),
        Zoo("mnasnet05", "MnasNet-0.5", "ImageNet classification", ModelCategory.CLASSIFICATION, 7, 3, labels = true),
        Zoo("efficientnet_b0", "EfficientNet-B0", "ImageNet classification", ModelCategory.CLASSIFICATION, 18, 5, labels = true),
        Zoo("googlenet", "GoogLeNet", "ImageNet classification", ModelCategory.CLASSIFICATION, 23, 5, labels = true),
        Zoo("resnet18", "ResNet-18", "ImageNet classification", ModelCategory.CLASSIFICATION, 41, 6, labels = true),
        Zoo("resnet50", "ResNet-50", "ImageNet classification", ModelCategory.CLASSIFICATION, 90, 9, labels = true),
        Zoo("inception_v3", "Inception-V3", "ImageNet classification", ModelCategory.CLASSIFICATION, 84, 10, labels = true),
        Zoo("convnext_tiny", "ConvNeXt-Tiny", "ImageNet classification", ModelCategory.CLASSIFICATION, 101, 15, labels = true),
        Zoo("swin_tiny", "Swin-Tiny", "ImageNet classification (transformer)", ModelCategory.CLASSIFICATION, 100, 20, labels = true),
        Zoo("midas", "MiDaS", "Monocular depth", ModelCategory.DEPTH, 58, 15),
        Zoo("depth_anything_v2", "Depth Anything V2", "Monocular depth", ModelCategory.DEPTH, 87, 30),
        Zoo("deeplabv3_plus_mobilenet", "DeepLabV3+", "Semantic segmentation", ModelCategory.SEGMENTATION, 20, 12),
        Zoo("ffnet_40s", "FFNet-40S", "Semantic segmentation (Cityscapes)", ModelCategory.SEGMENTATION, 49, 15),
        Zoo("unet_segmentation", "U-Net", "Segmentation", ModelCategory.SEGMENTATION, 109, 30),
        Zoo("fastsam_s", "FastSAM-S", "Segment anything", ModelCategory.SEGMENTATION, 41, 25),
        Zoo("sesr_m5", "SESR-M5", "Super-resolution ×4", ModelCategory.SUPER_RESOLUTION, 1, 4),
        Zoo("xlsr", "XLSR", "Super-resolution ×4", ModelCategory.SUPER_RESOLUTION, 1, 4),
        Zoo("quicksrnetlarge", "QuickSRNet Large", "Super-resolution ×4", ModelCategory.SUPER_RESOLUTION, 1, 5),
        Zoo("real_esrgan_general_x4v3", "Real-ESRGAN general x4v3", "Super-resolution ×4", ModelCategory.SUPER_RESOLUTION, 4, 12),
        Zoo("real_esrgan_x4plus", "Real-ESRGAN x4plus", "Super-resolution ×4", ModelCategory.SUPER_RESOLUTION, 59, 70),
        Zoo("esrgan", "ESRGAN", "Super-resolution ×4", ModelCategory.SUPER_RESOLUTION, 59, 70),
        Zoo("aotgan", "AOT-GAN", "Image inpainting", ModelCategory.INPAINTING, 53, 45),
        Zoo("face_det_lite", "FaceDetLite", "Face detection", ModelCategory.DETECTION, 3, 3),
        Zoo("foot_track_net", "FootTrackNet", "Person/foot detection", ModelCategory.DETECTION, 9, 4),
        Zoo("posenet_mobilenet", "PoseNet", "Pose estimation", ModelCategory.POSE, 11, 5),
        Zoo("litehrnet", "LiteHRNet", "Pose estimation", ModelCategory.POSE, 4, 6),
        Zoo("mediapipe_face", "MediaPipe Face", "Face detection + landmarks", ModelCategory.DETECTION, 3, 3,
            members = listOf("face_detector.dlc", "face_landmark_detector.dlc")),
        Zoo("mediapipe_hand", "MediaPipe Hand", "Hand detection + landmarks", ModelCategory.DETECTION, 10, 4,
            members = listOf("hand_detector.dlc", "hand_landmark_detector.dlc")),
    )

    private val zoo: List<ModelAsset> = zooEntries.map { e ->
        val members = e.members.ifEmpty { listOf("${e.id}.dlc") }
        val extract = members.associateWith { "zoo/$it" }.toMutableMap()
        // ImageNet classifiers ship labels.txt — pull it once (all are 1000-class).
        if (e.labels) extract["labels.txt"] = "zoo/${e.id}.labels.txt"
        ModelAsset(
            id = "zoo_${e.id}",
            kind = ModelKind.ZOO,
            category = e.category,
            displayName = e.name,
            description = "${e.task} · float DLC · HTP / GPU / CPU",
            expectedFiles = members.map { "zoo/$it" },
            approxMs = e.ms,
            installSource = InstallSource(
                zipUrl = "$S3/${e.id}/releases/$REL/${e.id}-qnn_dlc-float.zip",
                extractMap = extract,
                approxMib = e.mib,
            ),
        )
    }

    /* ---------------------------- catalog ---------------------------- */

    val all: List<ModelAsset> = listOf(
        ModelAsset(
            id = "stable_diffusion_1_5",
            kind = ModelKind.STABLE_DIFFUSION_1_5,
            category = ModelCategory.TEXT_TO_IMAGE,
            displayName = "Stable Diffusion 1.5",
            description = "512×512 text→image, ~20 steps, 3-model chain. Generate tab.",
            expectedFiles = listOf(
                "sd15/text_encoder.bin", "sd15/unet.bin", "sd15/vae_decoder.bin",
                "sd15/tokenizer/vocab.json", "sd15/tokenizer/merges.txt",
            ),
            approxMs = 12_000,
            installSource = InstallSource(
                zipUrl = ctxBin("stable_diffusion_v1_5", "w8a16"),
                // Real v0.54.0 members: text_encoder.bin / unet.bin / vae.bin.
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
        whisper("whisper_tiny", ModelKind.WHISPER_TINY, "whisper_tiny", "Whisper Tiny",
            "Speech recognition, 39M. Fastest. Speech tab.",
            "https://huggingface.co/openai/whisper-tiny", 101, 800),
        whisper("whisper_base", ModelKind.WHISPER_BASE, "whisper", "Whisper Base",
            "Speech recognition, 74M. Fast, fair quality. Speech tab.",
            "https://huggingface.co/openai/whisper-base", 175, 1_500),
        whisper("whisper_small", ModelKind.WHISPER_SMALL, "whisper_small", "Whisper Small",
            "Speech recognition, 244M. Much better RU/UK. Speech tab.",
            "https://huggingface.co/openai/whisper-small", 546, 4_000),
        whisper("whisper_large_v3_turbo", ModelKind.WHISPER_LARGE_V3_TURBO, "whisper_large_v3_turbo",
            "Whisper Large-v3 Turbo", "Speech recognition, 809M, 128-mel. Best quality, ~1.9 GB. Speech tab.",
            "https://huggingface.co/openai/whisper-large-v3-turbo", 1923, 12_000),
    ) + zoo

    fun byKind(kind: ModelKind): ModelAsset = all.first { it.kind == kind }
    fun byId(id: String): ModelAsset? = all.firstOrNull { it.id == id }
}

class ModelStore(private val ctx: Context) {
    val root: File = File(ctx.getExternalFilesDir(null), "models").also { it.mkdirs() }

    fun pathOf(relative: String): File = File(root, relative)

    fun isInstalled(asset: ModelAsset): Boolean =
        asset.expectedFiles.all { pathOf(it).exists() }

    /** Ids of installed catalog assets — stable per-asset, NOT per-kind. */
    fun installedIds(): Set<String> =
        ModelCatalog.all.filter { isInstalled(it) }.map { it.id }.toSet()

    fun missingFilesFor(asset: ModelAsset): List<String> =
        asset.expectedFiles.filterNot { pathOf(it).exists() }

    /** User-supplied models dropped into models/custom/ (storage import or push). */
    fun customAssets(): List<ModelAsset> = scanCustomBins(File(root, "custom"))
}

/** Pure scanner — separated from ModelStore so JVM tests can cover it. */
internal fun scanCustomBins(customDir: File): List<ModelAsset> =
    customDir.listFiles { f ->
        f.isFile && (f.name.endsWith(".bin") || f.name.endsWith(".dlc")) && f.length() > 0
    }
        ?.sortedBy { it.name.lowercase() }
        ?.map { f ->
            val dlc = f.name.endsWith(".dlc")
            ModelAsset(
                id = "custom_${f.name}",
                kind = ModelKind.CUSTOM,
                category = ModelCategory.OTHER,
                displayName = f.name.removeSuffix(".bin").removeSuffix(".dlc"),
                description = if (dlc) "Custom DLC from models/custom/ (HTP/GPU/CPU)"
                else "Custom context binary from models/custom/ (HTP only)",
                expectedFiles = listOf("custom/${f.name}"),
                approxMs = 100,
            )
        }
        ?: emptyList()
