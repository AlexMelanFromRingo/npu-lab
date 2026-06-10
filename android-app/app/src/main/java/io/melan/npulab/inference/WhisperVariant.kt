package io.melan.npulab.inference

/**
 * A deployable Whisper flavor. Geometry (layer count, KV-cache shapes, mel
 * bins, decode window) is read from the context binaries at load time — what
 * a variant pins down is the on-device directory and the tokenizer's special
 * token ids, which differ between the v2 vocab (tiny…medium, 51 865 ids) and
 * the v3 vocab (large-v3 family, 51 866 ids: +<|yue|>, shifted task tokens).
 *
 * Ids verified with HF `WhisperTokenizer` for each checkpoint.
 */
data class WhisperVariant(
    val kind: ModelKind,
    /** Directory under models/ holding encoder.bin / decoder.bin / tokenizer. */
    val dirName: String,
    val displayName: String,
    val transcribeId: Int,
    val noTimestampsId: Int,
    val langFirst: Int,
    val langLast: Int,
) {
    val encoderPath get() = "$dirName/encoder.bin"
    val decoderPath get() = "$dirName/decoder.bin"
    val vocabPath get() = "$dirName/tokenizer/vocab.json"

    fun requiredFiles(): List<String> = listOf(encoderPath, decoderPath, vocabPath)

    companion object {
        val BASE = WhisperVariant(
            kind = ModelKind.WHISPER_BASE,
            dirName = "whisper",
            displayName = "Base",
            transcribeId = 50359, noTimestampsId = 50363,
            langFirst = 50259, langLast = 50357,
        )
        val SMALL = WhisperVariant(
            kind = ModelKind.WHISPER_SMALL,
            dirName = "whisper_small",
            displayName = "Small",
            transcribeId = 50359, noTimestampsId = 50363,
            langFirst = 50259, langLast = 50357,
        )
        val LARGE_V3_TURBO = WhisperVariant(
            kind = ModelKind.WHISPER_LARGE_V3_TURBO,
            dirName = "whisper_large_v3_turbo",
            displayName = "Large-v3 Turbo",
            // v3 vocab: languages run through <|yue|>=50358, task ids shift by +1.
            transcribeId = 50360, noTimestampsId = 50364,
            langFirst = 50259, langLast = 50358,
        )
        val ALL = listOf(BASE, SMALL, LARGE_V3_TURBO)
    }
}
