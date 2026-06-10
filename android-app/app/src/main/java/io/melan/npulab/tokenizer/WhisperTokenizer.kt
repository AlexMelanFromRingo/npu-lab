package io.melan.npulab.tokenizer

import java.io.File

/**
 * Decode-only tokenizer for Whisper (multilingual GPT-2-style byte-level BPE,
 * 51 865 ids). Decoding needs no merge ranks — just the reverse vocab plus the
 * byte-level un-mapping, so vocab.json is the only required file.
 *
 * Special ids (≥ [FIRST_SPECIAL]) — <|startoftranscript|>, language tags,
 * task tags, timestamps — are skipped during decode, mirroring
 * `tokenizer.decode(…, skip_special_tokens=True)` in the reference app.
 */
class WhisperTokenizer(vocabFile: File) {

    private val idToToken: Array<String?>

    init {
        val vocab = VocabParser.parse(vocabFile.readText())
        val maxId = vocab.values.max()
        idToToken = arrayOfNulls(maxId + 1)
        for ((tok, id) in vocab) idToToken[id] = tok
    }

    /** Decode token ids to text, skipping special tokens. */
    fun decode(ids: List<Int>): String {
        val bytes = ArrayList<Byte>(ids.size * 3)
        for (id in ids) {
            if (id >= FIRST_SPECIAL) continue
            val tok = idToToken.getOrNull(id) ?: continue
            for (ch in tok) {
                val b = BYTE_DECODER[ch.code]
                if (b >= 0) bytes.add(b.toByte())
            }
        }
        return String(bytes.toByteArray(), Charsets.UTF_8)
    }

    companion object {
        /** <|endoftext|> — everything from here up is special. */
        const val FIRST_SPECIAL = 50257
        const val EOT = 50257
        const val SOT = 50258               // <|startoftranscript|>
        const val TRANSCRIBE = 50359        // <|transcribe|>
        const val TRANSLATE = 50358         // <|translate|>
        const val NO_TIMESTAMPS = 50363     // <|notimestamps|>

        /** Language token range: <|en|>=50259 … <|su|>=50357 (99 languages). */
        const val LANG_FIRST = 50259
        const val LANG_LAST = 50357

        /** Ids verified against HF WhisperTokenizer (openai/whisper-base). */
        val LANGUAGE_IDS: Map<String, Int> = mapOf(
            "en" to 50259, "de" to 50261, "es" to 50262, "ru" to 50263,
            "fr" to 50265, "uk" to 50280,
        )

        /**
         * Inverse of the GPT-2 byte→unicode table (see ClipTokenizer.BYTE_ENCODER
         * — Whisper uses the same mapping). Index = mapped char code, value =
         * original byte or -1.
         */
        internal val BYTE_DECODER: IntArray = run {
            val dec = IntArray(512) { -1 }
            val enc = ClipTokenizer.BYTE_ENCODER
            for (b in 0..255) dec[enc[b].code] = b
            dec
        }
    }
}

/**
 * Shared flat {"token": id} JSON parser (same format for CLIP and Whisper
 * vocab.json files).
 */
internal object VocabParser {
    fun parse(json: String): Map<String, Int> {
        val out = HashMap<String, Int>(60_000)
        var i = json.indexOf('{')
        require(i >= 0) { "vocab.json: missing {" }
        i++
        while (i < json.length) {
            while (i < json.length && (json[i].isWhitespace() || json[i] == ',')) i++
            if (i >= json.length || json[i] == '}') break
            require(json[i] == '"') { "vocab.json: expected string key at $i" }
            i++
            val keyBuilder = StringBuilder()
            while (i < json.length && json[i] != '"') {
                if (json[i] == '\\') {
                    when (val c = json[i + 1]) {
                        '"', '\\', '/' -> { keyBuilder.append(c); i += 2 }
                        'n' -> { keyBuilder.append('\n'); i += 2 }
                        't' -> { keyBuilder.append('\t'); i += 2 }
                        'r' -> { keyBuilder.append('\r'); i += 2 }
                        'b' -> { keyBuilder.append('\b'); i += 2 }
                        'u' -> {
                            val cp = json.substring(i + 2, i + 6).toInt(16)
                            keyBuilder.append(cp.toChar())
                            i += 6
                        }
                        else -> { keyBuilder.append(c); i += 2 }
                    }
                } else {
                    keyBuilder.append(json[i]); i++
                }
            }
            i++ // closing quote
            while (i < json.length && (json[i].isWhitespace() || json[i] == ':')) i++
            val numStart = i
            while (i < json.length && (json[i].isDigit() || json[i] == '-')) i++
            out[keyBuilder.toString()] = json.substring(numStart, i).toInt()
        }
        return out
    }
}
