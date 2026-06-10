package io.melan.npulab.inference

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.float
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonObject
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Contract tests against the REAL Qualcomm AI Hub context binaries.
 *
 * The JSON fixtures in src/test/resources/binmeta/ are unmodified
 * `qnn-context-binary-utility --json_file` dumps of the .bin files the app
 * loads on the S26 Ultra. These tests pin every assumption the Kotlin
 * pipeline makes about them: target SoC/arch, graph names, tensor names,
 * order, dtypes, shapes, quantization grids, and non-zero tensor ids (the
 * id is mandatory in QnnGraph_execute — the historical id=0 bug is what
 * these guard against).
 *
 * To refresh after re-downloading models:
 *   $QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-utility \
 *     --context_binary models/sd15/unet.bin --json_file .../binmeta/sd15_unet.json
 */
class BinaryMetadataContractTest {

    private data class Tensor(
        val id: Int,
        val name: String,
        val dataType: String,
        val dims: List<Int>,
        val scale: Float?,
        val offset: Int?,
    )

    private data class Graph(
        val name: String,
        val inputs: List<Tensor>,
        val outputs: List<Tensor>,
    )

    private data class BinMeta(
        val socModel: Int,
        val dspArch: Int,
        val graphs: List<Graph>,
    )

    private fun load(resource: String): BinMeta {
        val text = javaClass.getResourceAsStream("/binmeta/$resource")!!
            .readBytes().decodeToString()
        val root = Json.parseToJsonElement(text)
        val socModel = findInt(root, "socModel")
        val dspArch = findInt(root, "dspArch")
        assertNotNull(socModel, "$resource: socModel missing")
        assertNotNull(dspArch, "$resource: dspArch missing")

        val graphsEl = findElement(root, "graphs") as? JsonArray
        assertNotNull(graphsEl, "$resource: graphs[] missing")
        val graphs = graphsEl.map { g ->
            val gi = (g.jsonObject["info"] ?: g).jsonObject
            Graph(
                name = (gi["graphName"] as JsonPrimitive).content,
                inputs = tensors(gi["graphInputs"]),
                outputs = tensors(gi["graphOutputs"]),
            )
        }
        return BinMeta(socModel!!, dspArch!!, graphs)
    }

    private fun tensors(el: JsonElement?): List<Tensor> {
        if (el !is JsonArray) return emptyList()
        return el.map { t ->
            val ti = (t.jsonObject["info"] ?: t).jsonObject
            val so = (ti["quantizeParams"] as? JsonObject)
                ?.get("scaleOffset") as? JsonObject
            Tensor(
                id = (ti["id"] as JsonPrimitive).int,
                name = (ti["name"] as JsonPrimitive).content,
                dataType = (ti["dataType"] as JsonPrimitive).content,
                dims = (ti["dimensions"] as JsonArray).map { (it as JsonPrimitive).int },
                scale = (so?.get("scale") as? JsonPrimitive)?.float,
                offset = (so?.get("offset") as? JsonPrimitive)?.int,
            )
        }
    }

    private fun findElement(el: JsonElement, key: String): JsonElement? {
        when (el) {
            is JsonObject -> {
                el[key]?.let { return it }
                for (v in el.values) findElement(v, key)?.let { return it }
            }
            is JsonArray -> for (v in el) findElement(v, key)?.let { return it }
            else -> {}
        }
        return null
    }

    private fun findInt(el: JsonElement, key: String): Int? =
        (findElement(el, key) as? JsonPrimitive)?.int

    /* ------------------------------------------------------------------ */

    @Test
    fun `all binaries target SM8850 with Hexagon V81`() {
        for (res in listOf(
            "sd15_text_encoder.json", "sd15_unet.json", "sd15_vae_decoder.json",
            "whisper_encoder.json", "whisper_decoder.json",
        )) {
            val meta = load(res)
            assertEquals(87, meta.socModel, "$res: QNN_SOC_MODEL_SM8850")
            assertEquals(81, meta.dspArch, "$res: Hexagon V81")
        }
    }

    @Test
    fun `every graph tensor carries a nonzero backend id`() {
        // QnnGraph_execute matches tensors by this id; passing 0 is the bug
        // that made the first on-device runs fail with INVALID_TENSOR.
        for (res in listOf(
            "sd15_text_encoder.json", "sd15_unet.json", "sd15_vae_decoder.json",
            "whisper_encoder.json", "whisper_decoder.json",
        )) {
            val meta = load(res)
            for (g in meta.graphs) {
                for (t in g.inputs + g.outputs) {
                    assertTrue(t.id > 0, "$res ${g.name}/${t.name}: id=${t.id}")
                }
            }
        }
    }

    @Test
    fun `text encoder schema matches pipeline expectations`() {
        val g = load("sd15_text_encoder.json").graphs.single()
        assertEquals(SdSchema.TEXT_ENCODER_GRAPH, g.name)
        assertEquals(listOf(SdSchema.IN_TOKENS), g.inputs.map { it.name })
        assertEquals("QNN_DATATYPE_INT_32", g.inputs[0].dataType)
        assertEquals(listOf(1, 77), g.inputs[0].dims)
        assertEquals(listOf(SdSchema.OUT_TEXT_EMBEDDING), g.outputs.map { it.name })
        assertEquals("QNN_DATATYPE_UFIXED_POINT_16", g.outputs[0].dataType)
        assertEquals(listOf(1, 77, 768), g.outputs[0].dims)
    }

    @Test
    fun `unet schema matches pipeline expectations`() {
        val g = load("sd15_unet.json").graphs.single()
        assertEquals(SdSchema.UNET_GRAPH, g.name)
        assertEquals(
            setOf(SdSchema.IN_TIMESTEP, SdSchema.IN_LATENT, SdSchema.IN_TEXT_EMB),
            g.inputs.map { it.name }.toSet(),
            "pipeline resolves these by name",
        )
        val latent = g.inputs.single { it.name == SdSchema.IN_LATENT }
        assertEquals(listOf(1, 64, 64, 4), latent.dims, "NHWC layout")
        assertEquals("QNN_DATATYPE_UFIXED_POINT_16", latent.dataType)

        val out = g.outputs.single()
        assertEquals(SdSchema.OUT_LATENT, out.name)
        assertEquals(listOf(1, 64, 64, 4), out.dims)

        // Every quantized tensor must expose a usable scale.
        for (t in g.inputs + g.outputs) {
            assertNotNull(t.scale, "${t.name} quant scale")
            assertTrue(t.scale!! > 0f)
        }
    }

    @Test
    fun `unet timestep grid covers the leading schedule but not linspace`() {
        val g = load("sd15_unet.json").graphs.single()
        val t = g.inputs.single { it.name == SdSchema.IN_TIMESTEP }
        val maxRepresentable = t.scale!! * (0xFFFF + (t.offset ?: 0))
        assertTrue(
            maxRepresentable >= SdSchema.MAX_SCHEDULER_TIMESTEP,
            "grid max $maxRepresentable must cover leading max " +
                SdSchema.MAX_SCHEDULER_TIMESTEP,
        )
        assertTrue(
            maxRepresentable < 999f,
            "if this starts covering 999, the leading-spacing constraint can be lifted",
        )
    }

    @Test
    fun `vae expects RAW latents — quant range matches undivided latent magnitudes`() {
        val g = load("sd15_vae_decoder.json").graphs.single()
        assertEquals(SdSchema.VAE_GRAPH, g.name)
        val latent = g.inputs.single()
        assertEquals(SdSchema.IN_LATENT, latent.name)
        val lo = latent.scale!! * (latent.offset ?: 0)
        val hi = latent.scale!! * (0xFFFF + (latent.offset ?: 0))
        // Raw SD latents live within roughly ±11. Latents pre-divided by
        // 0.18215 would span ±60 — far outside this grid. This pins the
        // "no double division before VAE" fix.
        assertTrue(lo > -20f && lo < -6f, "low end $lo sized for raw latents")
        assertTrue(hi > 6f && hi < 20f, "high end $hi sized for raw latents")

        val image = g.outputs.single()
        assertEquals(SdSchema.OUT_IMAGE, image.name)
        assertEquals(listOf(1, 512, 512, 3), image.dims)
        val imgLo = image.scale!! * (image.offset ?: 0)
        val imgHi = image.scale!! * (0xFFFF + (image.offset ?: 0))
        assertEquals(0f, imgLo, 1e-4f, "image range starts at 0")
        assertEquals(1f, imgHi, 1e-2f, "image range ends at 1")
    }

    @Test
    fun `whisper graphs expose the expected interfaces`() {
        val enc = load("whisper_encoder.json").graphs.single()
        assertEquals("whisper_base_encoder", enc.name)
        assertEquals(listOf("input_features"), enc.inputs.map { it.name })
        assertEquals(listOf(1, 80, 3000), enc.inputs[0].dims)
        assertEquals(12, enc.outputs.size, "6 layers × cross K/V")

        val dec = load("whisper_decoder.json").graphs.single()
        assertEquals("whisper_base_decoder", dec.name)
        assertEquals(27, dec.inputs.size)
        assertEquals(13, dec.outputs.size)
        assertTrue(dec.outputs.any { it.name == "logits" })
    }
}
