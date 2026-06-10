package io.melan.npulab.inference

import org.junit.Test
import kotlin.test.assertTrue

class ModelCatalogTest {

    @Test
    fun `every extractMap destination is an expected file`() {
        for (asset in ModelCatalog.all) {
            val source = asset.installSource ?: continue
            for ((suffix, dest) in source.extractMap) {
                assertTrue(
                    dest in asset.expectedFiles,
                    "${asset.kind}: extractMap '$suffix' → '$dest' not in expectedFiles",
                )
            }
        }
    }

    @Test
    fun `every aux file destination is an expected file`() {
        for (asset in ModelCatalog.all) {
            val source = asset.installSource ?: continue
            for (aux in source.auxFiles) {
                assertTrue(
                    aux.destRelative in asset.expectedFiles,
                    "${asset.kind}: aux '${aux.url}' → '${aux.destRelative}' not in expectedFiles",
                )
            }
        }
    }

    @Test
    fun `sd15 covers the real zip member names`() {
        // Verified against the actual v0.54.0 S3 archive: members are
        // text_encoder.bin, unet.bin and *vae.bin* (not vae_decoder.bin).
        val sd = ModelCatalog.byKind(ModelKind.STABLE_DIFFUSION_1_5)
        val suffixes = sd.installSource!!.extractMap.keys
        for (member in listOf("text_encoder.bin", "unet.bin", "vae.bin")) {
            assertTrue(
                suffixes.any { member.endsWith(it) },
                "zip member '$member' would not be extracted (suffixes: $suffixes)",
            )
        }
    }

    @Test
    fun `installable assets reference every bin they need`() {
        for (asset in ModelCatalog.all) {
            val source = asset.installSource ?: continue
            val binsExpected = asset.expectedFiles.filter { it.endsWith(".bin") }
            val binsProvided = source.extractMap.values.toSet()
            for (bin in binsExpected) {
                assertTrue(
                    bin in binsProvided,
                    "${asset.kind}: '$bin' expected on device but no zip member maps to it",
                )
            }
        }
    }
}
