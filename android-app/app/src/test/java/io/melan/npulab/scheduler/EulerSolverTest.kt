package io.melan.npulab.scheduler

import org.junit.Test
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class EulerSolverTest {

    @Test
    fun `timesteps are linspace 999 to 0 and init sigma matches SD 1_5`() {
        val s = EulerDiscreteSolver(numInferenceSteps = 20)
        assertEquals(999f, s.timesteps.first(), 1e-3f)
        assertEquals(0f, s.timesteps.last(), 1e-3f)
        // Strictly descending
        for (i in 1 until s.timesteps.size) {
            assertTrue(s.timesteps[i] < s.timesteps[i - 1])
        }
        // SD 1.5 reference: sigma_max ≈ 14.61, terminal sigma 0 appended.
        assertEquals(14.61f, s.initNoiseSigma, 0.05f)
        assertEquals(0f, s.sigmas.last())
        assertEquals(21, s.sigmas.size)
    }

    /**
     * Exactness invariant: with a model that returns the TRUE epsilon of a
     * fixed (x0, eps) pair, the Euler trajectory must stay exactly on
     * x_i = x0 + sigma_i * eps and finish at exactly x0 (sigma_terminal = 0).
     */
    @Test
    fun `perfect eps model lands exactly on x0`() {
        val steps = 20
        val s = EulerDiscreteSolver(numInferenceSteps = steps)
        val n = 64
        val rng = Random(7)
        val x0 = FloatArray(n) { rng.nextFloat() * 2f - 1f }
        val eps = FloatArray(n) { gaussian(rng) }

        var x = FloatArray(n) { i -> x0[i] + s.sigmas[0] * eps[i] }
        for (i in 0 until steps) {
            // True epsilon at the current state — what a perfect UNet returns.
            val sigma = s.sigmas[i]
            val pred = FloatArray(n) { j -> (x[j] - x0[j]) / sigma }
            x = s.step(i, x, pred)
            val sNext = s.sigmas[i + 1]
            for (j in 0 until n) {
                val want = x0[j] + sNext * eps[j]
                assertTrue(
                    abs(x[j] - want) < 1e-3f,
                    "step $i: x=$j ${x[j]} want $want",
                )
            }
        }
        for (j in 0 until n) {
            assertTrue(abs(x[j] - x0[j]) < 1e-3f, "final must equal x0")
        }
    }

    @Test
    fun `scale model input matches the VP transform`() {
        val s = EulerDiscreteSolver(numInferenceSteps = 20)
        val x = FloatArray(8) { 2f }
        val scaled = s.scaleModelInput(x, 0)
        val expect = 2f / sqrt(s.sigmas[0] * s.sigmas[0] + 1f)
        for (v in scaled) assertEquals(expect, v, 1e-5f)
        // At the last real step sigma is small → scaling ≈ 1.
        val nearOne = s.scaleModelInput(x, 19)
        assertTrue(abs(nearOne[0] - 2f) < 0.05f)
    }

    @Test
    fun `initial latents scale with init noise sigma`() {
        val s = EulerDiscreteSolver(numInferenceSteps = 20)
        val lat = s.initialLatents(4 * 64 * 64, seed = 42)
        var acc = 0.0
        for (v in lat) acc += v.toDouble() * v
        val std = sqrt(acc / lat.size).toFloat()
        assertEquals(s.initNoiseSigma, std, s.initNoiseSigma * 0.05f)
        // determinism
        val again = s.initialLatents(4 * 64 * 64, seed = 42)
        assertTrue(lat.contentEquals(again))
    }

    private fun gaussian(rng: Random): Float {
        var u1: Float
        do { u1 = rng.nextFloat() } while (u1 == 0f)
        val u2 = rng.nextFloat()
        return (sqrt(-2f * kotlin.math.ln(u1)) *
            kotlin.math.cos(2f * Math.PI.toFloat() * u2))
    }
}
