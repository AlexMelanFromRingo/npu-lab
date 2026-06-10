package io.melan.npulab.scheduler

import org.junit.Test
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DpmSolverTest {

    @Test
    fun `timesteps use leading spacing with offset 1`() {
        val s = DpmSolverMultistep(numInferenceSteps = 20)
        // Diffusers "leading": (arange(20) * 50)[::-1] + 1 → 951, 901, …, 51, 1.
        val expected = IntArray(20) { (19 - it) * 50 + 1 }
        assertContentEquals(expected, s.timesteps)
        // Max timestep must stay inside the AI Hub UNet's quantized `timestep`
        // grid (scale*65535 ≈ 968): linspace's 999 would clamp, 951 must not.
        assertTrue(s.timesteps.first() <= 951)
        assertEquals(1, s.timesteps.last())
    }

    @Test
    fun `beta schedule matches SD scaled_linear endpoints`() {
        val s = DpmSolverMultistep()
        // alphas_cumprod[0] = 1 - beta_0 = 1 - 0.00085
        assertEquals(1f - 0.00085f, s.alphasCumprod[0], 1e-6f)
        // alphas_cumprod must be strictly decreasing into (0, 1)
        for (i in 1 until s.alphasCumprod.size) {
            assertTrue(s.alphasCumprod[i] < s.alphasCumprod[i - 1])
            assertTrue(s.alphasCumprod[i] > 0f)
        }
        // SD 1.5 reference value: alphas_cumprod[999] ≈ 0.004682
        assertEquals(0.004682f, s.alphasCumprod[999], 5e-4f)
    }

    /**
     * Exactness invariant of DPM-Solver++ in VP parametrization: if the model
     * predicts the TRUE epsilon of a fixed (x0, eps) pair at every step, the
     * trajectory must stay exactly on x_t = alpha_t * x0 + sigma_t * eps and
     * finish at exactly x0.
     *
     * This catches the historical bug where Karras sigmas (sigma/alpha) were
     * used inside the VP-form update — that skews the noise term by
     * alpha_t/alpha_next per step and the trajectory leaves the posterior path.
     */
    @Test
    fun `perfect eps model keeps trajectory on the exact posterior path`() {
        val steps = 20
        val s = DpmSolverMultistep(numInferenceSteps = steps)
        val n = 64
        val rng = Random(42)
        val x0 = FloatArray(n) { rng.nextFloat() * 2f - 1f }
        val eps = FloatArray(n) { gaussian(rng) }

        fun alpha(t: Int) = sqrt(s.alphasCumprod[t])
        fun sigma(t: Int) = sqrt(1f - s.alphasCumprod[t])

        val t0 = s.timesteps[0]
        var x = FloatArray(n) { i -> alpha(t0) * x0[i] + sigma(t0) * eps[i] }

        for (step in 0 until steps) {
            val t = s.timesteps[step]
            // True epsilon for the current state — what a perfect UNet returns.
            val noisePred = FloatArray(n) { i -> (x[i] - alpha(t) * x0[i]) / sigma(t) }
            x = s.step(step, x, noisePred)
            if (step < steps - 1) {
                val tNext = s.timesteps[step + 1]
                for (i in 0 until n) {
                    val want = alpha(tNext) * x0[i] + sigma(tNext) * eps[i]
                    assertTrue(
                        abs(x[i] - want) < 2e-3f,
                        "step $step: x[$i]=${x[i]} want $want (drift off posterior path)",
                    )
                }
            }
        }
        for (i in 0 until n) {
            assertTrue(abs(x[i] - x0[i]) < 2e-3f, "final latent must equal x0")
        }
    }

    @Test
    fun `initial latents are deterministic per seed and standard normal`() {
        val s = DpmSolverMultistep()
        val shape = intArrayOf(4, 64, 64)
        val a = s.initialLatents(shape, seed = 123)
        val b = s.initialLatents(shape, seed = 123)
        val c = s.initialLatents(shape, seed = 124)
        assertContentEquals(a, b)
        assertTrue(!a.contentEquals(c))
        assertEquals(4 * 64 * 64, a.size)

        val mean = a.sum() / a.size
        var varAcc = 0.0
        for (v in a) varAcc += (v - mean) * (v - mean)
        val std = sqrt(varAcc / a.size).toFloat()
        assertTrue(abs(mean) < 0.03f, "mean=$mean")
        assertTrue(abs(std - 1f) < 0.03f, "std=$std")
    }

    @Test
    fun `reset clears multistep history`() {
        val x = FloatArray(8) { 0.5f }
        val epsA = FloatArray(8) { 0.1f }
        val epsB = FloatArray(8) { 0.7f }

        // Without reset: step 1 takes the 2M branch (uses step-0 history).
        val sKeep = DpmSolverMultistep(numInferenceSteps = 4)
        sKeep.step(0, x, epsA)
        val withHistory = sKeep.step(1, x, epsB)

        // With reset: history dropped → step 1 falls back to first-order,
        // identical to a fresh solver that never saw step 0.
        val sReset = DpmSolverMultistep(numInferenceSteps = 4)
        sReset.step(0, x, epsA)
        sReset.reset()
        val afterReset = sReset.step(1, x, epsB)

        val sFresh = DpmSolverMultistep(numInferenceSteps = 4)
        val freshFirstOrder = sFresh.step(1, x, epsB)

        assertContentEquals(freshFirstOrder, afterReset)
        assertTrue(
            !withHistory.contentEquals(afterReset),
            "2M branch must differ from first-order when history exists",
        )
    }

    private fun gaussian(rng: Random): Float {
        var u1: Float
        do { u1 = rng.nextFloat() } while (u1 == 0f)
        val u2 = rng.nextFloat()
        return (sqrt(-2f * kotlin.math.ln(u1)) *
            kotlin.math.cos(2f * Math.PI.toFloat() * u2))
    }
}
