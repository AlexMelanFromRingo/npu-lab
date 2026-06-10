package io.melan.npulab.scheduler

import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * DPM-Solver++ (2nd order multistep) for Stable Diffusion 1.5.
 *
 * The reference is Lu et al. 2022 ("DPM-Solver++"). For the discrete schedule
 * used by SD 1.5 we precompute alphas_cumprod for 1000 train timesteps and
 * subsample to numInferenceSteps using uniform spacing.
 *
 * State is one tensor of shape [4, 64, 64] in NCHW order (latents).
 * All arithmetic is done in FP32 here on the CPU — even on NPU the U-Net runs
 * one fp16 step at a time, so scheduler overhead is dominated by U-Net cost.
 */
class DpmSolverMultistep(
    val numInferenceSteps: Int = 20,
    val numTrainTimesteps: Int = 1000,
    val betaStart: Float = 0.00085f,
    val betaEnd: Float = 0.012f,
) {
    // Pre-computed schedule
    val alphasCumprod: FloatArray
    val sigmas: FloatArray              // length numTrainTimesteps
    val timesteps: IntArray             // length numInferenceSteps, descending
    val lambdas: FloatArray             // log(alpha/sigma) at each inference step

    private var prevModelOutput: FloatArray? = null

    init {
        // Scaled linear beta schedule used by SD 1.5
        val betas = FloatArray(numTrainTimesteps) { i ->
            val t = i.toFloat() / (numTrainTimesteps - 1)
            val b = sqrt(betaStart) + t * (sqrt(betaEnd) - sqrt(betaStart))
            b * b
        }
        alphasCumprod = FloatArray(numTrainTimesteps)
        var acc = 1.0
        for (i in 0 until numTrainTimesteps) {
            acc *= (1.0 - betas[i])
            alphasCumprod[i] = acc.toFloat()
        }
        // Variance-preserving sigmas: sigma_t = sqrt(1 - alphas_cumprod).
        // (NOT the Karras parametrization sigma/alpha — DPM-Solver++ formulas
        // below are written in diffusers' alpha_t/sigma_t convention.)
        sigmas = FloatArray(numTrainTimesteps) { i ->
            sqrt(1f - alphasCumprod[i])
        }
        // "Leading" spacing with steps_offset=1 — the schedule SD 1.5 ships with
        // (PNDM config) and the one Qualcomm's AI Hub binaries were calibrated
        // against. For 20 steps: [951, 901, ..., 51, 1].
        //
        // Deliberately NOT "linspace": that starts at t=999, which falls outside
        // the UNet `timestep` input quantization range of the AI Hub w8a16
        // binary (scale*65535 ≈ 968) and would silently clamp at the first —
        // structurally most important — step.
        timesteps = IntArray(numInferenceSteps) { i ->
            val stepRatio = numTrainTimesteps / numInferenceSteps
            (numInferenceSteps - 1 - i) * stepRatio + 1
        }
        lambdas = FloatArray(numInferenceSteps) { i ->
            val t = timesteps[i]
            val ac = alphasCumprod[t]
            0.5f * ln(ac / (1f - ac))
        }
    }

    /** Sample initial latents ~ N(0, I) * init_noise_sigma (==1 for DPM++). */
    fun initialLatents(shape: IntArray, seed: Long): FloatArray {
        val n = shape.fold(1) { a, b -> a * b }
        val rng = Random(seed)
        return FloatArray(n) {
            // Box-Muller
            var u1: Float
            do { u1 = rng.nextFloat() } while (u1 == 0f)
            val u2 = rng.nextFloat()
            (sqrt(-2f * ln(u1)) * kotlin.math.cos(2f * Math.PI.toFloat() * u2))
        }
    }

    /**
     * Convert U-Net's epsilon prediction into x0 prediction using the alphas
     * at the current timestep.
     */
    private fun convertEpsToX0(latents: FloatArray, epsPred: FloatArray, t: Int): FloatArray {
        val ac = alphasCumprod[t]
        val sqrtAc = sqrt(ac)
        val sqrtOneMinusAc = sqrt(1f - ac)
        return FloatArray(latents.size) { i ->
            (latents[i] - sqrtOneMinusAc * epsPred[i]) / sqrtAc
        }
    }

    /**
     * One DPM++ 2M multistep update. Returns the new latents.
     *
     * @param step index into [timesteps] (0..numInferenceSteps-1)
     * @param latents current latents (modified in place is fine but we return new)
     * @param noisePred U-Net's epsilon prediction at this step (post-CFG)
     */
    fun step(step: Int, latents: FloatArray, noisePred: FloatArray): FloatArray {
        val t = timesteps[step]
        val x0 = convertEpsToX0(latents, noisePred, t)

        if (step == numInferenceSteps - 1) {
            // Last step — return x0 directly (DPM++ behavior at terminal step)
            prevModelOutput = x0
            return x0
        }

        val tNext = timesteps[step + 1]
        val lambdaT = lambdas[step]
        val lambdaTNext = lambdas[step + 1]
        val h = lambdaTNext - lambdaT
        // Variance-preserving sigma/alpha (diffusers convention):
        //   x_t = alpha_t * x0 + sigma_t * eps,  alpha² + sigma² = 1.
        // The DPM++ update in x0-parametrization is
        //   x_next = alpha_next * X0 + sigma_next * (x_t - alpha_t * X0) / sigma_t
        // which is algebraically identical to diffusers'
        //   (sigma_next/sigma_t) x_t - alpha_next (e^{-h} - 1) X0.
        val alphaT = sqrt(alphasCumprod[t])
        val sigmaT = sqrt(1f - alphasCumprod[t])
        val sigmaTNext = sqrt(1f - alphasCumprod[tNext])
        val alphaTNext = sqrt(alphasCumprod[tNext])

        val prev = prevModelOutput
        val newLatents: FloatArray = if (prev == null || step == 0) {
            // First step — first-order (DPM-Solver-1 / DDIM) update
            FloatArray(latents.size) { i ->
                alphaTNext * x0[i] + sigmaTNext * (latents[i] - alphaT * x0[i]) / sigmaT
            }
        } else {
            // Multistep DPM++ 2M update — second-order using previous model output
            val lambdaPrev = lambdas[(step - 1).coerceAtLeast(0)]
            val hPrev = lambdaT - lambdaPrev
            val r = hPrev / h
            val d0 = x0
            val d1 = FloatArray(x0.size) { i -> (1f / r) * (x0[i] - prev[i]) }
            FloatArray(latents.size) { i ->
                alphaTNext * (d0[i] + 0.5f * d1[i]) + sigmaTNext *
                        (latents[i] - alphaT * d0[i]) / sigmaT
            }
        }
        prevModelOutput = x0
        return newLatents
    }

    fun reset() { prevModelOutput = null }
}
