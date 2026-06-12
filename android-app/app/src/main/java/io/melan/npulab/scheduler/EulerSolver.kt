package io.melan.npulab.scheduler

import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * EulerDiscreteScheduler — a 1:1 port of the diffusers scheduler that
 * Qualcomm's reference Stable Diffusion app uses (and, crucially, the one the
 * AI Hub w8a16 binaries were **calibrated** against):
 *
 *   - scaled_linear betas 0.00085→0.012 over 1000 train steps;
 *   - "linspace" timestep spacing: float timesteps 999…0;
 *   - sigmas σ_t = √((1−ᾱ)/ᾱ) linearly interpolated at those timesteps,
 *     with a terminal 0 appended;
 *   - VE-style state: x = x₀ + σ·ε, `init_noise_sigma = σ_max ≈ 14.6`;
 *   - the UNet sees `scale_model_input(x) = x/√(σ²+1)` — algebraically the
 *     exact VP x_t, so the quantization grids line up;
 *   - epsilon prediction, plain Euler step:
 *     x_{i+1} = x_i + ε·(σ_{i+1} − σ_i).
 *
 * Note: the first timestep is 999.0, which the UNet's quantized `timestep`
 * input clamps to ≈968. The reference app has the same behavior — the
 * calibration data covered it.
 */
class EulerDiscreteSolver(
    val numInferenceSteps: Int = 20,
    numTrainTimesteps: Int = 1000,
    betaStart: Float = 0.00085f,
    betaEnd: Float = 0.012f,
) {
    /** Float timesteps fed to the UNet, descending from 999. */
    val timesteps: FloatArray

    /** σ per inference step + terminal 0 (size numInferenceSteps + 1). */
    val sigmas: FloatArray

    val initNoiseSigma: Float

    init {
        val betas = FloatArray(numTrainTimesteps) { i ->
            val t = i.toFloat() / (numTrainTimesteps - 1)
            val b = sqrt(betaStart) + t * (sqrt(betaEnd) - sqrt(betaStart))
            b * b
        }
        val alphasCumprod = FloatArray(numTrainTimesteps)
        var acc = 1.0
        for (i in 0 until numTrainTimesteps) {
            acc *= (1.0 - betas[i])
            alphasCumprod[i] = acc.toFloat()
        }
        val sigmaAll = FloatArray(numTrainTimesteps) { i ->
            sqrt((1f - alphasCumprod[i]) / alphasCumprod[i])
        }

        timesteps = FloatArray(numInferenceSteps) { i ->
            (numTrainTimesteps - 1).toFloat() *
                (numInferenceSteps - 1 - i) / (numInferenceSteps - 1).toFloat()
        }
        sigmas = FloatArray(numInferenceSteps + 1) { i ->
            if (i == numInferenceSteps) {
                0f
            } else {
                // linear interpolation of sigmaAll at fractional timestep
                val t = timesteps[i]
                val lo = t.toInt().coerceIn(0, numTrainTimesteps - 1)
                val hi = (lo + 1).coerceAtMost(numTrainTimesteps - 1)
                val frac = t - lo
                sigmaAll[lo] * (1 - frac) + sigmaAll[hi] * frac
            }
        }
        initNoiseSigma = sigmas.max()
    }

    /** x_T = N(0, I) · σ_max (VE convention). */
    fun initialLatents(numel: Int, seed: Long): FloatArray {
        val rng = Random(seed)
        return FloatArray(numel) { gaussian(rng) * initNoiseSigma }
    }

    /** What the UNet consumes: x/√(σ²+1) — exactly the VP-space x_t. */
    fun scaleModelInput(latents: FloatArray, step: Int): FloatArray {
        val s = 1f / sqrt(sigmas[step] * sigmas[step] + 1f)
        return FloatArray(latents.size) { latents[it] * s }
    }

    /** Plain Euler update with epsilon prediction. */
    fun step(step: Int, latents: FloatArray, noisePred: FloatArray): FloatArray {
        val dSigma = sigmas[step + 1] - sigmas[step]
        return FloatArray(latents.size) { i -> latents[i] + noisePred[i] * dSigma }
    }

    private fun gaussian(rng: Random): Float {
        var u1: Float
        do { u1 = rng.nextFloat() } while (u1 == 0f)
        val u2 = rng.nextFloat()
        return (sqrt(-2f * ln(u1)) *
            kotlin.math.cos(2f * Math.PI.toFloat() * u2))
    }
}
