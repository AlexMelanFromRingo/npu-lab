package io.melan.npulab.device

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import java.io.File

data class DeviceInfo(
    val deviceModel: String,
    val androidVersion: String,
    val socName: String,
    val socManufacturer: String,
    val supportedAbis: List<String>,
    val totalRamBytes: Long,
    val availRamBytes: Long,
    val cpuCores: Int,
    val cpuMaxMhz: Long,
    val htpArchGuess: Int,
    val qnnLibrariesPresent: List<String>,
    val runtimeJson: String?,        // populated after a backend is initialized
) {
    val totalRamMb: Long get() = totalRamBytes / (1024L * 1024L)
    val availRamMb: Long get() = availRamBytes / (1024L * 1024L)
    val cpuMaxGhz: Float get() = cpuMaxMhz / 1000f
}

object DeviceInfoCollector {

    /**
     * SoC name → likely HTP architecture (used to choose libQnnHtpV*Stub.so).
     * Add new chips here as Qualcomm releases them.
     */
    private val HTP_ARCH_BY_SOC = listOf(
        Regex("(?i)SM8850|8 Elite Gen 5")                        to 81,   // v81 — verified in QAIRT 2.46 docs
        Regex("(?i)SM8750|8 Elite\\b")                           to 79,
        Regex("(?i)SM8650|8 Gen 3")                              to 75,
        Regex("(?i)SM8550|8 Gen 2")                              to 73,
        Regex("(?i)SM8475|8\\+ Gen 1")                           to 69,
        Regex("(?i)SM8450|8 Gen 1")                              to 69,
    )

    fun collect(ctx: Context, runtimeJson: String? = null): DeviceInfo {
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }

        val socName = Build.SOC_MODEL.takeIf { it.isNotBlank() && it != "unknown" }
            ?: readSystemProp("ro.soc.model")
            ?: readSystemProp("ro.board.platform")
            ?: "unknown"
        val socManufacturer = Build.SOC_MANUFACTURER.takeIf { it.isNotBlank() && it != "unknown" }
            ?: readSystemProp("ro.soc.manufacturer")
            ?: "unknown"

        val htp = HTP_ARCH_BY_SOC.firstOrNull { it.first.containsMatchIn(socName) }?.second ?: 0

        return DeviceInfo(
            deviceModel = "${Build.MANUFACTURER} ${Build.MODEL}",
            androidVersion = "Android ${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})",
            socName = socName,
            socManufacturer = socManufacturer,
            supportedAbis = Build.SUPPORTED_ABIS.toList(),
            totalRamBytes = mi.totalMem,
            availRamBytes = mi.availMem,
            cpuCores = Runtime.getRuntime().availableProcessors(),
            cpuMaxMhz = readMaxCpuFreqMhz(),
            htpArchGuess = htp,
            qnnLibrariesPresent = probeNativeLibs(ctx),
            runtimeJson = runtimeJson,
        )
    }

    private fun readSystemProp(name: String): String? = try {
        val c = Class.forName("android.os.SystemProperties")
        val m = c.getMethod("get", String::class.java)
        (m.invoke(null, name) as? String)?.takeIf { it.isNotBlank() }
    } catch (_: Throwable) { null }

    private fun readMaxCpuFreqMhz(): Long {
        var maxKhz = 0L
        for (cpu in 0..15) {
            val f = File("/sys/devices/system/cpu/cpu$cpu/cpufreq/cpuinfo_max_freq")
            if (!f.exists()) continue
            runCatching {
                val v = f.readText().trim().toLong()
                if (v > maxKhz) maxKhz = v
            }
        }
        return maxKhz / 1000
    }

    private fun probeNativeLibs(ctx: Context): List<String> {
        // Two ways to find shipped .so files:
        //   1) nativeLibraryDir/<lib>.so — works only when extractNativeLibs=true.
        //      Since AGP 3.6 the default is FALSE, so this returns empty even when
        //      the libs ARE bundled. That bug showed up on a real S26 Ultra build.
        //   2) Read lib/<abi>/ entries directly from the installed APK zip.
        // We check (2) first, fall back to (1) for the unusual extracted layout.
        val present = LinkedHashSet<String>()
        val apkPath = ctx.applicationInfo.publicSourceDir ?: ctx.applicationInfo.sourceDir
        if (!apkPath.isNullOrBlank()) {
            runCatching {
                java.util.zip.ZipFile(apkPath).use { zf ->
                    val abi = android.os.Build.SUPPORTED_ABIS.firstOrNull() ?: "arm64-v8a"
                    val prefix = "lib/$abi/"
                    val entries = zf.entries()
                    while (entries.hasMoreElements()) {
                        val name = entries.nextElement().name
                        if (name.startsWith(prefix) && name.endsWith(".so")) {
                            present += name.removePrefix(prefix)
                        }
                    }
                }
            }
        }
        val extractedDir = File(ctx.applicationInfo.nativeLibraryDir)
        if (extractedDir.exists()) {
            extractedDir.listFiles { f -> f.isFile && f.name.endsWith(".so") }
                ?.forEach { present += it.name }
        }
        // Sort so libQnn* float to the top — that's what the user usually wants to see.
        return present.sortedWith(compareBy(
            { !it.startsWith("libQnn") },
            { !it.startsWith("libnpulab") },
            { it },
        ))
    }
}
