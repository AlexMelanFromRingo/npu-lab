package io.melan.npulab

import android.app.Application
import io.melan.npulab.inference.QnnRuntimeLibs

/**
 * Application bootstrap. ADSP_LIBRARY_PATH (and LD_LIBRARY_PATH) must point at
 * applicationInfo.nativeLibraryDir — where the installer extracted our QNN libs,
 * including the Hexagon-side libQnnHtpV81Skel.so / libQnnHtpV81.so — BEFORE the
 * first dlopen of libQnnHtp.so. The QNN dispatcher reads ADSP_LIBRARY_PATH once
 * during its initial FastRPC setup; setting it after the fact is too late and
 * produces the misleading rc=14001 (QNN_DEVICE_ERROR_INVALID_CONFIG — actually
 * "DSP skeleton not found").
 *
 * This runs in Application.onCreate(), the only spot guaranteed to fire before
 * any Activity or ViewModel touches QnnRuntime.
 */
class NpuLabApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        QnnRuntimeLibs.setup(this)
    }
}
