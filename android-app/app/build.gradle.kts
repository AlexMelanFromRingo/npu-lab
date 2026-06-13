import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("org.jetbrains.kotlin.plugin.serialization")
}

val localProps = Properties().apply {
    val f = rootProject.file("local.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}
val qnnSdkRoot: String? = (localProps.getProperty("qnn.sdk.root")
    ?: System.getenv("QNN_SDK_ROOT"))?.takeIf { it.isNotBlank() }

val releaseStoreFile: String? = localProps.getProperty("release.store.file")
val releaseStorePassword: String? = localProps.getProperty("release.store.password")
val releaseKeyAlias: String? = localProps.getProperty("release.key.alias")
val releaseKeyPassword: String? = localProps.getProperty("release.key.password")
val hasReleaseSigning = !releaseStoreFile.isNullOrBlank() && !releaseStorePassword.isNullOrBlank() &&
    !releaseKeyAlias.isNullOrBlank() && !releaseKeyPassword.isNullOrBlank()

android {
    namespace = "io.melan.npulab"
    compileSdk = 35
    ndkVersion = "26.3.11579264"

    defaultConfig {
        applicationId = "io.melan.npulab"
        minSdk = 31
        targetSdk = 35
        versionCode = 11
        versionName = "0.10.0"

        ndk {
            // S26 Ultra is arm64-v8a. We do not target other ABIs because QNN HTP
            // shared libraries we ship are arm64 only.
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17", "-fexceptions", "-frtti")
                val args = mutableListOf("-DANDROID_STL=c++_shared")
                if (qnnSdkRoot != null) {
                    args += "-DQNN_SDK_ROOT=$qnnSdkRoot"
                    println("NpuLab build: QNN SDK at $qnnSdkRoot")
                } else {
                    println("NpuLab build: no QNN SDK — using stubs (set qnn.sdk.root in local.properties to enable)")
                }
                arguments += args
            }
        }
    }

    signingConfigs {
        if (hasReleaseSigning) {
            create("release") {
                storeFile = rootProject.file(releaseStoreFile!!)
                storePassword = releaseStorePassword
                keyAlias = releaseKeyAlias
                keyPassword = releaseKeyPassword
            }
        }
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
            if (hasReleaseSigning) {
                signingConfig = signingConfigs.getByName("release")
            } else {
                println("NpuLab release: no signing config — APK will be unsigned. " +
                    "Set release.* props in local.properties to enable signing.")
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    packaging {
        jniLibs {
            // CRITICAL: `useLegacyPackaging = true` extracts native libs to
            // `nativeLibraryDir/` at install time as actual files.
            //
            // Default (false) keeps libs inside the APK, relying on Android's
            // APK-aware linker. That works for *our* process loading libQnn*.so,
            // but it does NOT work for the Hexagon DSP's FastRPC loader, which
            // needs a real on-disk path (set via ADSP_LIBRARY_PATH). Without
            // this flag, the DSP can't find libQnnHtpV81Skel.so and
            // QnnContext_createFromBinary fails with rc=14001
            // (QNN_DEVICE_ERROR_INVALID_CONFIG — a misleading name; the real
            // diagnostic is "skeleton library not found on DSP side").
            //
            // Same setting is used in the canonical github.com/qualcomm/ai-hub-apps
            // chatapp_android build.gradle.
            useLegacyPackaging = true
            pickFirsts += listOf("**/libc++_shared.so")
            // libQnnHtpV*Skel.so / libQnnHtpV*.so are Hexagon (DSP6) ELFs that
            // ride in jniLibs/arm64-v8a next to the aarch64 libs — exactly how
            // Qualcomm's ChatApp ships them. AGP's strip task only understands
            // aarch64, so exclude them from stripping instead of letting it
            // warn (or worse, mangle them). Wildcarded so bundling extra HTP
            // arches (V73/V75/V79/V81…) for older Snapdragons needs no change.
            keepDebugSymbols += listOf("**/libQnnHtpV*Skel.so", "**/libQnnHtpV*.so")
        }
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")
    implementation(composeBom)
    androidTestImplementation(composeBom)

    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")

    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")

    implementation("androidx.navigation:navigation-compose:2.8.5")
    implementation("androidx.security:security-crypto:1.1.0-alpha06")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")

    // ONNX Runtime + QNN Execution Provider: runs ANY .onnx on the Hexagon NPU
    // on-device. The -qnn AAR ships only libonnxruntime.so (~7.5 MB), no QNN
    // libs — it loads OUR bundled libQnnHtp.so (2.46) via backend_path, so there
    // is no second QNN stack to conflict with the hand-built pipeline.
    implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.26.0")

    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit")
}
