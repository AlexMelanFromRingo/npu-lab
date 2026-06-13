# JNI: keep NpuLabNative accessors
-keep class io.melan.npulab.inference.NpuLabNative { *; }
-keep class io.melan.npulab.inference.QnnException { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}

# kotlinx.serialization — keep @Serializable companion serializers
-if @kotlinx.serialization.Serializable class **
-keepclassmembers class <1> {
    static <1>$Companion Companion;
}
-keepclassmembers class **$Companion {
    *** serializer(...);
}
-keep,includedescriptorclasses class io.melan.npulab.account.AiHubAccount { *; }
-keep,includedescriptorclasses class io.melan.npulab.account.AccountsState { *; }
-keep,includedescriptorclasses class **$$serializer { *; }

# Coroutines internals occasionally surface through reflection in optimized builds.
-keepclassmembers class kotlinx.coroutines.internal.MainDispatcherFactory { *; }
-dontwarn kotlinx.coroutines.**

# androidx.security uses Google Tink, which references @CheckReturnValue / @Immutable
# annotations from error_prone_annotations (compile-only — not on the runtime classpath).
-dontwarn com.google.errorprone.annotations.**
-dontwarn javax.annotation.**
-dontwarn javax.annotation.concurrent.**
# Tink also has Protobuf-Lite plumbing that reflects into its own message classes:
-keep class com.google.crypto.tink.proto.** { *; }

# ONNX Runtime Java API (loaded reflectively / via JNI)
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**
