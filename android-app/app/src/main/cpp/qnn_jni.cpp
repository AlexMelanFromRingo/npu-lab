// JNI glue between io.melan.npulab.inference.NpuLabNative and qnn_runtime.cpp.

#include "qnn_runtime.h"

#include <android/log.h>
#include <jni.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define LOG_TAG "NpuLabJNI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

std::mutex g_mu;
std::unordered_map<uint64_t, std::unique_ptr<npulab::QnnRuntime>> g_runtimes;
uint64_t g_next_id = 1;

npulab::QnnRuntime* Lookup(uint64_t handle) {
    std::lock_guard<std::mutex> g(g_mu);
    auto it = g_runtimes.find(handle);
    return it == g_runtimes.end() ? nullptr : it->second.get();
}

// Context handles are pointers into a specific backend's context registry, so
// several runtimes (HTP + GPU + CPU during a benchmark) can be alive at once.
// Route every context-scoped call to the runtime that actually owns it.
npulab::QnnRuntime* OwnerOfContext(uint64_t ctx_handle) {
    std::lock_guard<std::mutex> g(g_mu);
    for (auto& kv : g_runtimes) {
        if (kv.second->HasContext(ctx_handle)) return kv.second.get();
    }
    return nullptr;
}

std::string DescToWire(const npulab::TensorDesc& td) {
    // Wire format: "name:dtype:d0,d1,...:scale:offset"
    // Must match TensorDtype.parse + parseTensorSpec on the Kotlin side.
    const char* dtype_name;
    switch (td.dtype) {
        case 0x0232: dtype_name = "float32"; break;
        case 0x0216: dtype_name = "float16"; break;
        case 0x0032: dtype_name = "int32"; break;
        case 0x0132: dtype_name = "uint32"; break;
        case 0x0016: dtype_name = "int16"; break;
        case 0x0116: dtype_name = "uint16"; break;
        case 0x0008: dtype_name = "int8"; break;
        case 0x0108: dtype_name = "uint8"; break;
        case 0x0416: dtype_name = "ufixed16"; break;
        case 0x0316: dtype_name = "sfixed16"; break;
        case 0x0408: dtype_name = "ufixed8"; break;
        case 0x0308: dtype_name = "sfixed8"; break;
        default:     dtype_name = "uint8"; break;
    }
    char buf[64];
    std::string s = td.name;
    s += ':'; s += dtype_name; s += ':';
    for (size_t i = 0; i < td.dims.size(); ++i) {
        if (i) s += ',';
        s += std::to_string(td.dims[i]);
    }
    snprintf(buf, sizeof(buf), ":%.10g:%d", static_cast<double>(td.scale), td.offset);
    s += buf;
    return s;
}

void ThrowQnnException(JNIEnv* env, const std::string& msg) {
    jclass cls = env->FindClass("io/melan/npulab/inference/QnnException");
    if (cls) env->ThrowNew(cls, msg.c_str());
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_io_melan_npulab_inference_NpuLabNative_initBackend(
    JNIEnv* env, jclass, jint backend, jint htp_arch, jint soc_model, jstring jnativeLibDir) {
    std::string nativeLibDir;
    const char* nativeLibDirCStr = nullptr;
    if (jnativeLibDir != nullptr) {
        const char* tmp = env->GetStringUTFChars(jnativeLibDir, nullptr);
        if (tmp) { nativeLibDir = tmp; env->ReleaseStringUTFChars(jnativeLibDir, tmp); }
        nativeLibDirCStr = nativeLibDir.c_str();
    }
    auto r = npulab::QnnRuntime::Create(
        static_cast<npulab::Backend>(backend), htp_arch, soc_model, nativeLibDirCStr);
    if (!r) return 0;
    if (!r->Initialized()) {
        // Surface the real bring-up failure (incl. the QNN log tail) instead
        // of returning a handle that dies later with INVALID_HANDLE.
        ThrowQnnException(env, r->last_error().empty()
                                   ? "QNN backend initialization failed"
                                   : r->last_error());
        return 0;
    }
    std::lock_guard<std::mutex> g(g_mu);
    uint64_t id = g_next_id++;
    g_runtimes[id] = std::move(r);
    return static_cast<jlong>(id);
}

JNIEXPORT jlong JNICALL
Java_io_melan_npulab_inference_NpuLabNative_loadContextBinary(
    JNIEnv* env, jclass, jlong backend_handle, jstring jpath) {
    auto* rt = Lookup(static_cast<uint64_t>(backend_handle));
    if (!rt) { ThrowQnnException(env, "invalid backend handle"); return 0; }
    const char* c = env->GetStringUTFChars(jpath, nullptr);
    std::string path(c);
    env->ReleaseStringUTFChars(jpath, c);
    uint64_t h = rt->LoadContextBinary(path);
    if (h == 0) ThrowQnnException(env, rt->last_error());
    return static_cast<jlong>(h);
}

JNIEXPORT jobjectArray JNICALL
Java_io_melan_npulab_inference_NpuLabNative_getGraphInputs(
    JNIEnv* env, jclass, jlong ctx_handle, jint graph_index) {
    npulab::QnnRuntime* owner = OwnerOfContext(static_cast<uint64_t>(ctx_handle));
    if (!owner) {
        ThrowQnnException(env, "no runtime owns this context handle");
        return env->NewObjectArray(0, env->FindClass("java/lang/String"), nullptr);
    }
    auto descs = owner->GraphInputs(static_cast<uint64_t>(ctx_handle), graph_index);
    jclass str_cls = env->FindClass("java/lang/String");
    jobjectArray out = env->NewObjectArray((jsize)descs.size(), str_cls, nullptr);
    for (jsize i = 0; i < (jsize)descs.size(); ++i) {
        env->SetObjectArrayElement(out, i, env->NewStringUTF(DescToWire(descs[i]).c_str()));
    }
    return out;
}

JNIEXPORT jobjectArray JNICALL
Java_io_melan_npulab_inference_NpuLabNative_getGraphOutputs(
    JNIEnv* env, jclass, jlong ctx_handle, jint graph_index) {
    npulab::QnnRuntime* owner = OwnerOfContext(static_cast<uint64_t>(ctx_handle));
    if (!owner) {
        ThrowQnnException(env, "no runtime owns this context handle");
        return env->NewObjectArray(0, env->FindClass("java/lang/String"), nullptr);
    }
    auto descs = owner->GraphOutputs(static_cast<uint64_t>(ctx_handle), graph_index);
    jclass str_cls = env->FindClass("java/lang/String");
    jobjectArray out = env->NewObjectArray((jsize)descs.size(), str_cls, nullptr);
    for (jsize i = 0; i < (jsize)descs.size(); ++i) {
        env->SetObjectArrayElement(out, i, env->NewStringUTF(DescToWire(descs[i]).c_str()));
    }
    return out;
}

JNIEXPORT jlong JNICALL
Java_io_melan_npulab_inference_NpuLabNative_execute(
    JNIEnv* env, jclass, jlong ctx_handle, jint graph_index,
    jobjectArray inputs, jobjectArray outputs) {
    npulab::QnnRuntime* owner = OwnerOfContext(static_cast<uint64_t>(ctx_handle));
    if (!owner) { ThrowQnnException(env, "no runtime owns this context handle"); return -1; }

    bool bad_buffer = false;
    auto unpack = [&](jobjectArray arr, const char* what) {
        std::vector<std::pair<void*, size_t>> out;
        const jsize n = env->GetArrayLength(arr);
        for (jsize i = 0; i < n; ++i) {
            jobject bb = env->GetObjectArrayElement(arr, i);
            void* addr = env->GetDirectBufferAddress(bb);
            jlong cap = env->GetDirectBufferCapacity(bb);
            if (addr == nullptr || cap < 0) {
                // Non-direct ByteBuffer (heap-backed) — GetDirectBufferAddress
                // returns null. Fail loudly instead of executing on garbage.
                char msg[96];
                snprintf(msg, sizeof(msg),
                         "%s[%d] is not a direct ByteBuffer — use allocateDirect()",
                         what, (int)i);
                ThrowQnnException(env, msg);
                bad_buffer = true;
            }
            out.push_back({addr, static_cast<size_t>(cap < 0 ? 0 : cap)});
            env->DeleteLocalRef(bb);
            if (bad_buffer) break;
        }
        return out;
    };
    auto ins = unpack(inputs, "inputs");
    if (bad_buffer) return -1;
    auto outs = unpack(outputs, "outputs");
    if (bad_buffer) return -1;
    int64_t us = owner->Execute(static_cast<uint64_t>(ctx_handle), graph_index, ins, outs);
    if (us < 0) ThrowQnnException(env, owner->last_error());
    return static_cast<jlong>(us);
}

JNIEXPORT jboolean JNICALL
Java_io_melan_npulab_inference_NpuLabNative_serializeContext(
    JNIEnv* env, jclass, jlong ctx_handle, jstring jpath) {
    npulab::QnnRuntime* owner = OwnerOfContext(static_cast<uint64_t>(ctx_handle));
    if (!owner) { ThrowQnnException(env, "no runtime owns this context handle"); return JNI_FALSE; }
    const char* c = env->GetStringUTFChars(jpath, nullptr);
    std::string path(c ? c : "");
    if (c) env->ReleaseStringUTFChars(jpath, c);
    bool ok = owner->SerializeContext(static_cast<uint64_t>(ctx_handle), path);
    if (!ok) { ThrowQnnException(env, owner->last_error()); return JNI_FALSE; }
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_io_melan_npulab_inference_NpuLabNative_queryDeviceInfo(
    JNIEnv* env, jclass, jlong backend_handle) {
    auto* rt = Lookup(static_cast<uint64_t>(backend_handle));
    if (!rt) return env->NewStringUTF("{}");
    return env->NewStringUTF(rt->DeviceInfoJson().c_str());
}

JNIEXPORT jstring JNICALL
Java_io_melan_npulab_inference_NpuLabNative_npuSelfTest(
    JNIEnv* env, jclass, jstring jlibDir, jstring jmodelPath) {
    auto toStd = [&](jstring s) {
        std::string out;
        if (s != nullptr) {
            const char* c = env->GetStringUTFChars(s, nullptr);
            if (c) { out = c; env->ReleaseStringUTFChars(s, c); }
        }
        return out;
    };
    std::string report = npulab::RunNpuSelfTest(toStd(jlibDir), toStd(jmodelPath));
    return env->NewStringUTF(report.c_str());
}

JNIEXPORT void JNICALL
Java_io_melan_npulab_inference_NpuLabNative_freeContext(
    JNIEnv*, jclass, jlong ctx_handle) {
    std::lock_guard<std::mutex> g(g_mu);
    for (auto& kv : g_runtimes) {
        if (kv.second->FreeContext(static_cast<uint64_t>(ctx_handle))) return;
    }
}

JNIEXPORT void JNICALL
Java_io_melan_npulab_inference_NpuLabNative_freeBackend(
    JNIEnv*, jclass, jlong backend_handle) {
    std::lock_guard<std::mutex> g(g_mu);
    g_runtimes.erase(static_cast<uint64_t>(backend_handle));
}

}  // extern "C"
