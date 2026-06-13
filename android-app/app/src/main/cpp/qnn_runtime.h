// QNN runtime wrapper. Encapsulates dlopen'd handles + the QnnInterface_t vtable
// for a single backend (HTP / GPU / CPU). All methods are MT-unsafe — call them
// from one thread or use external synchronization.

#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace npulab {

enum class Backend : int { HTP = 0, GPU = 1, CPU = 2 };

struct TensorDesc {
    // Backend-assigned tensor id from the context binary. QnnGraph_execute
    // requires every tensor to carry the id it was registered with — passing 0
    // makes HTP fail with QNN_GRAPH_ERROR_INVALID_TENSOR. Preserved verbatim
    // from QnnSystemContext_getBinaryInfo.
    uint32_t id = 0;
    std::string name;
    int dtype;                  // mirrors Qnn_DataType_t — see headers
    std::vector<int> dims;
    // SCALE_OFFSET quantization params. Valid when dtype is a *FIXED_POINT_*.
    // For floating dtypes scale==0 (meaning: pass FP bytes directly).
    float scale = 0.f;
    int32_t offset = 0;
    size_t byteSize() const;
};

/**
 * Field diagnostics: runs the full HTP bring-up matrix (every deviceCreate
 * flavor × contextCreateFromBinary with the given model) plus platform info
 * and the QNN internal log, and returns a copy-pasteable text report. Safe to
 * call regardless of whether a QnnRuntime is alive.
 */
std::string RunNpuSelfTest(const std::string& native_lib_dir,
                           const std::string& model_path);

class QnnRuntime {
public:
    /**
     * @param native_lib_dir  Absolute path to the app's nativeLibraryDir
     *                        (`getApplicationInfo().nativeLibraryDir`). Used to
     *                        seed ADSP_LIBRARY_PATH so the Hexagon DSP can find
     *                        the matching libQnnHtpV*Skel.so we shipped in the APK.
     *                        Pass nullptr to skip — the DSP will only look in
     *                        /vendor/dsp/cdsp and friends.
     */
    static std::unique_ptr<QnnRuntime> Create(Backend backend, int htp_arch, int soc_model,
                                              const char* native_lib_dir);
    virtual ~QnnRuntime() = default;

    /** Load a serialized .bin context binary. Returns an opaque handle (>0) or 0. */
    virtual uint64_t LoadContextBinary(const std::string& path) = 0;
    virtual bool FreeContext(uint64_t ctx_handle) = 0;

    /** True if this runtime owns the given context handle. Used by the JNI
     *  layer to route context-scoped calls to the right backend instance. */
    virtual bool HasContext(uint64_t ctx_handle) const = 0;

    /**
     * Serialize a finalized context to a QNN context binary on disk. After a
     * DLC has been composed + finalized on-device (on-device compilation for
     * the connected chip), this caches the result as a `.bin` that
     * contextCreateFromBinary can load directly — skipping the slow prepare on
     * every subsequent load. Returns false (with last_error set) on failure.
     */
    virtual bool SerializeContext(uint64_t ctx_handle, const std::string& out_path) = 0;

    virtual std::vector<TensorDesc> GraphInputs(uint64_t ctx_handle, int graph_index) = 0;
    virtual std::vector<TensorDesc> GraphOutputs(uint64_t ctx_handle, int graph_index) = 0;

    /** Execute. Returns wall-time in µs or -1 on failure. */
    virtual int64_t Execute(uint64_t ctx_handle,
                            int graph_index,
                            const std::vector<std::pair<void*, size_t>>& inputs,
                            const std::vector<std::pair<void*, size_t>>& outputs) = 0;

    virtual std::string DeviceInfoJson() = 0;

    /** False when backend/device bring-up failed — last_error() has the cause
     *  (including the recent QNN-internal log tail). The JNI layer turns this
     *  into a QnnException instead of handing out a half-dead handle that
     *  would fail later with a misleading INVALID_HANDLE. */
    bool Initialized() const { return init_ok_; }

    const std::string& last_error() const { return last_error_; }
protected:
    std::string last_error_;
    bool init_ok_ = true;
};

}  // namespace npulab
