// QNN runtime implementation against the real Qualcomm AI Engine Direct headers
// (tested against SDK 2.46). Loads two shared libraries at runtime:
//
//   libQnnSystem.so             — for binary metadata introspection
//   libQnnHtp.so / Cpu / Gpu    — for actual graph execution
//
// On Android we don't link against any of these; they're picked up via dlopen
// from the APK's lib/ directory.
//
// If NPULAB_HAVE_QNN is not defined, only the stub path compiles — see header
// fallback in qnn_stubs/QnnInterface.h.

#include "qnn_runtime.h"

#include <android/log.h>
#include <dlfcn.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "NpuLab", __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  "NpuLab", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "NpuLab", __VA_ARGS__)

#ifdef NPULAB_HAVE_QNN
#include "QnnInterface.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnBackend.h"
#include "QnnDevice.h"
#include "QnnTensor.h"
#include "QnnTypes.h"
#include "QnnLog.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "System/QnnSystemInterface.h"
#include "System/QnnSystemContext.h"

#include "binary_info_walk.h"

// Ring buffer of recent QNN-internal WARN/ERROR lines. The numeric rc codes
// (e.g. 14001) are useless on their own — the actual cause ("could not load
// skel", "fastrpc open failed", …) only appears in the QNN log stream. We keep
// the tail and attach it to error messages so the UI is self-diagnosing even
// without adb access.
namespace {
std::mutex g_qnn_log_mu;
std::deque<std::string> g_qnn_log_tail;       // WARN/ERROR — always captured
std::deque<std::string> g_qnn_log_full;       // all levels — self-test only
bool g_qnn_log_capture_all = false;
constexpr size_t kQnnLogTailMax = 25;
constexpr size_t kQnnLogFullMax = 120;

void PushQnnLogLine(const char* line, bool warn_or_err) {
    std::lock_guard<std::mutex> g(g_qnn_log_mu);
    if (warn_or_err) {
        g_qnn_log_tail.emplace_back(line);
        while (g_qnn_log_tail.size() > kQnnLogTailMax) g_qnn_log_tail.pop_front();
    }
    if (g_qnn_log_capture_all) {
        g_qnn_log_full.emplace_back(line);
        while (g_qnn_log_full.size() > kQnnLogFullMax) g_qnn_log_full.pop_front();
    }
}

void SetQnnLogCaptureAll(bool on) {
    std::lock_guard<std::mutex> g(g_qnn_log_mu);
    g_qnn_log_capture_all = on;
    if (on) g_qnn_log_full.clear();
}

std::string DrainQnnLogFull() {
    std::lock_guard<std::mutex> g(g_qnn_log_mu);
    std::string out;
    for (const auto& l : g_qnn_log_full) {
        out += l;
        out += '\n';
    }
    g_qnn_log_full.clear();
    return out;
}
}  // namespace

namespace npulab {
std::string RecentQnnLog() {
    std::lock_guard<std::mutex> g(g_qnn_log_mu);
    if (g_qnn_log_tail.empty()) return "";
    std::string out = "\n--- QNN log tail ---";
    for (const auto& l : g_qnn_log_tail) {
        out += '\n';
        out += l;
    }
    return out;
}
}  // namespace npulab

// QNN logger callback — pipes QNN-internal diagnostics into logcat tag "QNN"
// and keeps WARN/ERROR lines in the ring buffer above.
extern "C" void NpulabQnnLogCallback(const char* fmt, QnnLog_Level_t level,
                                     uint64_t /*timestamp*/, va_list args) {
    int prio = ANDROID_LOG_INFO;
    switch (level) {
        case QNN_LOG_LEVEL_ERROR:   prio = ANDROID_LOG_ERROR; break;
        case QNN_LOG_LEVEL_WARN:    prio = ANDROID_LOG_WARN;  break;
        case QNN_LOG_LEVEL_INFO:    prio = ANDROID_LOG_INFO;  break;
        case QNN_LOG_LEVEL_VERBOSE:
        case QNN_LOG_LEVEL_DEBUG:   prio = ANDROID_LOG_DEBUG; break;
        default: break;
    }
    const bool warn_or_err =
        (level == QNN_LOG_LEVEL_ERROR || level == QNN_LOG_LEVEL_WARN);
    // Format the line when it must be buffered: always for WARN/ERROR, for
    // every level while a self-test capture is active.
    bool capture_all;
    {
        std::lock_guard<std::mutex> g(g_qnn_log_mu);
        capture_all = g_qnn_log_capture_all;
    }
    if (warn_or_err || capture_all) {
        va_list copy;
        va_copy(copy, args);
        char buf[512];
        vsnprintf(buf, sizeof(buf), fmt, copy);
        va_end(copy);
        // Trim trailing newline so the tail stays compact.
        size_t len = strlen(buf);
        while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) buf[--len] = '\0';
        if (len > 0) PushQnnLogLine(buf, warn_or_err);
    }
    __android_log_vprint(prio, "QNN", fmt, args);
}
#endif

namespace npulab {

size_t TensorDesc::byteSize() const {
    int bytes_per_elem = 1;
#ifdef NPULAB_HAVE_QNN
    switch (dtype) {
        case QNN_DATATYPE_FLOAT_32: case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UINT_32:  case QNN_DATATYPE_SFIXED_POINT_32:
        case QNN_DATATYPE_UFIXED_POINT_32:
            bytes_per_elem = 4; break;
        case QNN_DATATYPE_FLOAT_16: case QNN_DATATYPE_BFLOAT_16:
        case QNN_DATATYPE_INT_16:   case QNN_DATATYPE_UINT_16:
        case QNN_DATATYPE_SFIXED_POINT_16: case QNN_DATATYPE_UFIXED_POINT_16:
            bytes_per_elem = 2; break;
        case QNN_DATATYPE_INT_64: case QNN_DATATYPE_UINT_64:
        case QNN_DATATYPE_FLOAT_64:
            bytes_per_elem = 8; break;
        case QNN_DATATYPE_INT_8: case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8: case QNN_DATATYPE_UFIXED_POINT_8:
            bytes_per_elem = 1; break;
        default: bytes_per_elem = 1;
    }
#endif
    size_t n = 1;
    for (int d : dims) n *= (d > 0 ? size_t(d) : 1);
    return n * bytes_per_elem;
}

#ifdef NPULAB_HAVE_QNN
namespace {

const char* BackendLib(Backend b) {
    switch (b) {
        case Backend::HTP: return "libQnnHtp.so";
        case Backend::GPU: return "libQnnGpu.so";
        case Backend::CPU: return "libQnnCpu.so";
    }
    return "libQnnCpu.so";
}

const char* BackendName(Backend b) {
    switch (b) { case Backend::HTP: return "HTP";
                 case Backend::GPU: return "GPU";
                 case Backend::CPU: return "CPU"; }
    return "?";
}

/** Decode the QNN error namespace prefix so logs don't just say "rc=14001". */
const char* qnnErrorName(uint32_t rc) {
    if (rc == 0) return "OK";
    uint32_t mod = (rc / 1000) * 1000;
    switch (mod) {
        case 1000: return "COMMON";
        case 2000: return "PROPERTY";
        case 3000: return "OP_PACKAGE";
        case 4000: return "BACKEND";
        case 5000: return "CONTEXT";
        case 6000: return "GRAPH";
        case 7000: return "TENSOR";
        case 8000: return "MEM";
        case 9000: return "SIGNAL";
        case 10000: return "ERROR";
        case 11000: return "LOG";
        case 12000: return "PROFILE";
        case 13000: return "PERF";
        case 14000: return "DEVICE";
        case 15000: return "GLOBAL_CONFIG";
        case 30000: return "SYSTEM";
        case 60000: return "INTERFACE";
        default: return "?";
    }
}

bool SlurpFile(const std::string& path, std::vector<uint8_t>* out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    auto sz = f.tellg();
    if (sz <= 0) return false;
    out->resize(static_cast<size_t>(sz));
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(out->data()), sz);
    return f.good();
}

struct LoadedContext {
    Qnn_ContextHandle_t ctx = nullptr;
    Qnn_GraphHandle_t   graph = nullptr;
    std::string         graph_name;
    std::vector<TensorDesc> inputs;
    std::vector<TensorDesc> outputs;
    // Stable storage for dimensions arrays passed into Qnn_Tensor_t.
    // Indexed by tensor position; rebuilt on Execute.
    std::vector<std::vector<uint32_t>> in_dims_storage;
    std::vector<std::vector<uint32_t>> out_dims_storage;
};

class QnnRuntimeImpl : public QnnRuntime {
public:
    QnnRuntimeImpl(Backend b, int htp_arch, int soc_model, const char* native_lib_dir)
        : backend_(b), htp_arch_(htp_arch), soc_model_(soc_model),
          native_lib_dir_(native_lib_dir) {}

    ~QnnRuntimeImpl() override {
        if (perf_infra_ && perf_config_id_ != 0 && perf_infra_->destroyPowerConfigId) {
            perf_infra_->destroyPowerConfigId(perf_config_id_);
        }
        for (auto& kv : contexts_) {
            if (iface_) iface_->QNN_INTERFACE_VER_NAME.contextFree(kv.second.ctx, nullptr);
        }
        if (device_handle_ && iface_ && iface_->QNN_INTERFACE_VER_NAME.deviceFree) {
            iface_->QNN_INTERFACE_VER_NAME.deviceFree(device_handle_);
        }
        if (backend_handle_ && iface_) {
            iface_->QNN_INTERFACE_VER_NAME.backendFree(backend_handle_);
        }
        if (log_handle_ && iface_) {
            iface_->QNN_INTERFACE_VER_NAME.logFree(log_handle_);
        }
        if (sys_ctx_ && sys_iface_) {
            sys_iface_->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextFree(sys_ctx_);
        }
        if (system_dl_) dlclose(system_dl_);
        if (backend_dl_) dlclose(backend_dl_);
    }

    void MarkInitFailed() { init_ok_ = false; }

    /**
     * Self-diagnosis block appended to HTP failures so the on-screen error is
     * actionable without adb: which deviceCreate path won, what we believe the
     * SoC/arch to be, whether the skel files actually exist on disk, the DSP
     * search path, and the recent QNN-internal WARN/ERROR lines.
     */
    std::string HtpDiagnostics() const {
        std::ostringstream os;
        os << "\n--- npulab r2 diag ---";
        os << "\ndeviceCreate path: "
           << (device_handle_ ? device_create_path_ : "(none — device is null)");
        os << "\nsocModel=" << soc_model_ << " htpArch=v" << htp_arch_;
        os << "\nlogCreate rc=" << log_create_rc_;
        if (native_lib_dir_ && *native_lib_dir_) {
            for (const char* skel : {"libQnnHtpV81Skel.so", "libQnnHtpV81.so",
                                     "libQnnHtpV81Stub.so"}) {
                std::string p = std::string(native_lib_dir_) + "/" + skel;
                std::ifstream f(p, std::ios::binary | std::ios::ate);
                if (f) {
                    os << "\n" << skel << ": present ("
                       << static_cast<long long>(f.tellg()) << " B)";
                } else {
                    os << "\n" << skel << ": MISSING in " << native_lib_dir_;
                }
            }
        } else {
            os << "\nnative_lib_dir: (not provided)";
        }
        const char* adsp = getenv("ADSP_LIBRARY_PATH");
        os << "\nADSP_LIBRARY_PATH=" << (adsp ? adsp : "(unset)");
        std::string tail = RecentQnnLog();
        if (tail.empty()) {
            os << "\n(QNN log tail empty — no WARN/ERROR emitted by the QNN stack;"
                  " if logCreate rc!=0 the internal logger is inactive)";
        } else {
            os << tail;
        }
        return os.str();
    }

    bool Initialize() {
        // ADSP_LIBRARY_PATH is set from Application.onCreate() in Kotlin BEFORE
        // System.loadLibrary("npulab") fires — see QnnRuntimeLibs.setup(). It
        // points at the app's nativeLibraryDir, where the installer extracted
        // libQnnHtpV81Skel.so + libQnnHtpV81.so from jniLibs (the canonical
        // deployment used by Qualcomm's ai-hub-apps ChatApp). If it's missing
        // here, the FastRPC loader on the DSP will fail to resolve the V81 skel
        // and QnnContext_createFromBinary fails (often rc=14001, DEVICE).
        const char* adsp = getenv("ADSP_LIBRARY_PATH");
        LOGI("Initialize: ADSP_LIBRARY_PATH=%s", adsp ? adsp : "(unset)");
        // native_lib_dir_ is the app's nativeLibraryDir. Mirror it into
        // LD_LIBRARY_PATH so the host-side dispatcher can also reach
        // libQnnHtpV81Stub.so if its linker namespace search would miss it.
        if (native_lib_dir_ && *native_lib_dir_) {
            std::string ld = native_lib_dir_;
            if (const char* cur = getenv("LD_LIBRARY_PATH"); cur && *cur) {
                ld += ':'; ld += cur;
            }
            setenv("LD_LIBRARY_PATH", ld.c_str(), 1);
        }

        // 1) Load libQnnSystem.so for binary introspection
        system_dl_ = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
        if (!system_dl_) {
            last_error_ = std::string("dlopen libQnnSystem.so failed: ") + dlerror();
            LOGE("%s", last_error_.c_str());
            return false;
        }
        using SystemGetProviders = decltype(&QnnSystemInterface_getProviders);
        auto sysGet = reinterpret_cast<SystemGetProviders>(
            dlsym(system_dl_, "QnnSystemInterface_getProviders"));
        if (!sysGet) {
            last_error_ = "QnnSystemInterface_getProviders missing in libQnnSystem.so";
            return false;
        }
        const QnnSystemInterface_t** sys_ifaces = nullptr;
        uint32_t sys_num = 0;
        if (sysGet(&sys_ifaces, &sys_num) != QNN_SUCCESS || sys_num == 0) {
            last_error_ = "QnnSystemInterface_getProviders returned no providers";
            return false;
        }
        sys_iface_ = sys_ifaces[0];
        if (sys_iface_->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextCreate(&sys_ctx_) != QNN_SUCCESS) {
            last_error_ = "systemContextCreate failed";
            return false;
        }

        // 2) Load backend library
        const char* lib = BackendLib(backend_);
        backend_dl_ = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
        if (!backend_dl_) {
            last_error_ = std::string("dlopen ") + lib + " failed: " + dlerror();
            LOGE("%s", last_error_.c_str());
            return false;
        }
        using GetProviders = decltype(&QnnInterface_getProviders);
        auto getProviders = reinterpret_cast<GetProviders>(
            dlsym(backend_dl_, "QnnInterface_getProviders"));
        if (!getProviders) {
            last_error_ = std::string("QnnInterface_getProviders missing in ") + lib;
            return false;
        }
        const QnnInterface_t** ifaces = nullptr;
        uint32_t num = 0;
        if (getProviders(&ifaces, &num) != QNN_SUCCESS || num == 0) {
            last_error_ = "QnnInterface_getProviders returned no providers";
            return false;
        }
        iface_ = ifaces[0];

        // 3) Verbose logger — pipes QNN-internal messages to logcat tag "QNN".
        // Without this we have no visibility into what fails inside libQnnHtp.so.
        auto lrc = iface_->QNN_INTERFACE_VER_NAME.logCreate(
            NpulabQnnLogCallback, QNN_LOG_LEVEL_VERBOSE, &log_handle_);
        log_create_rc_ = static_cast<uint32_t>(lrc);
        LOGI("logCreate rc=%u handle=%p", (unsigned)lrc, log_handle_);

        // 4) Backend create
        auto rc = iface_->QNN_INTERFACE_VER_NAME.backendCreate(log_handle_, nullptr, &backend_handle_);
        if (rc != QNN_SUCCESS) {
            last_error_ = std::string("backendCreate failed rc=") + std::to_string(rc) +
                          " (" + qnnErrorName(rc) + ")" + RecentQnnLog();
            LOGE("%s", last_error_.c_str());
            return false;
        }
        LOGI("backendCreate OK handle=%p", backend_handle_);

        if (iface_->QNN_INTERFACE_VER_NAME.backendGetBuildId) {
            const char* build = nullptr;
            iface_->QNN_INTERFACE_VER_NAME.backendGetBuildId(&build);
            if (build) LOGI("QNN %s backend build: %s", BackendName(backend_), build);
        }

        // 5) Device. Attempt chain (first success wins):
        //    a) HTP only: explicit UNSIGNED process domain. Unsigned PD is the
        //       documented default, but requesting it explicitly is the
        //       deterministic form — and the leading suspect for the on-device
        //       rc=14001 at contextCreateFromBinary is a PD selection issue.
        //    b) nullptr config — the canonical on-device path (QnnSampleApp,
        //       ai-hub-apps ChatApp).
        //    c) HTP custom SOC/ARCH — these options are meant for OFFLINE
        //       prepare on x86; last-resort only.
        // The Device-screen self-test runs the same matrix and reports which
        // flavor can actually deserialize a context binary on this phone.
        if (iface_->QNN_INTERFACE_VER_NAME.deviceCreate) {
            Qnn_ErrorHandle_t drc = QNN_SUCCESS;
            if (backend_ == Backend::HTP) {
                QnnHtpDevice_CustomConfig_t pdCfg{};
                pdCfg.option = QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD;
                pdCfg.useSignedProcessDomain.deviceId = 0;
                pdCfg.useSignedProcessDomain.useSignedProcessDomain = false;
                QnnDevice_Config_t cfg{};
                cfg.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
                cfg.customConfig = &pdCfg;
                const QnnDevice_Config_t* cfgPtrs[] = {&cfg, nullptr};
                drc = iface_->QNN_INTERFACE_VER_NAME.deviceCreate(
                    log_handle_, cfgPtrs, &device_handle_);
                if (drc == QNN_SUCCESS && device_handle_ != nullptr) {
                    device_create_path_ = "unsignedPD(explicit)";
                    LOGI("deviceCreate (unsignedPD) OK handle=%p", device_handle_);
                } else {
                    LOGW("deviceCreate (unsignedPD) failed rc=%u (%s)",
                         (unsigned)drc, qnnErrorName(drc));
                    device_handle_ = nullptr;
                }
            }
            if (device_handle_ == nullptr) {
                drc = iface_->QNN_INTERFACE_VER_NAME.deviceCreate(
                    log_handle_, nullptr /*config*/, &device_handle_);
                if (drc == QNN_SUCCESS && device_handle_ != nullptr) {
                    device_create_path_ = "nullptr-config";
                    LOGI("deviceCreate (nullptr cfg) OK handle=%p", device_handle_);
                } else {
                    LOGW("deviceCreate (nullptr cfg) failed rc=%u (%s)",
                         (unsigned)drc, qnnErrorName(drc));
                    device_handle_ = nullptr;
                }
            }
            if (device_handle_ == nullptr &&
                backend_ == Backend::HTP && soc_model_ > 0 && htp_arch_ > 0) {
                QnnHtpDevice_CustomConfig_t socCfg{};
                socCfg.option   = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
                socCfg.socModel = static_cast<uint32_t>(soc_model_);

                QnnHtpDevice_CustomConfig_t archCfg{};
                archCfg.option       = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
                archCfg.arch.deviceId = 0;
                archCfg.arch.arch    = static_cast<QnnHtpDevice_Arch_t>(htp_arch_);

                QnnDevice_Config_t cfgA{}, cfgB{};
                cfgA.option       = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
                cfgA.customConfig = &socCfg;
                cfgB.option       = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
                cfgB.customConfig = &archCfg;
                const QnnDevice_Config_t* cfgPtrs[] = {&cfgA, &cfgB, nullptr};

                LOGI("deviceCreate last-resort (custom cfg): socModel=%d arch=v%d",
                     soc_model_, htp_arch_);
                drc = iface_->QNN_INTERFACE_VER_NAME.deviceCreate(
                    log_handle_, cfgPtrs, &device_handle_);
                if (drc == QNN_SUCCESS && device_handle_ != nullptr) {
                    device_create_path_ = "custom-config(soc+arch)";
                    LOGI("deviceCreate (custom cfg) OK handle=%p", device_handle_);
                } else {
                    LOGW("deviceCreate (custom cfg) failed rc=%u (%s)",
                         (unsigned)drc, qnnErrorName(drc));
                    device_handle_ = nullptr;
                }
            }
            if (device_handle_ == nullptr) {
                // Pull a human-readable QNN message if available.
                std::string detail;
                if (iface_->QNN_INTERFACE_VER_NAME.errorGetMessage) {
                    const char* msg = nullptr;
                    iface_->QNN_INTERFACE_VER_NAME.errorGetMessage(drc, &msg);
                    if (msg && *msg) detail = std::string(" — ") + msg;
                }
                last_error_ = "deviceCreate failed rc=" + std::to_string(drc) +
                              " (" + qnnErrorName(drc) + ")" + detail;
                LOGE("%s", last_error_.c_str());
                if (backend_ == Backend::HTP) {
                    // No DSP session — every later call would fail with a
                    // misleading 14001/5002 at context-create time. Fail NOW
                    // with full diagnostics naming the real culprit.
                    last_error_ += HtpDiagnostics();
                    return false;
                }
                // GPU/CPU: deviceCreate is optional — keep going.
            }
        } else {
            LOGW("deviceCreate is null in the interface — skipping");
        }

        // 6) Diagnostic — dump what the device actually reports.
        if (device_handle_ && iface_->QNN_INTERFACE_VER_NAME.deviceGetPlatformInfo) {
            const QnnDevice_PlatformInfo_t* pi = nullptr;
            auto prc = iface_->QNN_INTERFACE_VER_NAME.deviceGetPlatformInfo(
                log_handle_, &pi);
            if (prc == QNN_SUCCESS && pi) {
                if (pi->version == QNN_DEVICE_PLATFORM_INFO_VERSION_1) {
                    const auto& v1 = pi->v1;
                    LOGI("Device platform: %u hardware device(s)",
                         (unsigned)v1.numHwDevices);
                    for (uint32_t i = 0; i < v1.numHwDevices; ++i) {
                        const auto& d = v1.hwDevices[i].v1;
                        LOGI("  hwDevice[%u]: id=%u type=%u numCores=%u",
                             i, (unsigned)d.deviceId, (unsigned)d.deviceType,
                             (unsigned)d.numCores);
                    }
                }
                if (iface_->QNN_INTERFACE_VER_NAME.deviceFreePlatformInfo) {
                    iface_->QNN_INTERFACE_VER_NAME.deviceFreePlatformInfo(log_handle_, pi);
                }
            } else {
                LOGW("deviceGetPlatformInfo rc=%u (%s)",
                     (unsigned)prc, qnnErrorName(prc));
            }
        }

        // 7) HTP performance mode. Without an explicit power config the DSP
        //    runs on the default DCVS policy: clocks ramp lazily and the first
        //    inferences (and any benchmark) measure power management, not the
        //    network. The recipe below is the documented "high performance"
        //    setup from the SDK's htp_backend guidelines: DCVS off pinned to
        //    max voltage corner + low RPC control latency + max RPC polling.
        if (backend_ == Backend::HTP) SetupHtpBurstMode();
        return true;
    }

    void SetupHtpBurstMode() {
        if (!iface_->QNN_INTERFACE_VER_NAME.deviceGetInfrastructure) {
            LOGW("HTP perf: deviceGetInfrastructure unavailable — skipping");
            return;
        }
        QnnDevice_Infrastructure_t infra = nullptr;
        auto rc = iface_->QNN_INTERFACE_VER_NAME.deviceGetInfrastructure(&infra);
        if (rc != QNN_SUCCESS || infra == nullptr) {
            LOGW("HTP perf: deviceGetInfrastructure rc=%u (%s) — skipping",
                 (unsigned)rc, qnnErrorName(rc));
            return;
        }
        auto* htp_infra = static_cast<QnnHtpDevice_Infrastructure_t*>(infra);
        if (htp_infra->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
            LOGW("HTP perf: unexpected infraType=%d — skipping", (int)htp_infra->infraType);
            return;
        }
        perf_infra_ = &htp_infra->perfInfra;
        if (!perf_infra_->createPowerConfigId || !perf_infra_->setPowerConfig) {
            LOGW("HTP perf: power config fns are null — skipping");
            perf_infra_ = nullptr;
            return;
        }
        rc = perf_infra_->createPowerConfigId(/*deviceId*/ 0, /*coreId*/ 0, &perf_config_id_);
        if (rc != QNN_SUCCESS) {
            LOGW("HTP perf: createPowerConfigId rc=%u — skipping", (unsigned)rc);
            perf_infra_ = nullptr;
            return;
        }

        QnnHtpPerfInfrastructure_PowerConfig_t dcvs;
        memset(&dcvs, 0, sizeof(dcvs));
        dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        dcvs.dcvsV3Config.contextId       = perf_config_id_;
        dcvs.dcvsV3Config.setDcvsEnable   = 1;
        dcvs.dcvsV3Config.dcvsEnable      = 0;   // pin clocks, no dynamic scaling
        dcvs.dcvsV3Config.powerMode       =
            QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        dcvs.dcvsV3Config.setSleepLatency = 1;
        dcvs.dcvsV3Config.sleepLatency    = 40;  // µs, min recommended for burst
        dcvs.dcvsV3Config.setSleepDisable = 1;
        dcvs.dcvsV3Config.sleepDisable    = 1;
        dcvs.dcvsV3Config.setBusParams    = 1;
        dcvs.dcvsV3Config.busVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.busVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.setCoreParams   = 1;
        dcvs.dcvsV3Config.coreVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        dcvs.dcvsV3Config.coreVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

        QnnHtpPerfInfrastructure_PowerConfig_t rpc_latency;
        memset(&rpc_latency, 0, sizeof(rpc_latency));
        rpc_latency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
        rpc_latency.rpcControlLatencyConfig = 100;  // µs

        QnnHtpPerfInfrastructure_PowerConfig_t rpc_polling;
        memset(&rpc_polling, 0, sizeof(rpc_polling));
        rpc_polling.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
        rpc_polling.rpcPollingTimeConfig = 9999;    // µs, max — busy-wait through a burst

        const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {
            &dcvs, &rpc_latency, &rpc_polling, nullptr};
        rc = perf_infra_->setPowerConfig(perf_config_id_, cfgs);
        if (rc != QNN_SUCCESS) {
            LOGW("HTP perf: setPowerConfig rc=%u — running on default DCVS", (unsigned)rc);
        } else {
            LOGI("HTP perf: burst mode active (DCVS off, max corners, polling 9999us)");
        }
    }

    uint64_t LoadContextBinary(const std::string& path) override {
        std::vector<uint8_t> buf;
        if (!SlurpFile(path, &buf)) {
            last_error_ = "Failed to read " + path;
            return 0;
        }

        // Step A — introspect to discover graph name and tensor schemas
        // (incl. the tensor ids graphExecute matches on). Shared with the
        // host-side introspection test — see binary_info_walk.h.
        const QnnSystemContext_BinaryInfo_t* info = nullptr;
        Qnn_ContextBinarySize_t info_size = 0;
        auto sys_rc = sys_iface_->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextGetBinaryInfo(
            sys_ctx_, buf.data(), buf.size(), &info, &info_size);
        if (sys_rc != QNN_SUCCESS || info == nullptr) {
            last_error_ = "systemContextGetBinaryInfo failed rc=" + std::to_string(sys_rc) + " for " + path;
            return 0;
        }

        introspect::GraphSchema schema;
        std::string walk_err;
        if (!introspect::FirstGraphSchema(info, &schema, &walk_err)) {
            last_error_ = walk_err + " for " + path;
            return 0;
        }
        LoadedContext lc;
        lc.graph_name = std::move(schema.name);
        lc.inputs = std::move(schema.inputs);
        lc.outputs = std::move(schema.outputs);

        // Step B — actually create the context on the backend.
        LOGI("contextCreateFromBinary: backend=%p device=%p size=%zu graph='%s'",
             backend_handle_, device_handle_, buf.size(), lc.graph_name.c_str());
        Qnn_ContextHandle_t ctx = nullptr;
        auto rc = iface_->QNN_INTERFACE_VER_NAME.contextCreateFromBinary(
            backend_handle_, device_handle_, nullptr /*config*/,
            buf.data(), buf.size(), &ctx, nullptr /*signal*/);
        if (rc != QNN_SUCCESS) {
            // Pull a human-readable QNN message if available.
            std::string detail;
            if (iface_->QNN_INTERFACE_VER_NAME.errorGetMessage) {
                const char* msg = nullptr;
                iface_->QNN_INTERFACE_VER_NAME.errorGetMessage(rc, &msg);
                if (msg && *msg) detail = std::string(" — ") + msg;
            }
            last_error_ = "contextCreateFromBinary rc=" + std::to_string(rc) +
                          " (" + qnnErrorName(rc) + ")" + detail + " for " + path;
            if (backend_ == Backend::HTP) {
                last_error_ +=
                    " — common causes: (1) DSP could not resolve libQnnHtpV81Skel.so"
                    " via ADSP_LIBRARY_PATH, (2) binary compiled for a different"
                    " SoC/arch (qnn-context-binary-utility should report socModel=87,"
                    " dspArch=81 for S26 Ultra), (3) skel/stub version mismatch";
            } else {
                last_error_ +=
                    " — note: AI Hub context binaries are compiled FOR THE HTP"
                    " backend; the GPU/CPU backends cannot deserialize them."
                    " This combination is expected to fail — benchmark those"
                    " backends with a TFLite/ONNX build of the model instead.";
            }
            last_error_ += (backend_ == Backend::HTP) ? HtpDiagnostics()
                                                      : RecentQnnLog();
            LOGE("%s", last_error_.c_str());
            return 0;
        }
        LOGI("contextCreateFromBinary OK ctx=%p", ctx);

        // Step C — retrieve graph handle by name (from introspection).
        Qnn_GraphHandle_t graph = nullptr;
        rc = iface_->QNN_INTERFACE_VER_NAME.graphRetrieve(ctx, lc.graph_name.c_str(), &graph);
        if (rc != QNN_SUCCESS) {
            last_error_ = "graphRetrieve('" + lc.graph_name + "') rc=" + std::to_string(rc);
            iface_->QNN_INTERFACE_VER_NAME.contextFree(ctx, nullptr);
            return 0;
        }

        lc.ctx = ctx;
        lc.graph = graph;
        lc.in_dims_storage.resize(lc.inputs.size());
        lc.out_dims_storage.resize(lc.outputs.size());
        const uint64_t handle = reinterpret_cast<uint64_t>(ctx);
        contexts_[handle] = std::move(lc);
        LOGI("loaded %s — graph='%s' inputs=%zu outputs=%zu",
             path.c_str(), contexts_[handle].graph_name.c_str(),
             contexts_[handle].inputs.size(),
             contexts_[handle].outputs.size());
        return handle;
    }

    bool FreeContext(uint64_t handle) override {
        auto it = contexts_.find(handle);
        if (it == contexts_.end()) return false;
        if (iface_) iface_->QNN_INTERFACE_VER_NAME.contextFree(it->second.ctx, nullptr);
        contexts_.erase(it);
        return true;
    }

    bool HasContext(uint64_t handle) const override {
        return contexts_.count(handle) != 0;
    }

    std::vector<TensorDesc> GraphInputs(uint64_t handle, int) override {
        auto it = contexts_.find(handle);
        return it == contexts_.end() ? std::vector<TensorDesc>{} : it->second.inputs;
    }

    std::vector<TensorDesc> GraphOutputs(uint64_t handle, int) override {
        auto it = contexts_.find(handle);
        return it == contexts_.end() ? std::vector<TensorDesc>{} : it->second.outputs;
    }

    int64_t Execute(uint64_t handle, int,
                    const std::vector<std::pair<void*, size_t>>& inputs,
                    const std::vector<std::pair<void*, size_t>>& outputs) override {
        auto it = contexts_.find(handle);
        if (it == contexts_.end()) { last_error_ = "Unknown context handle"; return -1; }
        LoadedContext& lc = it->second;
        if (inputs.size() != lc.inputs.size() || outputs.size() != lc.outputs.size()) {
            last_error_ = "Input/output count mismatch: graph wants " +
                          std::to_string(lc.inputs.size()) + " in / " +
                          std::to_string(lc.outputs.size()) + " out, got " +
                          std::to_string(inputs.size()) + " / " +
                          std::to_string(outputs.size());
            return -1;
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            const size_t want = lc.inputs[i].byteSize();
            if (inputs[i].first == nullptr || inputs[i].second != want) {
                last_error_ = "Input '" + lc.inputs[i].name + "' size mismatch: expected " +
                              std::to_string(want) + " B, got " +
                              std::to_string(inputs[i].second) + " B" +
                              (inputs[i].first ? "" : " (null buffer)");
                return -1;
            }
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            const size_t want = lc.outputs[i].byteSize();
            if (outputs[i].first == nullptr || outputs[i].second != want) {
                last_error_ = "Output '" + lc.outputs[i].name + "' size mismatch: expected " +
                              std::to_string(want) + " B, got " +
                              std::to_string(outputs[i].second) + " B" +
                              (outputs[i].first ? "" : " (null buffer)");
                return -1;
            }
        }

        std::vector<Qnn_Tensor_t> in_tensors(lc.inputs.size());
        std::vector<Qnn_Tensor_t> out_tensors(lc.outputs.size());

        for (size_t i = 0; i < lc.inputs.size(); ++i) {
            auto& dims_buf = lc.in_dims_storage[i];
            dims_buf.assign(lc.inputs[i].dims.begin(), lc.inputs[i].dims.end());
            in_tensors[i] = Qnn_Tensor_t{QNN_TENSOR_VERSION_2, {}};
            in_tensors[i].v2 = QNN_TENSOR_V2_INIT;
            // id is how graphExecute matches this struct to the graph tensor —
            // it must be the id from the context binary, not 0.
            in_tensors[i].v2.id = lc.inputs[i].id;
            in_tensors[i].v2.name = lc.inputs[i].name.c_str();
            in_tensors[i].v2.type = QNN_TENSOR_TYPE_APP_WRITE;
            in_tensors[i].v2.dataType = static_cast<Qnn_DataType_t>(lc.inputs[i].dtype);
            in_tensors[i].v2.rank = static_cast<uint32_t>(dims_buf.size());
            in_tensors[i].v2.dimensions = dims_buf.data();
            in_tensors[i].v2.memType = QNN_TENSORMEMTYPE_RAW;
            in_tensors[i].v2.clientBuf.data = inputs[i].first;
            in_tensors[i].v2.clientBuf.dataSize = static_cast<uint32_t>(inputs[i].second);
        }
        for (size_t i = 0; i < lc.outputs.size(); ++i) {
            auto& dims_buf = lc.out_dims_storage[i];
            dims_buf.assign(lc.outputs[i].dims.begin(), lc.outputs[i].dims.end());
            out_tensors[i] = Qnn_Tensor_t{QNN_TENSOR_VERSION_2, {}};
            out_tensors[i].v2 = QNN_TENSOR_V2_INIT;
            out_tensors[i].v2.id = lc.outputs[i].id;
            out_tensors[i].v2.name = lc.outputs[i].name.c_str();
            out_tensors[i].v2.type = QNN_TENSOR_TYPE_APP_READ;
            out_tensors[i].v2.dataType = static_cast<Qnn_DataType_t>(lc.outputs[i].dtype);
            out_tensors[i].v2.rank = static_cast<uint32_t>(dims_buf.size());
            out_tensors[i].v2.dimensions = dims_buf.data();
            out_tensors[i].v2.memType = QNN_TENSORMEMTYPE_RAW;
            out_tensors[i].v2.clientBuf.data = outputs[i].first;
            out_tensors[i].v2.clientBuf.dataSize = static_cast<uint32_t>(outputs[i].second);
        }

        const auto t0 = std::chrono::steady_clock::now();
        auto rc = iface_->QNN_INTERFACE_VER_NAME.graphExecute(
            lc.graph,
            in_tensors.data(), static_cast<uint32_t>(in_tensors.size()),
            out_tensors.data(), static_cast<uint32_t>(out_tensors.size()),
            /*profile*/ nullptr, /*signal*/ nullptr);
        const auto t1 = std::chrono::steady_clock::now();
        if (rc != QNN_SUCCESS) {
            last_error_ = "graphExecute rc=" + std::to_string(rc) +
                          " (" + qnnErrorName(static_cast<uint32_t>(rc)) + ")" +
                          RecentQnnLog();
            return -1;
        }
        return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }

    std::string DeviceInfoJson() override {
        std::ostringstream os;
        os << "{";
        os << "\"backend\":\"" << BackendName(backend_) << "\",";
        os << "\"htp_arch\":" << htp_arch_ << ",";
        os << "\"qnn_compiled_in\":true";
        if (iface_ && iface_->QNN_INTERFACE_VER_NAME.backendGetBuildId) {
            const char* build = nullptr;
            iface_->QNN_INTERFACE_VER_NAME.backendGetBuildId(&build);
            if (build) os << ",\"build_id\":\"" << build << "\"";
        }
        if (iface_ && iface_->QNN_INTERFACE_VER_NAME.backendGetApiVersion) {
            Qnn_ApiVersion_t v{};
            if (iface_->QNN_INTERFACE_VER_NAME.backendGetApiVersion(&v) == QNN_SUCCESS) {
                os << ",\"core_api\":\"" << v.coreApiVersion.major << "."
                   << v.coreApiVersion.minor << "." << v.coreApiVersion.patch << "\"";
            }
        }
        os << "}";
        return os.str();
    }

private:
    Backend backend_;
    int htp_arch_;
    int soc_model_ = 0;
    const char* native_lib_dir_ = nullptr;
    void* system_dl_ = nullptr;
    void* backend_dl_ = nullptr;
    const QnnInterface_t* iface_ = nullptr;
    const QnnSystemInterface_t* sys_iface_ = nullptr;
    Qnn_BackendHandle_t backend_handle_ = nullptr;
    QnnSystemContext_Handle_t sys_ctx_ = nullptr;
    Qnn_LogHandle_t log_handle_ = nullptr;
    Qnn_DeviceHandle_t device_handle_ = nullptr;
    const QnnHtpDevice_PerfInfrastructure_t* perf_infra_ = nullptr;
    uint32_t perf_config_id_ = 0;
    uint32_t log_create_rc_ = 0;
    const char* device_create_path_ = "(not attempted)";
    std::unordered_map<uint64_t, LoadedContext> contexts_;
};

}  // namespace

/**
 * One-tap field diagnostics: runs the full HTP bring-up matrix in-process and
 * returns a copy-pasteable report. Tries every deviceCreate flavor and, for
 * each one that succeeds, attempts contextCreateFromBinary with the given
 * model — so a single run answers "which init path can actually load contexts
 * on THIS phone", alongside what the QNN stack thinks the SoC/arch is and the
 * full internal log. Independent of any live QnnRuntime instance.
 */
std::string RunNpuSelfTest(const std::string& native_lib_dir,
                           const std::string& model_path) {
    std::ostringstream r;
    r << "=== NPU Lab self-test r3 ===\n";

    for (const char* var : {"ADSP_LIBRARY_PATH", "CDSP_LIBRARY_PATH",
                            "DSP_LIBRARY_PATH"}) {
        const char* v = getenv(var);
        r << var << "=" << (v ? v : "(unset)") << "\n";
    }

    if (!native_lib_dir.empty()) {
        r << "\n[files] " << native_lib_dir << "\n";
        for (const char* f : {"libQnnHtp.so", "libQnnHtpV81Stub.so",
                              "libQnnHtpV81Skel.so", "libQnnHtpV81.so",
                              "libQnnSystem.so"}) {
            std::string p = native_lib_dir + "/" + f;
            std::ifstream s(p, std::ios::binary | std::ios::ate);
            if (s) {
                r << "  " << f << "  " << static_cast<long long>(s.tellg()) << " B\n";
            } else {
                r << "  " << f << "  MISSING\n";
            }
        }
    }

    SetQnnLogCaptureAll(true);
    struct CaptureGuard {
        ~CaptureGuard() { SetQnnLogCaptureAll(false); }
    } capture_guard;

    void* dl = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
    if (!dl) {
        r << "\ndlopen libQnnHtp.so FAILED: " << dlerror() << "\n";
        return r.str();
    }
    auto getProviders = reinterpret_cast<decltype(&QnnInterface_getProviders)>(
        dlsym(dl, "QnnInterface_getProviders"));
    const QnnInterface_t** ifaces = nullptr;
    uint32_t num = 0;
    if (!getProviders || getProviders(&ifaces, &num) != QNN_SUCCESS || num == 0) {
        r << "\nQnnInterface_getProviders FAILED\n";
        return r.str();
    }
    const auto& q = ifaces[0]->QNN_INTERFACE_VER_NAME;

    Qnn_LogHandle_t log = nullptr;
    auto lrc = q.logCreate(NpulabQnnLogCallback, QNN_LOG_LEVEL_VERBOSE, &log);
    r << "\nlogCreate rc=" << static_cast<unsigned>(lrc) << "\n";

    Qnn_BackendHandle_t backend = nullptr;
    auto brc = q.backendCreate(log, nullptr, &backend);
    r << "backendCreate rc=" << static_cast<unsigned>(brc) << " ("
      << qnnErrorName(static_cast<uint32_t>(brc)) << ")\n";
    if (brc != QNN_SUCCESS) {
        r << "\n--- QNN internal log ---\n" << DrainQnnLogFull();
        return r.str();
    }
    if (q.backendGetBuildId) {
        const char* build = nullptr;
        q.backendGetBuildId(&build);
        if (build) r << "backend build: " << build << "\n";
    }

    // What does the runtime think this chip is?
    if (q.deviceGetPlatformInfo) {
        const QnnDevice_PlatformInfo_t* pi = nullptr;
        auto prc = q.deviceGetPlatformInfo(log, &pi);
        r << "deviceGetPlatformInfo rc=" << static_cast<unsigned>(prc) << "\n";
        if (prc == QNN_SUCCESS && pi &&
            pi->version == QNN_DEVICE_PLATFORM_INFO_VERSION_1) {
            for (uint32_t i = 0; i < pi->v1.numHwDevices; ++i) {
                const auto& d = pi->v1.hwDevices[i].v1;
                r << "  hwDevice[" << i << "]: id=" << d.deviceId
                  << " type=" << d.deviceType << " cores=" << d.numCores;
                auto* ext = reinterpret_cast<QnnHtpDevice_DeviceInfoExtension_t*>(
                    d.deviceInfoExtension);
                if (ext && ext->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) {
                    r << "  onChip{socModel=" << ext->onChipDevice.socModel
                      << " arch=v" << static_cast<int>(ext->onChipDevice.arch)
                      << " signedPdSupport=" << (ext->onChipDevice.signedPdSupport ? 1 : 0)
                      << " dlbcSupport=" << (ext->onChipDevice.dlbcSupport ? 1 : 0)
                      << "}";
                }
                r << "\n";
            }
            if (q.deviceFreePlatformInfo) q.deviceFreePlatformInfo(log, pi);
        }
    }

    std::vector<uint8_t> model;
    const bool have_model = !model_path.empty() && SlurpFile(model_path, &model);
    r << "model: " << (model_path.empty() ? "(none provided)" : model_path);
    if (have_model) r << " (" << model.size() << " B)";
    else if (!model_path.empty()) r << " — NOT READABLE";
    r << "\n\n[deviceCreate × contextCreateFromBinary matrix]\n";

    for (int variant = 0; variant < 3; ++variant) {
        const char* name = variant == 0 ? "unsignedPD(explicit)"
                          : variant == 1 ? "nullptr-config"
                                         : "custom SOC=87+ARCH=81";
        QnnHtpDevice_CustomConfig_t pdCfg{};
        QnnHtpDevice_CustomConfig_t socCfg{};
        QnnHtpDevice_CustomConfig_t archCfg{};
        QnnDevice_Config_t c1{}, c2{};
        const QnnDevice_Config_t* cfgs[3] = {nullptr, nullptr, nullptr};
        const QnnDevice_Config_t** cfg_arg = nullptr;
        if (variant == 0) {
            pdCfg.option = QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD;
            pdCfg.useSignedProcessDomain.deviceId = 0;
            pdCfg.useSignedProcessDomain.useSignedProcessDomain = false;
            c1.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
            c1.customConfig = &pdCfg;
            cfgs[0] = &c1;
            cfg_arg = cfgs;
        } else if (variant == 2) {
            socCfg.option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
            socCfg.socModel = 87;
            archCfg.option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
            archCfg.arch.deviceId = 0;
            archCfg.arch.arch = QNN_HTP_DEVICE_ARCH_V81;
            c1.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
            c1.customConfig = &socCfg;
            c2.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
            c2.customConfig = &archCfg;
            cfgs[0] = &c1;
            cfgs[1] = &c2;
            cfg_arg = cfgs;
        }

        Qnn_DeviceHandle_t dev = nullptr;
        auto drc = q.deviceCreate(log, cfg_arg, &dev);
        r << "  " << name << ": deviceCreate rc=" << static_cast<unsigned>(drc)
          << " (" << qnnErrorName(static_cast<uint32_t>(drc)) << ")";
        if (drc == QNN_SUCCESS && dev != nullptr && have_model) {
            Qnn_ContextHandle_t ctx = nullptr;
            auto crc = q.contextCreateFromBinary(backend, dev, nullptr,
                                                 model.data(), model.size(),
                                                 &ctx, nullptr);
            r << " → contextCreateFromBinary rc=" << static_cast<unsigned>(crc)
              << " (" << qnnErrorName(static_cast<uint32_t>(crc)) << ")";
            if (crc == QNN_SUCCESS && ctx) {
                r << " ✓ WORKS";
                q.contextFree(ctx, nullptr);
            }
        }
        r << "\n";
        if (dev && q.deviceFree) q.deviceFree(dev);
    }

    r << "\n--- QNN internal log (all levels, tail) ---\n" << DrainQnnLogFull();
    if (q.backendFree) q.backendFree(backend);
    if (q.logFree && log) q.logFree(log);
    return r.str();
}

#endif  // NPULAB_HAVE_QNN

// Stub implementation when QNN SDK is unavailable.
#ifndef NPULAB_HAVE_QNN
namespace {
class StubRuntime : public QnnRuntime {
public:
    StubRuntime() {
        last_error_ = "QNN SDK not compiled in — set qnn.sdk.root in local.properties";
    }
    uint64_t LoadContextBinary(const std::string&) override { return 0; }
    bool FreeContext(uint64_t) override { return false; }
    bool HasContext(uint64_t) const override { return false; }
    std::vector<TensorDesc> GraphInputs(uint64_t, int) override { return {}; }
    std::vector<TensorDesc> GraphOutputs(uint64_t, int) override { return {}; }
    int64_t Execute(uint64_t, int,
                    const std::vector<std::pair<void*, size_t>>&,
                    const std::vector<std::pair<void*, size_t>>&) override { return -1; }
    std::string DeviceInfoJson() override {
        return "{\"qnn_compiled_in\":false,\"note\":\"set qnn.sdk.root in local.properties\"}";
    }
};
}  // namespace

std::string RunNpuSelfTest(const std::string&, const std::string&) {
    return "QNN SDK not compiled in — set qnn.sdk.root in local.properties and rebuild";
}
#endif

std::unique_ptr<QnnRuntime> QnnRuntime::Create(Backend backend, int htp_arch, int soc_model,
                                                const char* native_lib_dir) {
#ifdef NPULAB_HAVE_QNN
    auto r = std::make_unique<QnnRuntimeImpl>(backend, htp_arch, soc_model, native_lib_dir);
    if (!r->Initialize()) {
        // Keep the object so the JNI layer can read last_error(), but mark it
        // unusable — handing out a half-initialized runtime turns into
        // INVALID_HANDLE errors three calls later.
        r->MarkInitFailed();
    }
    return r;
#else
    (void)backend; (void)htp_arch; (void)soc_model; (void)native_lib_dir;
    return std::make_unique<StubRuntime>();
#endif
}

}  // namespace npulab
