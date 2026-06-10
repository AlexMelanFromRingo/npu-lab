// Host-side introspection test: runs the app's real BinaryInfo walk
// (android-app/app/src/main/cpp/binary_info_walk.h) against the actual model
// .bin files using the QNN SDK's x86_64 libQnnSystem.so — the same library
// family the app dlopens on the phone.
//
// Verifies, for every binary the app loads on the S26 Ultra:
//   - the binary parses (BinaryInfo version is one we handle),
//   - the expected graph name and tensor names/order are present,
//   - EVERY graph tensor carries a nonzero backend id (QnnGraph_execute
//     matches tensors by id; id=0 was the bug that broke on-device runs).
//
// Build & run: scripts/run-host-introspect-test.sh

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

#include "binary_info_walk.h"

namespace {

bool Slurp(const std::string& path, std::vector<uint8_t>* out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    auto sz = f.tellg();
    if (sz <= 0) return false;
    out->resize(static_cast<size_t>(sz));
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(out->data()), sz);
    return f.good();
}

struct Expectation {
    const char* file;
    const char* graph;
    std::vector<const char*> inputs;   // expected names, in declared order
    std::vector<const char*> outputs;  // leading subset is enough
};

int g_failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) {
        printf("  [ok] %s\n", what.c_str());
    } else {
        printf("  [FAIL] %s\n", what.c_str());
        ++g_failures;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <models-dir>\n", argv[0]);
        return 2;
    }
    const std::string models = argv[1];

    void* dl = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
    if (!dl) {
        fprintf(stderr, "dlopen libQnnSystem.so failed: %s\n"
                "(run via scripts/run-host-introspect-test.sh so LD_LIBRARY_PATH "
                "points at $QNN_SDK_ROOT/lib/x86_64-linux-clang)\n", dlerror());
        return 2;
    }
    using GetProviders = decltype(&QnnSystemInterface_getProviders);
    auto get = reinterpret_cast<GetProviders>(dlsym(dl, "QnnSystemInterface_getProviders"));
    if (!get) { fprintf(stderr, "QnnSystemInterface_getProviders missing\n"); return 2; }

    const QnnSystemInterface_t** ifaces = nullptr;
    uint32_t num = 0;
    if (get(&ifaces, &num) != QNN_SUCCESS || num == 0) {
        fprintf(stderr, "no system interface providers\n");
        return 2;
    }
    const auto& sys = ifaces[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
    QnnSystemContext_Handle_t ctx = nullptr;
    if (sys.systemContextCreate(&ctx) != QNN_SUCCESS) {
        fprintf(stderr, "systemContextCreate failed\n");
        return 2;
    }

    const std::vector<Expectation> expectations = {
        {"sd15/text_encoder.bin", "stable_diffusion_v1_5_text_encoder",
         {"tokens"}, {"text_embedding"}},
        {"sd15/unet.bin", "stable_diffusion_v1_5_unet",
         {"timestep", "latent", "text_emb"}, {"output_latent"}},
        {"sd15/vae_decoder.bin", "stable_diffusion_v1_5_vae",
         {"latent"}, {"image"}},
        {"whisper/encoder.bin", "whisper_base_encoder",
         {"input_features"}, {"k_cache_cross_0", "v_cache_cross_0"}},
        {"whisper/decoder.bin", "whisper_base_decoder",
         {"input_ids", "position_ids"}, {"k_cache_self_0_out"}},
    };

    for (const auto& exp : expectations) {
        const std::string path = models + "/" + exp.file;
        printf("== %s\n", path.c_str());
        std::vector<uint8_t> buf;
        if (!Slurp(path, &buf)) {
            printf("  [skip] not downloaded (run scripts/fetch-models.py)\n");
            continue;
        }
        const QnnSystemContext_BinaryInfo_t* info = nullptr;
        Qnn_ContextBinarySize_t info_size = 0;
        auto rc = sys.systemContextGetBinaryInfo(ctx, buf.data(), buf.size(),
                                                 &info, &info_size);
        Check(rc == QNN_SUCCESS && info != nullptr, "systemContextGetBinaryInfo");
        if (rc != QNN_SUCCESS || info == nullptr) continue;

        npulab::introspect::GraphSchema schema;
        std::string err;
        Check(npulab::introspect::FirstGraphSchema(info, &schema, &err),
              "FirstGraphSchema (" + err + ")");
        Check(schema.name == exp.graph, "graph name '" + schema.name + "'");

        Check(schema.inputs.size() >= exp.inputs.size(),
              "input count " + std::to_string(schema.inputs.size()));
        for (size_t i = 0; i < exp.inputs.size() && i < schema.inputs.size(); ++i) {
            Check(schema.inputs[i].name == exp.inputs[i],
                  "input[" + std::to_string(i) + "] = '" + schema.inputs[i].name + "'");
        }
        for (size_t i = 0; i < exp.outputs.size() && i < schema.outputs.size(); ++i) {
            Check(schema.outputs[i].name == exp.outputs[i],
                  "output[" + std::to_string(i) + "] = '" + schema.outputs[i].name + "'");
        }
        bool all_ids = true;
        for (const auto& t : schema.inputs) all_ids &= (t.id != 0);
        for (const auto& t : schema.outputs) all_ids &= (t.id != 0);
        Check(all_ids, "all tensor ids nonzero (graphExecute id contract)");

        // Shape sanity for a known tensor (ufixed16 → 2 bytes/elem).
        for (const auto& t : schema.inputs) {
            if (schema.name == "stable_diffusion_v1_5_unet" && t.name == "latent") {
                size_t numel = 1;
                for (int d : t.dims) numel *= static_cast<size_t>(d);
                Check(numel * 2 == 1ull * 64 * 64 * 4 * 2,
                      "unet latent bytes " + std::to_string(numel * 2));
            }
        }
    }

    sys.systemContextFree(ctx);
    printf("\n%s (%d failure%s)\n", g_failures == 0 ? "PASSED" : "FAILED",
           g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
