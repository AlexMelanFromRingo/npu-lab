// Shared walk over QnnSystemContext_BinaryInfo_t — the introspection step that
// discovers graph name + tensor schemas (including the backend-assigned tensor
// ids that QnnGraph_execute requires) from a serialized context binary.
//
// Used by BOTH:
//   - qnn_runtime.cpp (on device, before contextCreateFromBinary), and
//   - tools/host_introspect (x86 host test that runs this exact code against
//     the real .bin files via the SDK's x86_64 libQnnSystem.so).
//
// Requires the QNN headers (compile with NPULAB_HAVE_QNN).

#pragma once

#include <string>
#include <vector>

#include "QnnTypes.h"
#include "System/QnnSystemContext.h"

#include "qnn_runtime.h"  // TensorDesc

namespace npulab {
namespace introspect {

// Read a Qnn_Tensor_t (versioned) into our flat TensorDesc. The id MUST be
// preserved: QnnGraph_execute matches tensors by the id assigned at creation
// (see QnnGraph.h: "Tensors in inputs and outputs must carry the same ID").
inline TensorDesc TensorToDesc(const Qnn_Tensor_t& t) {
    TensorDesc d;
    const Qnn_QuantizeParams_t* qp = nullptr;
    if (t.version == QNN_TENSOR_VERSION_1) {
        d.id = t.v1.id;
        d.name = t.v1.name ? t.v1.name : "";
        d.dtype = static_cast<int>(t.v1.dataType);
        d.dims.reserve(t.v1.rank);
        for (uint32_t i = 0; i < t.v1.rank; ++i) {
            d.dims.push_back(static_cast<int>(t.v1.dimensions[i]));
        }
        qp = &t.v1.quantizeParams;
    } else if (t.version == QNN_TENSOR_VERSION_2) {
        d.id = t.v2.id;
        d.name = t.v2.name ? t.v2.name : "";
        d.dtype = static_cast<int>(t.v2.dataType);
        d.dims.reserve(t.v2.rank);
        for (uint32_t i = 0; i < t.v2.rank; ++i) {
            d.dims.push_back(static_cast<int>(t.v2.dimensions[i]));
        }
        qp = &t.v2.quantizeParams;
    }
    if (qp && qp->quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        d.scale = qp->scaleOffsetEncoding.scale;
        d.offset = qp->scaleOffsetEncoding.offset;
    }
    return d;
}

struct GraphSchema {
    std::string name;
    std::vector<TensorDesc> inputs;
    std::vector<TensorDesc> outputs;
};

/**
 * Extract the first graph's schema out of a BinaryInfo blob (multi-graph
 * binaries are rare for AI Hub outputs; index 0 matches runtime behavior).
 * Returns false and fills *err on unknown struct versions / empty binaries.
 */
inline bool FirstGraphSchema(const QnnSystemContext_BinaryInfo_t* info,
                             GraphSchema* out, std::string* err) {
    const QnnSystemContext_GraphInfo_t* graphs = nullptr;
    uint32_t numGraphs = 0;
    switch (info->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
            graphs = info->contextBinaryInfoV1.graphs;
            numGraphs = info->contextBinaryInfoV1.numGraphs;
            break;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
            graphs = info->contextBinaryInfoV2.graphs;
            numGraphs = info->contextBinaryInfoV2.numGraphs;
            break;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
            graphs = info->contextBinaryInfoV3.graphs;
            numGraphs = info->contextBinaryInfoV3.numGraphs;
            break;
        default:
            if (err) *err = "Unknown BinaryInfo version";
            return false;
    }
    if (numGraphs == 0 || graphs == nullptr) {
        if (err) *err = "Binary has no graphs";
        return false;
    }

    const QnnSystemContext_GraphInfo_t& gi = graphs[0];
    out->inputs.clear();
    out->outputs.clear();
    if (gi.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        out->name = gi.graphInfoV1.graphName ? gi.graphInfoV1.graphName : "";
        for (uint32_t i = 0; i < gi.graphInfoV1.numGraphInputs; ++i) {
            out->inputs.push_back(TensorToDesc(gi.graphInfoV1.graphInputs[i]));
        }
        for (uint32_t i = 0; i < gi.graphInfoV1.numGraphOutputs; ++i) {
            out->outputs.push_back(TensorToDesc(gi.graphInfoV1.graphOutputs[i]));
        }
    } else if (gi.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        out->name = gi.graphInfoV2.graphName ? gi.graphInfoV2.graphName : "";
        for (uint32_t i = 0; i < gi.graphInfoV2.numGraphInputs; ++i) {
            out->inputs.push_back(TensorToDesc(gi.graphInfoV2.graphInputs[i]));
        }
        for (uint32_t i = 0; i < gi.graphInfoV2.numGraphOutputs; ++i) {
            out->outputs.push_back(TensorToDesc(gi.graphInfoV2.graphOutputs[i]));
        }
    } else if (gi.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        out->name = gi.graphInfoV3.graphName ? gi.graphInfoV3.graphName : "";
        for (uint32_t i = 0; i < gi.graphInfoV3.numGraphInputs; ++i) {
            out->inputs.push_back(TensorToDesc(gi.graphInfoV3.graphInputs[i]));
        }
        for (uint32_t i = 0; i < gi.graphInfoV3.numGraphOutputs; ++i) {
            out->outputs.push_back(TensorToDesc(gi.graphInfoV3.graphOutputs[i]));
        }
    } else {
        if (err) *err = "Unknown GraphInfo version";
        return false;
    }
    return true;
}

}  // namespace introspect
}  // namespace npulab
