// Minimal stub of QNN headers so the project compiles when the real SDK is not
// installed. The actual structures and function signatures are picked from the
// public Qualcomm AI Engine Direct SDK documentation. These stubs are NOT a
// usable runtime — they let CMake succeed so you can iterate on UI code.

#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t Qnn_ErrorHandle_t;
typedef void* Qnn_BackendHandle_t;
typedef void* Qnn_ContextHandle_t;
typedef void* Qnn_GraphHandle_t;
typedef void* Qnn_DeviceHandle_t;
typedef void* Qnn_LogHandle_t;

typedef enum {
  QNN_SUCCESS = 0,
  QNN_COMMON_ERROR_GENERAL = 1,
} Qnn_ErrorCode_t;

typedef enum {
  QNN_DATATYPE_FLOAT_32 = 0x0008,
  QNN_DATATYPE_FLOAT_16 = 0x0004,
  QNN_DATATYPE_INT_32 = 0x0032,
  QNN_DATATYPE_INT_8 = 0x0008,
  QNN_DATATYPE_UINT_8 = 0x0408,
} Qnn_DataType_t;

typedef struct {
  uint32_t rank;
  uint32_t* dimensions;
} Qnn_TensorShape_t;

typedef struct {
  const char* name;
  uint32_t id;
  Qnn_DataType_t dataType;
  Qnn_TensorShape_t shape;
  void* data;
  size_t dataSize;
} Qnn_Tensor_t;

typedef struct {
  const char* (*backendGetBuildId)(void);
  Qnn_ErrorHandle_t (*backendCreate)(Qnn_LogHandle_t, const void*, Qnn_BackendHandle_t*);
  Qnn_ErrorHandle_t (*backendFree)(Qnn_BackendHandle_t);
  Qnn_ErrorHandle_t (*contextCreateFromBinary)(Qnn_BackendHandle_t,
                                               Qnn_DeviceHandle_t,
                                               const void*,
                                               const void*,
                                               size_t,
                                               Qnn_ContextHandle_t*,
                                               void*);
  Qnn_ErrorHandle_t (*contextFree)(Qnn_ContextHandle_t, void*);
  Qnn_ErrorHandle_t (*graphRetrieve)(Qnn_ContextHandle_t, const char*, Qnn_GraphHandle_t*);
  Qnn_ErrorHandle_t (*graphExecute)(Qnn_GraphHandle_t,
                                     const Qnn_Tensor_t*, uint32_t,
                                     Qnn_Tensor_t*, uint32_t,
                                     void*, void*);
} QnnInterface_t;

typedef Qnn_ErrorHandle_t (*QnnInterface_getProviders_t)(const QnnInterface_t***, uint32_t*);

#ifdef __cplusplus
}
#endif
