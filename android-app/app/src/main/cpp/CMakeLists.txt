cmake_minimum_required(VERSION 3.22.1)
project(npulab CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QNN_SDK_ROOT is read from gradle.properties via the Android build (see app/build.gradle.kts).
# When set, we link against the real QNN headers and the app can talk to the NPU.
# When unset, we build with stubs so the project still compiles for CI / dry-run.
if(NOT DEFINED QNN_SDK_ROOT)
    set(QNN_SDK_ROOT $ENV{QNN_SDK_ROOT})
endif()

if(QNN_SDK_ROOT AND EXISTS "${QNN_SDK_ROOT}/include/QNN")
    message(STATUS "NpuLab: building WITH QNN SDK at ${QNN_SDK_ROOT}")
    add_compile_definitions(NPULAB_HAVE_QNN=1)
    include_directories(SYSTEM "${QNN_SDK_ROOT}/include/QNN")
else()
    message(WARNING
        "NpuLab: QNN_SDK_ROOT not set or invalid — building with QNN STUBS.\n"
        "        The APK will install and the UI will run, but model loading will\n"
        "        throw at runtime. Set QNN_SDK_ROOT in gradle.properties or as an\n"
        "        env var to enable real NPU inference. See docs/04-qnn-sdk-setup.md.")
    include_directories(SYSTEM "${CMAKE_CURRENT_SOURCE_DIR}/qnn_stubs")
endif()

add_library(npulab SHARED
    qnn_jni.cpp
    qnn_runtime.cpp
)

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(npulab
    ${log-lib}
    ${android-lib}
    dl
)

# QNN backends are dlopen'd at runtime (not link-time), so we don't link to any
# libQnn*.so here. The .so files must be packaged into the APK's jniLibs/arm64-v8a/.
