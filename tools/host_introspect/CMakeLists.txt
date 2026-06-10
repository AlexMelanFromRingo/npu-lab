cmake_minimum_required(VERSION 3.22)
project(host_introspect CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED QNN_SDK_ROOT)
    set(QNN_SDK_ROOT $ENV{QNN_SDK_ROOT})
endif()
if(NOT QNN_SDK_ROOT OR NOT EXISTS "${QNN_SDK_ROOT}/include/QNN")
    message(FATAL_ERROR "QNN_SDK_ROOT not set or invalid: '${QNN_SDK_ROOT}'")
endif()

add_executable(host_introspect main.cpp)
add_compile_definitions(NPULAB_HAVE_QNN=1)
target_include_directories(host_introspect SYSTEM PRIVATE
    "${QNN_SDK_ROOT}/include/QNN")
# binary_info_walk.h + qnn_runtime.h live in the app's cpp dir — the point is
# to test the exact code the APK ships.
target_include_directories(host_introspect PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../android-app/app/src/main/cpp")
target_link_libraries(host_introspect dl)
