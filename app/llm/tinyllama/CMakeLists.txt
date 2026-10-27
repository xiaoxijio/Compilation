cmake_minimum_required(VERSION 3.5.0)
if(NOT "${SAMPLE_BUILD_TARGET}" STREQUAL "sdk")
project(sample_tinyllama)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NPU_SAMPLE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../)

include(${NPU_SAMPLE_ROOT_DIR}/cmake/npu_sample_build.cmake)

if("${SAMPLE_BUILD_TARGET}" STREQUAL "sim")
    set(PLATFORM_ARCH "x86_64")
else()
    set(PLATFORM_ARCH "aarch64")
endif()

set(SAMPLE_TINYLLAMA_INC  ${NPU_SAMPLE_ROOT_DIR}/third_party/re2_lib/${PLATFORM_ARCH}/include
                    ${NPU_SAMPLE_ROOT_DIR}/third_party/sentencepiece_lib/${PLATFORM_ARCH}/include
                    ${NPU_SAMPLE_ROOT_DIR}/third_party/sentencepiece_lib/include
                    ${NPU_SAMPLE_ROOT_DIR}/third_party/protobuf_lib/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../common
                    ${CMAKE_CURRENT_SOURCE_DIR}/../common/runner
                    ${CMAKE_CURRENT_SOURCE_DIR}/../common/runner/utils
                    ${CMAKE_CURRENT_SOURCE_DIR}/../common/runner/ortm_runner
)
set(re2_DIR  ${NPU_SAMPLE_ROOT_DIR}/third_party/re2_lib/${PLATFORM_ARCH}/lib/cmake/re2)
set(absl_DIR ${NPU_SAMPLE_ROOT_DIR}/third_party/abseil-cpp_lib/${PLATFORM_ARCH}/lib/cmake/absl)
find_package(re2 CONFIG REQUIRED)
set(SAMPLE_TINYLLAMA_LIB   sentencepiece re2::re2)
set(SAMPLE_TINYLLAMA_LIB_DIRS
    ${NPU_SAMPLE_ROOT_DIR}/third_party/sentencepiece_lib/${PLATFORM_ARCH}/lib
    ${NPU_SAMPLE_ROOT_DIR}/third_party/abseil-cpp_lib/${PLATFORM_ARCH}/lib
    ${NPU_SAMPLE_ROOT_DIR}/third_party/re2_lib/${PLATFORM_ARCH}/lib
)
file(GLOB_RECURSE  NPU_SAMPLE_TINYLLAMA_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/*.cpp
)
# sample_tinyllama
build_binary(
    TARGET   linux_npu_sample_tinyllama
    SRCS     ${NPU_SAMPLE_TINYLLAMA_SRCS}
    INC_DIRS ${NPU_SAMPLE_INC_PATH} ./ ${SAMPLE_TINYLLAMA_INC}
    LIB_DIRS ${NPU_SAMPLE_LIB_DIRS} ${SAMPLE_TINYLLAMA_LIB_DIRS}
    LIBS     ${NPU_SAMPLE_LIB_NAME} ${SAMPLE_TINYLLAMA_LIB}
    OUT_NAME ${NPU_SAMPLE_TINYLLAMA_NAME}
    OUT_DIR  ${NPU_SAMPLE_OUT_DIR}
    DEFS LINUX
)
target_compile_options(linux_npu_sample_tinyllama PRIVATE -Wno-error -Wno-used -Wno-unused-variable -Wno-unused-function -fno-strict-aliasing -Wno-missing-field-initializers)