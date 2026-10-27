cmake_minimum_required(VERSION 3.5.0)
if(NOT "${SAMPLE_BUILD_TARGET}" STREQUAL "sdk")
project(sample_custom_abs)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NPU_SAMPLE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)

include(${NPU_SAMPLE_ROOT_DIR}/cmake/npu_sample_build.cmake)

file(GLOB_RECURSE  NPU_SAMPLE_CUSTOM_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_ops/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_ops/*.cpp
)
set(SAMPLE_CUSTOM_INC ${CMAKE_CURRENT_SOURCE_DIR}/custom_ops
                      ${CMAKE_CURRENT_SOURCE_DIR}/custom_ops/impl
                      ${SIMULATOR_BASE}/../custom_ops/include)
if("${SAMPLE_BUILD_TARGET}" STREQUAL "sim")
    set(SAMPLE_CUSTOM_LIB_NAME opschema ot_avp_npu_custom_utils ot_npu_utils)
    set(SAMPLE_CUSTOM_LIB_DIRS ${SIMULATOR_BASE}/../custom_ops/lib)
endif()
# custom_abs
build_binary(
    TARGET   linux_npu_custom_abs
    SRCS     ${NPU_SAMPLE_COMMOM_SRCS} ${NPU_SAMPLE_CUSTOM_SRCS}
    INC_DIRS ${NPU_SAMPLE_INC_PATH} ${SAMPLE_CUSTOM_INC}
    LIB_DIRS ${NPU_SAMPLE_LIB_DIRS} ${SAMPLE_CUSTOM_LIB_DIRS}
    LIBS     ${NPU_SAMPLE_LIB_NAME} ${SAMPLE_CUSTOM_LIB_NAME}
    OUT_NAME ${NPU_SAMPLE_CUSTOM_ABS_NAME}
    OUT_DIR  ${NPU_SAMPLE_OUT_DIR}
    DEFS LINUX
)
