cmake_minimum_required(VERSION 3.5.0)
if(NOT "${SAMPLE_BUILD_TARGET}" STREQUAL "sdk")
project(sample_async_forward_model)
endif()
set(NPU_SAMPLE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)

include(${NPU_SAMPLE_ROOT_DIR}/cmake/npu_sample_build.cmake)

# ortm_run_model
build_binary(
    TARGET   linux_npu_async_forward_model
    SRCS     ${NPU_SAMPLE_COMMOM_SRCS}
    INC_DIRS ${NPU_SAMPLE_INC_PATH} ./
    LIB_DIRS ${NPU_SAMPLE_LIB_DIRS}
    LIBS     ${NPU_SAMPLE_LIB_NAME}
    OUT_NAME ${NPU_SAMPLE_ASYNC_FORWARD_NAME}
    OUT_DIR  ${NPU_SAMPLE_OUT_DIR}
    DEFS LINUX
)
