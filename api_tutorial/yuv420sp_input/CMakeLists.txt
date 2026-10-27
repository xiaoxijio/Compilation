cmake_minimum_required(VERSION 3.5.0)
if(NOT "${SAMPLE_BUILD_TARGET}" STREQUAL "sdk")
project(sample_yuv420sp_input)
endif()

set(NPU_SAMPLE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)

include(${NPU_SAMPLE_ROOT_DIR}/cmake/npu_sample_build.cmake)

# ortm_run_model
build_binary(
    TARGET   linux_npu_yuv420sp_input
    SRCS     ${NPU_SAMPLE_COMMOM_SRCS}
    INC_DIRS ${NPU_SAMPLE_INC_PATH}
    LIB_DIRS ${NPU_SAMPLE_LIB_DIRS}
    LIBS     ${NPU_SAMPLE_LIB_NAME}
    OUT_NAME ${NPU_SAMPLE_YUV420SP_INPUT_NAME}
    OUT_DIR  ${NPU_SAMPLE_OUT_DIR}
    DEFS LINUX
)
