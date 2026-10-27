cmake_minimum_required(VERSION 3.5.0)
if(NOT "${SAMPLE_BUILD_TARGET}" STREQUAL "sdk")
project(sample_yolov7)
endif()

set(NPU_SAMPLE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../)
set(NPU_SAMPLE_MOD_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/../common/)
file(GLOB_RECURSE NPU_SAMPLE_MOD_COMMOM_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/*.cpp
)

include(${NPU_SAMPLE_ROOT_DIR}/cmake/npu_sample_build.cmake)

if(NOT "${SAMPLE_BUILD_TARGET}" STREQUAL "sim")
add_compile_options(-march=armv8.2-a+fp16)
endif()

# sample_yolov7
build_binary(
    TARGET   linux_npu_sample_yolov7
    SRCS     ${NPU_SAMPLE_COMMOM_SRCS} ${NPU_SAMPLE_MOD_COMMOM_SRCS}
    INC_DIRS ${NPU_SAMPLE_INC_PATH} ${NPU_SAMPLE_MOD_UTILS} 
    LIB_DIRS ${NPU_SAMPLE_LIB_DIRS}
    LIBS     ${NPU_SAMPLE_LIB_NAME}
    OUT_NAME ${NPU_SAMPLE_YOLOV7_NAME}
    OUT_DIR  ${NPU_SAMPLE_OUT_DIR}
    DEFS LINUX
)
