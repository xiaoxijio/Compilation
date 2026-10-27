cmake_minimum_required(VERSION 3.5.0)
if(NOT "${SAMPLE_BUILD_TARGET}" STREQUAL "sdk")
project(sample_lightstereo)
endif()

set(NPU_SAMPLE_LIGHTSTEREO_NAME sample_lightstereo)
set(NPU_SAMPLE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
include(${NPU_SAMPLE_ROOT_DIR}/cmake/npu_sample_build.cmake)
file(GLOB sample_lightstereo_src ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(NPU_SAMPLE_JSON_INC_PATH    ${NPU_SAMPLE_ROOT_DIR}/third_party/json/include)

# sample_lightstereo
build_binary(
    TARGET   linux_npu_sample_lightstereo
    SRCS     ${NPU_SAMPLE_COMMOM_SRCS} ${sample_lightstereo_src}
    INC_DIRS ${NPU_SAMPLE_INC_PATH} ${OPENCV_INCLUDE_DIR} ${NPU_SAMPLE_JSON_INC_PATH}
    LIB_DIRS ${NPU_SAMPLE_LIB_DIRS} ${OPENCV_OUT_DIR}
    LIBS     ${NPU_SAMPLE_LIB_NAME} ${OPENCV_LIBS_NAME}
    OUT_NAME ${NPU_SAMPLE_LIGHTSTEREO_NAME}
    OUT_DIR  ${NPU_SAMPLE_OUT_DIR}
    DEFS LINUX
)