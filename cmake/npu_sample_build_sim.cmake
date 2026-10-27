if (WIN32)
  set(target_platform "windows")
else()
  set(target_platform "linux")
endif()
if(EXISTS "${NPU_SAMPLE_ROOT_DIR}/../../${target_platform}/simulator")
  # 发布包+sample形式
  set(SIMULATOR_BASE      ${NPU_SAMPLE_ROOT_DIR}/../../${target_platform}/simulator)
  set(NPU_SAMPLE_LIB_DIRS ${SIMULATOR_BASE}/lib)
elseif(EXISTS "/avp/npu/simulator")
  set(SIMULATOR_BASE      /avp/npu/simulator)
  set(NPU_SAMPLE_LIB_DIRS ${SIMULATOR_BASE}/lib)
elseif (EXISTS "${NPU_SAMPLE_ROOT_DIR}/../src/core" OR WIN32)
  # 源码包+sample形式
  set(SIMULATOR_BASE      ${NPU_SAMPLE_ROOT_DIR}/..)
  set(NPU_SAMPLE_LIB_DIRS ${SIMULATOR_BASE}/output/lib)
else()
  # sample独立编译形式
  string(TIMESTAMP CMAKE_DATE "%Y-%m-%d" UTC)
  set(NPU_TOOLCHAIN_PATH  "/avprelease/dailybuild/${CMAKE_DATE}/release_unzip/npu_toolchain")
  set(SIMULATOR_BASE      ${NPU_TOOLCHAIN_PATH}/linux/simulator)
  set(NPU_SAMPLE_LIB_DIRS ${SIMULATOR_BASE}/lib)
endif()
set(OPENCV_INCLUDE_DIR          ${NPU_SAMPLE_ROOT_DIR}/third_party/opencv/include)
set(OPENCV_OUT_DIR              ${NPU_SAMPLE_ROOT_DIR}/third_party/opencv/lib)
set(OPENCV_LIBS_NAME            opencv_core opencv_imgcodecs opencv_imgproc)

set(NPU_SAMPLE_INC_PATH ${SIMULATOR_BASE}/include
                        ${NPU_SAMPLE_ROOT_DIR}/sample_utils)
if(WIN32)
  set(NPU_SAMPLE_LIB_NAME otsimulator)
  set(NPU_SAMPLE_SHARED_LIB_NAME otsimulator)
  # 设置目标可执行文件的调试环境
  set(CMAKE_VS_DEBUGGER_ENVIRONMENT  "PATH=${SIMULATOR_BASE}/bin;$ENV{PATH}")
else()
  set(NPU_SAMPLE_LIB_NAME otsimulator m)
  set(NPU_SAMPLE_SHARED_LIB_NAME otsimulator m)
endif()
set(NPU_SAMPLE_OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_compile_options(-DSIMULATOR)

file(GLOB NPU_SAMPLE_COMMOM_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.c*
     ${NPU_SAMPLE_ROOT_DIR}/sample_utils/*.c)
