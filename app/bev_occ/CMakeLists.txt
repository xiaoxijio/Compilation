cmake_minimum_required(VERSION 3.16)
project(bev_occ_pc_simulator C)
set(CMAKE_C_STANDARD 99)
if(NOT NPU_TOOLCHAIN_PATH)
  message(FATAL_ERROR "Pass -DNPU_TOOLCHAIN_PATH=/path/to/npu_toolchain")
endif()
set(SIM_ROOT "${NPU_TOOLCHAIN_PATH}/linux/simulator")
set(SAMPLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
# Reuse the official arbitrary-input/output runner, not the six-output BEV demo.
add_executable(ortm_run_sim_model
  "${SAMPLE_ROOT}/api_tutorial/ortm_run_model/main.c"
  "${SAMPLE_ROOT}/sample_utils/file_utils.c"
  "${SAMPLE_ROOT}/sample_utils/opt_args.c"
  "${SAMPLE_ROOT}/sample_utils/simulator_utils.c")
target_compile_definitions(ortm_run_sim_model PRIVATE SIMULATOR LINUX)
target_include_directories(ortm_run_sim_model PRIVATE
  "${SIM_ROOT}/include" "${SAMPLE_ROOT}/sample_utils")
find_library(OT_SIMULATOR otsimulator PATHS "${SIM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(ortm_run_sim_model PRIVATE "${OT_SIMULATOR}" m)
set_target_properties(ortm_run_sim_model PROPERTIES BUILD_RPATH "${SIM_ROOT}/lib")
