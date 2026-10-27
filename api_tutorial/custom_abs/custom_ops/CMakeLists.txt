project(custom_opsets_extern)

if (ENABLE_BOARD_V100 STREQUAL "ON")

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-otv02-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-otv02-linux-gnu-g++)
file(GLOB EXTERN_CUSTOMOP_IMPL impl/*.c)
set(EXTRERN_BASE ${PROJECT_SOURCE_DIR}/../../)
include_directories(
    ${EXTRERN_BASE}/include
    ${PROJECT_SOURCE_DIR}/impl
    ${PROJECT_SOURCE_DIR}
)
add_library(ot_avp_npu_custom_opsets_extern SHARED ${EXTERN_CUSTOMOP_IMPL})
install(TARGETS ot_avp_npu_custom_opsets_extern)

elseif (ENABLE_BOARD_V200 STREQUAL "ON")

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-none-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-none-linux-gnu-g++)
set(CPU_ENGINE_ROOT_DIR ${PROJECT_SOURCE_DIR}/../../../)

file(GLOB EXTERN_CUSTOMOP_IMPL impl/*.c)
include_directories(
    ${CPU_ENGINE_ROOT_DIR}/include/cpu_opsets
    ${custom_opsets_extern_SOURCE_DIR}/impl
    ${custom_opsets_extern_SOURCE_DIR}
)
add_library(ot_avp_npu_custom_opsets_extern SHARED ${EXTERN_CUSTOMOP_IMPL})
install(TARGETS ot_avp_npu_custom_opsets_extern)

else ()

SET(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")

file(GLOB EXTERN_CUSTOMOP_SRCS *.cpp)
file(GLOB EXTERN_CUSTOMOP_DEF def/*.cpp)
file(GLOB EXTERN_CUSTOMOP_IMPL impl/*.c)
set(EXTRERN_BASE ${PROJECT_SOURCE_DIR}/../../)

include_directories(
    ${EXTRERN_BASE}/include
    ${PROJECT_SOURCE_DIR}/impl
    ${PROJECT_SOURCE_DIR}
)

add_library(custom_opsets_extern SHARED ${EXTERN_CUSTOMOP_SRCS} ${EXTERN_CUSTOMOP_DEF} ${EXTERN_CUSTOMOP_IMPL})
target_link_libraries(custom_opsets_extern PRIVATE
                        ot_avp_npu_custom_utils opschema)
target_link_directories(custom_opsets_extern PRIVATE ${EXTRERN_BASE}/lib)
install(TARGETS custom_opsets_extern)

endif ()