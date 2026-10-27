project(protobuf)

add_library(libprotobuf SHARED IMPORTED GLOBAL)
add_library(libprotoc SHARED IMPORTED GLOBAL)
add_executable(protoc IMPORTED GLOBAL)

if (WIN32)
    # 
    # cmake ../cmake -G "Visual Studio 17 2022"
    # -Dprotobuf_BUILD_TESTS=OFF
    # -DWITH_PROTOC=0
    # -Dprotobuf_BUILD_PROTOC_BINARIES=ON
    # -Dprotobuf_BUILD_SHARED_LIBS=ON
    # 
    message("-- compile protobuf mode. configuration types: ${CMAKE_CONFIGURATION_TYPES}, build type: ${CMAKE_BUILD_TYPE}")
    if ("${CMAKE_CONFIGURATION_TYPES}" STREQUAL "Debug" OR 
        "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set_target_properties(libprotobuf PROPERTIES 
                            IMPORTED_IMPLIB_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib/libprotobufd.lib
                            IMPORTED_LOCATION_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin/libprotobufd.dll)
        set_target_properties(libprotoc PROPERTIES 
                            IMPORTED_IMPLIB_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib/libprotocd.lib
                            IMPORTED_LOCATION_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin/libprotocd.dll)
        file(GLOB protobuf_dll ${CMAKE_CURRENT_SOURCE_DIR}/bin/libprotobufd.dll ${CMAKE_CURRENT_SOURCE_DIR}/bin/libprotocd.dll)
        file(GLOB protobuf_lib ${CMAKE_CURRENT_SOURCE_DIR}/lib/libprotobufd.lib ${CMAKE_CURRENT_SOURCE_DIR}/lib/libprotocd.lib)
    else()
        set_target_properties(libprotobuf PROPERTIES 
                            IMPORTED_IMPLIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/libprotobuf.lib
                            IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/bin/libprotobuf.dll)
        set_target_properties(libprotoc PROPERTIES 
                            IMPORTED_IMPLIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/libprotoc.lib
                            IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/bin/libprotoc.dll)
        file(GLOB protobuf_dll ${CMAKE_CURRENT_SOURCE_DIR}/bin/libprotobuf.dll ${CMAKE_CURRENT_SOURCE_DIR}/bin/libprotoc.dll)
        file(GLOB protobuf_lib ${CMAKE_CURRENT_SOURCE_DIR}/lib/libprotobuf.lib ${CMAKE_CURRENT_SOURCE_DIR}/lib/libprotoc.lib)
    endif()
        set_target_properties(protoc PROPERTIES 
                            IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/bin/protoc.exe)
        install(FILES ${protobuf_dll} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
        install(FILES ${protobuf_lib} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
else()
    set_target_properties(libprotobuf PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/libprotobuf.so)
    set_target_properties(protoc PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/bin/protoc)
    file(GLOB protobuf_lib ${CMAKE_CURRENT_SOURCE_DIR}/lib/libprotobuf.so*)
    install(FILES ${protobuf_lib} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()

