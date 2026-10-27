project(opencv)

add_library(opencv_core SHARED IMPORTED GLOBAL)
add_library(opencv_imgcodecs SHARED IMPORTED GLOBAL)
add_library(opencv_imgproc SHARED IMPORTED GLOBAL)

if (UNIX)
    set_target_properties(opencv_core PROPERTIES 
                          IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/libopencv_core.so)
    set_target_properties(opencv_imgcodecs PROPERTIES 
                          IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/libopencv_imgcodecs.so)
    set_target_properties(opencv_imgproc PROPERTIES 
                          IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/libopencv_imgproc.so)
    file(GLOB opencv_lib ${CMAKE_CURRENT_SOURCE_DIR}/lib/libopencv*)
    install(FILES ${opencv_lib} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
elseif (WIN32)
    message("-- compile opencv mode. configuration types: ${CMAKE_CONFIGURATION_TYPES}, build type: ${CMAKE_BUILD_TYPE}")
    if ("${CMAKE_CONFIGURATION_TYPES}" STREQUAL "Debug" OR
        "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set_target_properties(opencv_core PROPERTIES
                            IMPORTED_IMPLIB_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib/opencv_core470d.lib
                            IMPORTED_LOCATION_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin/opencv_core470d.dll)
        set_target_properties(opencv_imgcodecs PROPERTIES
                            IMPORTED_IMPLIB_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib/opencv_imgcodecs470d.lib
                            IMPORTED_LOCATION_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin/opencv_imgcodecs470d.dll)
        set_target_properties(opencv_imgproc PROPERTIES
                            IMPORTED_IMPLIB_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib/opencv_imgproc470d.lib
                            IMPORTED_LOCATION_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin/opencv_imgproc470d.dll)
        file(GLOB opencv_dll ${CMAKE_CURRENT_SOURCE_DIR}/bin/opencv_core470d.dll
                             ${CMAKE_CURRENT_SOURCE_DIR}/bin/opencv_imgcodecs470d.dll
                             ${CMAKE_CURRENT_SOURCE_DIR}/bin/opencv_imgproc470d.dll)
        file(GLOB opencv_lib ${CMAKE_CURRENT_SOURCE_DIR}/lib/opencv_core470d.lib
                             ${CMAKE_CURRENT_SOURCE_DIR}/lib/opencv_imgcodecs470d.lib
                             ${CMAKE_CURRENT_SOURCE_DIR}/lib/opencv_imgproc470d.lib)
    else()
        # Release RelWithDebInfo MinSizeRel
        set_target_properties(opencv_core PROPERTIES
                            IMPORTED_IMPLIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/opencv_core470.lib
                            IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/bin/opencv_core470.dll)
        set_target_properties(opencv_imgcodecs PROPERTIES
                            IMPORTED_IMPLIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/opencv_imgcodecs470.lib
                            IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/bin/opencv_imgcodecs470.dll)
        set_target_properties(opencv_imgproc PROPERTIES
                            IMPORTED_IMPLIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/opencv_imgproc470.lib
                            IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/bin/opencv_imgproc470.dll)
        file(GLOB opencv_dll ${CMAKE_CURRENT_SOURCE_DIR}/bin/opencv_core470.dll
                             ${CMAKE_CURRENT_SOURCE_DIR}/bin/opencv_imgcodecs470.dll
                             ${CMAKE_CURRENT_SOURCE_DIR}/bin/opencv_imgproc470.dll)
        file(GLOB opencv_lib ${CMAKE_CURRENT_SOURCE_DIR}/lib/opencv_core470.lib
                             ${CMAKE_CURRENT_SOURCE_DIR}/lib/opencv_imgcodecs470.lib
                             ${CMAKE_CURRENT_SOURCE_DIR}/lib/opencv_imgproc470.lib)
    endif()

    install(FILES ${opencv_dll} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
    install(FILES ${opencv_lib} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif ()


