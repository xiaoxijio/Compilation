include(ExternalProject)

# @bref Gets all subdirectories, and the result contains the current directory
function(subdir_list result curdir)
    file(GLOB children LIST_DIRECTORIES true ${curdir}/*)
    set(dirlist "")
    foreach(child ${children})
        if(IS_DIRECTORY ${child})
            list(APPEND dirlist ${child})
            subdir_list(subdirlist ${child})
            list(APPEND dirlist ${subdirlist})
            endif()
        endforeach()
    set(${result} ${dirlist} ${curdir} PARENT_SCOPE)
endfunction()

# @bref Obtain all directories containing header files in the specified path
function(include_list result curdir)
    file(GLOB_RECURSE children "${curdir}/*.h" "${curdir}/*.hpp")
    set(dirlist "")
    foreach(child ${children})
        string(REGEX REPLACE "(.*)/.*" "\\1" dirpath ${child})
        if(IS_DIRECTORY ${dirpath})
            list(FIND dirlist ${dirpath} list_index)
            if(${list_index} LESS 0)
                list(APPEND dirlist ${dirpath})
            endif()
        endif()
    endforeach()
    set(${result} ${dirlist} PARENT_SCOPE)
endfunction()

# @bref Obtain all directories containing shared or static library files in the specified path
function(library_list result curdir libtype)
    if(${libtype} STREQUAL "SHARED")
        file(GLOB_RECURSE children "${curdir}/*.so")
    else()
        file(GLOB_RECURSE children "${curdir}/*.a")
    endif()
    set(dirlist "")
    foreach(child ${children})
        string(REGEX REPLACE "(.*)/.*" "\\1" dirpath ${child})
        if(IS_DIRECTORY ${dirpath})
            list(FIND dirlist ${dirpath} list_index)
            if(${list_index} LESS 0)
                list(APPEND dirlist ${dirpath})
            endif()
        endif()
    endforeach()
    set(${result} ${dirlist} PARENT_SCOPE)
endfunction()

# @bref Convenience functions that simultaneously create compilation and cleanup targets for libraries
# @param TARGET: target name for the libraries
# @param OUT_NAME: output file name for the libraries
# @param OUT_DIR: output directory for the libraries
# @param SRCS: source files for the libraries
# @param INC_DIRS: include directories for the libraries
# @param LIB_DIRS: dependent library directories for the libraries
# @param LIBS: dependent libraries for the libraries
# @param DEPENDS: dependent targets for the libraries
# @param DEFS: extra-macros defined for the libraries
# @param CFLAGS: compile options for the libraries
# @param LDFLAGS: link options for the libraries
function(build_library)
    set(opt_args SHARED)
    set(one_val_args TARGET OUT_NAME OUT_DIR)
    set(multi_val_args SRCS INC_DIRS LIB_DIRS LIBS DEPENDS DEFS CFLAGS LDFLAGS)
    cmake_parse_arguments(BL "${opt_args}" "${one_val_args}" "${multi_val_args}" ${ARGN})
    if(${BL_SHARED})
        add_library(${BL_TARGET} SHARED ${BL_SRCS})
        set(out_file lib${BL_OUT_NAME}.so)
    else()
        add_library(${BL_TARGET} STATIC ${BL_SRCS})
        set(out_file lib${BL_OUT_NAME}.a)
    endif()
    if(BL_INC_DIRS)
        target_include_directories(${BL_TARGET} PRIVATE ${BL_INC_DIRS})
    endif()
    if(BL_LIB_DIRS)
        target_link_directories(${BL_TARGET} PRIVATE ${BL_LIB_DIRS})
    endif()
    if(BL_LIBS)
        target_link_libraries(${BL_TARGET} ${BL_LIBS})
    endif()
    if(BL_DEFS)
        target_compile_definitions(${BL_TARGET} PRIVATE -D${BL_DEFS})
    endif()
    set_target_properties(${BL_TARGET} PROPERTIES OUTPUT_NAME ${BL_OUT_NAME})
    if(${BL_SHARED})
       set_target_properties(${BL_TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${BL_OUT_DIR})
    else()
        set_target_properties(${BL_TARGET} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${BL_OUT_DIR})
    endif()
    target_compile_options(${BL_TARGET} PRIVATE ${BL_CFLAGS})
    if(DEBUG_CAT)
        target_link_options(${BL_TARGET} PRIVATE ${BL_LDFLAGS})
    else()
        target_link_options(${BL_TARGET} PRIVATE -s ${BL_LDFLAGS})
    endif()
    add_custom_target(${BL_TARGET}_clean
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND ${CMAKE_COMMAND} -P CMakeFiles/${BL_TARGET}.dir/cmake_clean.cmake
    )
    if(BL_DEPENDS)
        add_dependencies(${BL_TARGET} ${BL_DEPENDS})
    endif()
endfunction()

# @bref Convenience functions that simultaneously create compilation and cleanup targets for executables
# @param TARGET: target name for the executables
# @param OUT_NAME: output file name for the executables
# @param OUT_DIR: output directory for the executables
# @param SRCS: source files for the executables
# @param INC_DIRS: include directories for the executables
# @param LIB_DIRS: dependent library directories for the executables
# @param LIBS: dependent libraries for the executables
# @param DEPENDS: dependent targets for the executables
# @param DEFS: extra-macros defined for the executables
# @param CFLAGS: the flags to compile
# @param LDFLAGS: the flags to link
function(build_binary)
    set(opt_args "")
    set(one_val_args TARGET OUT_NAME OUT_DIR)
    set(multi_val_args SRCS INC_DIRS LIB_DIRS LIBS DEPENDS DEFS CFLAGS LDFLAGS)
    cmake_parse_arguments(BB "${opt_args}" "${one_val_args}" "${multi_val_args}" ${ARGN})
    set(out_file ${BB_OUT_NAME})
    add_executable(${BB_TARGET} ${BB_SRCS})
    if(BB_INC_DIRS)
        target_include_directories(${BB_TARGET} PRIVATE ${BB_INC_DIRS})
    endif()
    if(BB_LIB_DIRS)
        target_link_directories(${BB_TARGET} PRIVATE ${BB_LIB_DIRS})
    endif()
    if(BB_LIBS)
        target_link_libraries(${BB_TARGET} ${BB_LIBS})
    endif()
    if(BB_DEFS)
        target_compile_definitions(${BB_TARGET} PRIVATE -D${BB_DEFS})
    endif()
    set_target_properties(${BB_TARGET} PROPERTIES OUTPUT_NAME ${BB_OUT_NAME})
    set_target_properties(${BB_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BB_OUT_DIR})
    target_compile_options(${BB_TARGET} PRIVATE ${BB_CFLAGS})
    if(DEBUG_CAT)
        target_link_options(${BB_TARGET} PRIVATE ${BB_LDFLAGS})
    else()
        target_link_options(${BB_TARGET} PRIVATE -s ${BB_LDFLAGS})
    endif()
    add_custom_target(${BB_TARGET}_clean
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND ${CMAKE_COMMAND} -P CMakeFiles/${BB_TARGET}.dir/cmake_clean.cmake
    )
    if(BB_DEPENDS)
        add_dependencies(${BB_TARGET} ${BB_DEPENDS})
    endif()
endfunction()

# @bref Compared with the install command that comes with cmake, it is a convenient function to
#       install files to a specific directory immediately at compile time
# @param TARGET: target name for intallation action
# @param WORK_DIR: execute the installations with the given current working directory
# @param SRC_FILES: source files to be installed relative to the working directory
# @param DST_DIR: destination directory to be installed
# @param PERM: permissions after the file is installed
# @param DEPENDS: dependent targets for the executables
# @param BEFORES: targets depend on the executables
function(install_immed)
    set(opt_args "")
    set(one_val_args TARGET WORK_DIR DST_DIR PERM)
    set(multi_val_args SRC_FILES DEPENDS BEFORES)
    cmake_parse_arguments(II "${opt_args}" "${one_val_args}" "${multi_val_args}" ${ARGN})
    if(NOT II_DST_DIR)
        message(FATAL_ERROR "[${II_TARGET}] The destination directory cannot be empty!")
    endif()

    if(NOT II_SRC_FILES)
        message(FATAL_ERROR "[${II_TARGET}] No sources!")
    endif()

    if(II_WORK_DIR)
        set(workdir ${II_WORK_DIR})
    else()
        set(workdir ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    if(II_PERM)
        set(perm ${II_PERM})
    else()
        set(perm 0)
    endif()

    add_custom_target(${II_TARGET} ALL
        WORKING_DIRECTORY ${workdir}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${II_DST_DIR}
    )

    foreach(src_file ${II_SRC_FILES})
        add_custom_command(TARGET ${II_TARGET} POST_BUILD
            WORKING_DIRECTORY ${workdir}
            COMMAND chmod +x ${SDK_SCRIPTS_DIR}/install_file.sh
            COMMAND ${SDK_SCRIPTS_DIR}/install_file.sh '${src_file}' ${II_DST_DIR} ${perm}
        )
    endforeach()

    if(II_DEPENDS)
        add_dependencies(${II_TARGET} ${II_DEPENDS})
    endif()

    if(II_BEFORES)
        foreach(before_target ${II_BEFORES})
            add_dependencies(${before_target} ${II_TARGET})
        endforeach()
    endif()
endfunction()

# @bref Build tool from source code
# @param name: the target name of the building
# @param SRCS: list of source code file
# @param LIBS: list of link libs
# @param INCS: list of include opts
# @param OUT_DIR: output directory for the executables
# @param CFLAGS: list of compile opts
# @param LDFLAGS: list of link opts
# @param INSTALL_COMMAND: install command
function(BoardPackage_Add name)
    set(opt_args "")
    set(one_val_args OUT_DIR)
    set(multi_val_args SRCS LIBS INCS CFLAGS LDFLAGS INSTALL_COMMAND)

    cmake_parse_arguments(BP "${opt_args}" "${one_val_args}" "${multi_val_args}" ${ARGN})

    if(SDK_STATIC_BUILD)
        list(APPEND BP_CFLAGS --static)
        list(APPEND BP_LDFLAGS --static)
    endif()
    add_executable(${name})
    target_sources(${name} PRIVATE ${BP_SRCS})
    target_link_libraries(${name} PRIVATE ${BP_LIBS})
    target_include_directories(${name} PRIVATE  ${BP_INCS})

    target_compile_options(${name} PRIVATE ${BP_CFLAGS})
    if(DEBUG_CAT)
        target_link_options(${name}  PRIVATE ${BP_LDFLAGS})
    else()
        target_link_options(${name}  PRIVATE -s ${BP_LDFLAGS})
    endif()

    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BP_OUT_DIR})
    set(INSTALL_SCRIPTS ${CMAKE_CURRENT_BINARY_DIR}/_${name}_install.sh)

    set(insideValues FALSE)
    if(BP_INSTALL_COMMAND)
        file(WRITE  ${INSTALL_SCRIPTS} "")
    else()
        file(WRITE  ${INSTALL_SCRIPTS}
                    "${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${name} ${INS_FS_DIR}/usr/bin/")
    endif()

    foreach(carg ${ARGN})
        list(FIND opt_args "${carg}" optionIndex)
        list(FIND one_val_args "${carg}" singleArgIndex)
        list(FIND multi_val_args "${carg}" multiArgIndex)

        if(${optionIndex} EQUAL -1  AND  ${singleArgIndex} EQUAL -1  AND  ${multiArgIndex} EQUAL -1)
            if(insideValues)
                file(APPEND ${INSTALL_SCRIPTS} "${carg}\n")
            endif()
         else()
            if("INSTALL_COMMAND" STREQUAL "${carg}")
                set(insideValues TRUE)
            else()
                set(insideValues FALSE)
            endif()
        endif()
    endforeach()

    add_custom_target(${name}_clean
        COMMAND ${CMAKE_MAKE_PROGRAM} clean
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_custom_command(TARGET ${name} POST_BUILD
        COMMAND chmod u+x ${INSTALL_SCRIPTS} && ${INSTALL_SCRIPTS}
        COMMAND rm ${INSTALL_SCRIPTS}
    )
endfunction()


# @bref Build tool/lib thirdparty package
# @param NAME: target name
# @param VER: package version
# @param PC: build for host pc if set
# @param IN_BUILD: build in build-dir if set
# @param BUILD_ALWAYS: build the package regardless of whether the source code has changed
# @param URL: package's  path
# @param MD5: package's md5 checksum
# @param DEPENDS: list of include depend target
# @param SRC_DIR: directory of source code
# @param BUILD_DIR: build directory
# @param CFLAGS: list of compile opts
# @param LDFLAGS: list of link opts
# @param MAKE_PATCH: command to apply patch
# @param CONFIG_OPTS: list of configure opts
# @param MAKE_CONFIG: command to configure
# @param MAKE_ELF: command to make target
# @param MAKE_INSTALL: command to install target
function(ExternalAutomakePackage_Add)
    set(opt_args PC IN_BUILD BUILD_ALWAYS)
    set(one_val_args )
    set(multi_val_args
        VER NAME URL MD5 DEPENDS SRC_DIR BUILD_DIR CFLAGS LDFLAGS
        MAKE_PATCH CONFIG_OPTS MAKE_CONFIG MAKE_ELF MAKE_INSTALL
    )

    cmake_parse_arguments(PACKAGE "${opt_args}" "${one_val_args}" "${multi_val_args}" ${ARGN})

    if(NOT DEFINED PACKAGE_VER)
        message(FATAL_ERROR "you should define PACKAGE_VER")
    endif()
    if(NOT DEFINED PACKAGE_NAME)
        message(FATAL_ERROR "you should define PACKAGE_NAME")
    endif()
    if(NOT DEFINED PACKAGE_URL)
        message(FATAL_ERROR "you should define PACKAGE_URL")
    endif()
    if(NOT DEFINED PACKAGE_MD5)
        message(FATAL_ERROR "you should define PACKAGE_MD5")
    endif()

    if(TOOLS_CROSS_TOOLCHAIN)
        string(REGEX REPLACE "(.*)-" "\\1" PACKAGE_HOST ${TOOLS_CROSS_TOOLCHAIN})
    endif()

    if(NOT DEFINED PACKAGE_SRC_DIR)
        set(PACKAGE_SRC_DIR     ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}/src/${PACKAGE_NAME}-${PACKAGE_VER})
    endif()
    set(REAL_BUILD_DIR ${PACKAGE_SRC_DIR})

    if(NOT DEFINED PACKAGE_BUILD_DIR)
        set(PACKAGE_BUILD_DIR   ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}/build)
    endif()

    if(PACKAGE_IN_BUILD)
        set(REAL_BUILD_DIR ${PACKAGE_BUILD_DIR})
    endif()

    if(PACKAGE_PC)
        set(TARGET_POSTFIX "_pc")
    else()
        set(TARGET_POSTFIX "_board")
    endif()

    if(PACKAGE_BUILD_ALWAYS)
        set(BUILD_ALWAYS yes)
    else()
        set(BUILD_ALWAYS no)
    endif()

    set(PACKAGE_TARGET_NAME     ${PACKAGE_NAME}${TARGET_POSTFIX})

    if(PACKAGE_DEPENDS)
        set(PACKAGE_DEPENDS_)
        foreach(dep ${PACKAGE_DEPENDS})
            string(APPEND dep ${TARGET_POSTFIX})
            list(APPEND PACKAGE_DEPENDS_  ${dep})
        endforeach()
        set(PACKAGE_DEPENDS ${PACKAGE_DEPENDS_})
        unset(PACKAGE_DEPENDS_)
    endif()

    set(PREFIX_DIR   ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}/install)
    if(PACKAGE_MAKE_CONFIG)
        set(PACKAGE_MAKE_CONFIG     cd ${PACKAGE_SRC_DIR} && ${PACKAGE_MAKE_CONFIG})
    else()
        set(PACKAGE_MAKE_CONFIG
            ${PACKAGE_SRC_DIR}/configure
            --host=${PACKAGE_HOST} --prefix=${PREFIX_DIR}
            CFLAGS=${PACKAGE_CFLAGS} LDFLAGS=${PACKAGE_LDFLAGS}
            ${PACKAGE_CONFIG_OPTS} > /dev/null
        )
    endif()

    if(PACKAGE_MAKE_ELF)
        set(PACKAGE_MAKE_ELF    cd ${REAL_BUILD_DIR} && ${PACKAGE_MAKE_ELF})
    else()
        set(PACKAGE_MAKE_ELF    cd ${REAL_BUILD_DIR} && ${PATIENTLY_MAKE} all > /dev/null 2>&1)
    endif()

    if(PACKAGE_MAKE_INSTALL)
        set(PACKAGE_MAKE_INSTALL    cd ${REAL_BUILD_DIR} && ${PACKAGE_MAKE_INSTALL})
    else()
        set(PACKAGE_MAKE_INSTALL    cd ${REAL_BUILD_DIR} && ${PATIENTLY_MAKE} install > /dev/null)
    endif()

    ExternalProject_Add(${PACKAGE_TARGET_NAME}
        BUILD_ALWAYS                ${BUILD_ALWAYS}
        PREFIX                      ${PACKAGE_NAME}
        URL                         ${PACKAGE_URL}
        URL_HASH                    MD5=${PACKAGE_MD5}
        SOURCE_DIR                  ${PACKAGE_SRC_DIR}
        BINARY_DIR                  ${PACKAGE_BUILD_DIR}
        PATCH_COMMAND               ${PACKAGE_MAKE_PATCH}
        CONFIGURE_COMMAND           ${PACKAGE_MAKE_CONFIG}
        BUILD_COMMAND               ${PACKAGE_MAKE_ELF}
        INSTALL_COMMAND             ${PACKAGE_MAKE_INSTALL}
        DEPENDS                     ${PACKAGE_DEPENDS}
    )

    add_custom_target(${PACKAGE_TARGET_NAME}_clean
        COMMAND ${CMAKE_MAKE_PROGRAM} clean
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endfunction()

# @bref package build for host pc wrapper of ExternalAutomakePackage_Add
macro(PCThirdPartyPackage_Add)
    ExternalAutomakePackage_Add(PC ${ARGN})
endmacro()

# @bref package build for board wrapper of ExternalAutomakePackage_Add
macro(BoardThirdPartyPackage_Add)
    ExternalAutomakePackage_Add(${ARGN})
endmacro()

# @bref Strip the specific files
# @param TARGET: target name for intallation action
# @param WORK_DIR: execute the installations with the given current working directory
# @param STRIP_PREFIX: the toolchain to be used to execute strip cmd
# @param SRC_FILES: source files to be installed relative to the working directory
# @param DEPENDS: dependent targets for the executables
function(strip_files)
    set(opt_args "")
    set(one_val_args TARGET WORK_DIR STRIP_PREFIX)
    set(multi_val_args SRC_FILES DEPENDS)
    cmake_parse_arguments(SF "${opt_args}" "${one_val_args}" "${multi_val_args}" ${ARGN})

    if(SF_STRIP_PREFIX)
        set(strip_cmd ${SF_STRIP_PREFIX})
    else()
        set(strip_cmd ${CMAKE_STRIP})
    endif()

    if(SF_WORK_DIR)
        set(workdir ${SF_WORK_DIR})
    else()
        set(workdir ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    add_custom_target(${SF_TARGET} ALL
        WORKING_DIRECTORY ${workdir}
    )

    if(SF_DEPENDS)
        add_dependencies(${SF_TARGET} ${SF_DEPENDS})
    endif()

    foreach(src_file ${SF_SRC_FILES})
        add_custom_command(TARGET ${SF_TARGET} POST_BUILD
            WORKING_DIRECTORY ${workdir}
            COMMAND chmod +x ${SDK_SCRIPTS_DIR}/strip_file.sh
            COMMAND ${SDK_SCRIPTS_DIR}/strip_file.sh ${strip_cmd} ${src_file}
        )
    endforeach()
endfunction()

function(get_xml_params keyline keyword value)
    set(PYTHON_SCRIPT ${SDK_SCRIPTS_DIR}/xml_extract_paras.py)
    if(DEBUG_CAT)
        set(XML_FILE ${SDK_ROOT_DIR}/mcu/boot/tables/partition/${CHIP_TYPE}_${BOARD_TYPE}_${BOOT_MEDIA}_debug.xml)
    else()
        set(XML_FILE ${SDK_ROOT_DIR}/mcu/boot/tables/partition/${CHIP_TYPE}_${BOARD_TYPE}_${BOOT_MEDIA}.xml)
    endif()

    execute_process(
        COMMAND python3 ${PYTHON_SCRIPT} ${XML_FILE} ${keyline} ${keyword}
        RESULT_VARIABLE script_result
        OUTPUT_VARIABLE script_output
    )
    if(NOT script_result EQUAL "0")
        message(FATAL_ERROR "${script_output}")
    endif()

    #ȥ��script_output����ֵ���˵Ŀհ��ַ����������з�
    string(STRIP ${script_output} output)
    set(${value} ${output} PARENT_SCOPE)
endfunction()

# @bref Install the specific files to sdk/out
# @param TARGET: target name for intallation action
# @param WORK_DIR: execute the installations with the given current working directory
# @param DST_DIR: destination directory to be installed
# @param HEADER_FILES: header files to be installed to DST_DIR/include
# @param LIB_FILES: lib files to be installed to DST_DIR/lib
# @param KO_FILES: ko files to be installed to DST_DIR/ko
# @param DEPENDS: dependent targets for the executables
function(out_install)
    set(opt_args "")
    set(one_val_args TARGET WORK_DIR DST_DIR)
    set(multi_val_args HEADER_FILES LIB_FILES KO_FILES DEPENDS)
    cmake_parse_arguments(OI "${opt_args}" "${one_val_args}" "${multi_val_args}" ${ARGN})

    if(NOT "${OI_DST_DIR}" STREQUAL "${SDK_OUT_DIR}" AND
       NOT "${OI_DST_DIR}" STREQUAL "${SDK_OUT_MAIN_DIR}" AND
       NOT "${OI_DST_DIR}" STREQUAL "${SDK_OUT_MAIN_RTOS_DIR}" AND
       NOT "${OI_DST_DIR}" STREQUAL "${SDK_OUT_MAIN_RICHOS_DIR}" AND
       NOT "${OI_DST_DIR}" STREQUAL "${SDK_OUT_MCU_DIR}")
        message(FATAL_ERROR "Illegal DST_DIR: ${OI_DST_DIR}!!!")
    endif()

    if(OI_WORK_DIR)
        set(workdir ${OI_WORK_DIR})
    else()
        set(workdir ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    add_custom_target(${OI_TARGET} ALL
        WORKING_DIRECTORY ${workdir}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SDK_OUT_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OI_DST_DIR}
    )

    if(OI_DEPENDS)
        add_dependencies(${OI_TARGET} ${OI_DEPENDS})
    endif()

    if(OI_HEADER_FILES)
        foreach(header_file ${OI_HEADER_FILES})
            add_custom_command(TARGET ${OI_TARGET} POST_BUILD
                WORKING_DIRECTORY ${workdir}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${OI_DST_DIR}/${SDK_OUT_HEADER_DIR_NAME}
                COMMAND chmod +x ${SDK_SCRIPTS_DIR}/install_file.sh
                COMMAND ${SDK_SCRIPTS_DIR}/install_file.sh '${header_file}' ${OI_DST_DIR}/${SDK_OUT_HEADER_DIR_NAME} 0
            )
        endforeach()
    endif()

    if(OI_LIB_FILES)
        foreach(lib_file ${OI_LIB_FILES})
            add_custom_command(TARGET ${OI_TARGET} POST_BUILD
                WORKING_DIRECTORY ${workdir}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${OI_DST_DIR}/${SDK_OUT_LIB_DIR_NAME}
                COMMAND chmod +x ${SDK_SCRIPTS_DIR}/install_file.sh
                COMMAND ${SDK_SCRIPTS_DIR}/install_file.sh '${lib_file}' ${OI_DST_DIR}/${SDK_OUT_LIB_DIR_NAME} 0
            )
        endforeach()
    endif()

    if(OI_KO_FILES)
        foreach(ko_file ${OI_KO_FILES})
            add_custom_command(TARGET ${OI_TARGET} POST_BUILD
                WORKING_DIRECTORY ${workdir}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${OI_DST_DIR}/${SDK_OUT_KO_DIR_NAME}
                COMMAND chmod +x ${SDK_SCRIPTS_DIR}/install_file.sh
                COMMAND ${SDK_SCRIPTS_DIR}/install_file.sh '${ko_file}' ${OI_DST_DIR}/${SDK_OUT_KO_DIR_NAME} 0
            )
        endforeach()
    endif()
endfunction()
