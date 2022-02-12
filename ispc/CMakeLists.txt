cmake_minimum_required(VERSION 3.20.0)

project(BoxBlurISPC VERSION 0.2 LANGUAGES CXX ISPC)

add_library(boxblur_ispc SHARED
    boxblur.ispc
    source.cpp)

set_target_properties(boxblur_ispc PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig QUIET MODULE)

if(PKG_CONFIG_FOUND)
    pkg_search_module(VS vapoursynth)

    if(VS_FOUND)
        message(STATUS "Found VapourSynth r${VS_VERSION}")

        cmake_path(APPEND install_dir ${VS_LIBDIR} vapoursynth)
        target_include_directories(boxblur_ispc PRIVATE ${VS_INCLUDE_DIRS})

        install(TARGETS boxblur_ispc LIBRARY DESTINATION ${install_dir})
    endif()
endif()

if(NOT VS_FOUND)
    set(VS_INCLUDE_DIR "" CACHE PATH "Path to VapourSynth headers")

    if(VS_INCLUDE_DIR EQUAL "")
        message(WARNING "VapourSynth not found")
    endif()

    target_include_directories(boxblur_ispc PRIVATE ${VS_INCLUDE_DIR})

    install(TARGETS boxblur_ispc LIBRARY RUNTIME)
endif()

find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --long --always
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE VCS_TAG
    )
    if(VCS_TAG)
        string(STRIP ${VCS_TAG} VCS_TAG)
    endif()
endif()

if(VCS_TAG)
    message(STATUS "VapourSynth-BoxBlur ${VCS_TAG}")
else()
    message(WARNING "unknown plugin version")
    set(VCS_TAG "unknown")
endif()

configure_file(../source/config.h.in config.h)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
