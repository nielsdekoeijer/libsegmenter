cmake_minimum_required(VERSION 3.10)
project(Segmenter VERSION 1.0 LANGUAGES CXX)
set(NAMESPACE Segmenter)

include(cmake/CPM.cmake)
CPMAddPackage("gh:macoustics/split-radix-fft#0.1")


add_library(${PROJECT_NAME} INTERFACE)
add_library(${NAMESPACE}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/libsegmenter)

option(SEGMENTER_BUILD_PYBINDINGS OFF)
message("Python bindings build set to ${SEGMENTER_BUILD_PYBINDINGS}")
if(${SEGMENTER_BUILD_PYBINDINGS})
    set(PYBIND11_NEWPYTHON ON)
    CPMAddPackage("gh:pybind/pybind11@2.12.0")
    pybind11_add_module(bindings libsegmenter/SegmenterBindings.cpp)
    install(TARGETS bindings LIBRARY DESTINATION libsegmenter)
endif()
