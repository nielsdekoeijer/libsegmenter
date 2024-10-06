cmake_minimum_required(VERSION 3.10)
project(Segmenter VERSION 1.0 LANGUAGES CXX)
set(NAMESPACE Segmenter)
set(CMAKE_CXX_STANDARD 17)

option(SEGMENTER_BUILD_PYBINDINGS OFF)
if(${SEGMENTER_BUILD_PYBINDINGS})
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

include(cmake/CPM.cmake)
CPMAddPackage("gh:macoustics/split-radix-fft#0.4.1")

add_library(${PROJECT_NAME} INTERFACE)
add_library(${NAMESPACE}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/libsegmenter/base/)
target_link_libraries(${PROJECT_NAME} INTERFACE SplitRadixFft::SplitRadixFft)

if(${SEGMENTER_BUILD_PYBINDINGS})
    message(STATUS "Building python bindings for libsegmenter...")
    set(PYBIND11_NEWPYTHON ON)
    CPMAddPackage("gh:pybind/pybind11@2.12.0")
    pybind11_add_module(bindings SHARED 
        src/libsegmenter/bindings/Bindings.cpp 
        src/libsegmenter/bindings/SegmenterBindings.cpp 
        src/libsegmenter/bindings/ColaBindings.cpp 
        src/libsegmenter/bindings/WindowsBindings.cpp 
    )
    target_link_libraries(bindings PRIVATE Segmenter::Segmenter)
    install(TARGETS bindings LIBRARY DESTINATION libsegmenter)
endif()
