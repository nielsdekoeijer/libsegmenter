cmake_minimum_required(VERSION 3.10)
project(Segmenter VERSION 1.0 LANGUAGES CXX)
set(NAMESPACE Segmenter)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/CPM.cmake)
CPMAddPackage("gh:macoustics/split-radix-fft#0.4.1")

# boost
# reference: https://github.com/cpm-cmake/CPM.cmake/blob/master/examples/boost/CMakeLists.txt
CPMAddPackage(
  NAME Boost
  VERSION 1.84.0
  URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.xz
  URL_HASH SHA256=2e64e5d79a738d0fa6fb546c6e5c2bd28f88d268a2a080546f74e5ff98f29d0e
  OPTIONS "BOOST_ENABLE_CMAKE ON" "BOOST_INCLUDE_LIBRARIES archive"
)

add_library(${PROJECT_NAME} INTERFACE)
add_library(${NAMESPACE}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/libsegmenter/base/)
target_link_libraries(${PROJECT_NAME} INTERFACE SplitRadixFft::SplitRadixFft)

option(SEGMENTER_BUILD_PYBINDINGS OFF)
message("Python bindings build set to ${SEGMENTER_BUILD_PYBINDINGS}")
if(${SEGMENTER_BUILD_PYBINDINGS})
    set(PYBIND11_NEWPYTHON ON)
    CPMAddPackage("gh:pybind/pybind11@2.12.0")
    pybind11_add_module(bindings SHARED 
        libsegmenter/bindings/Bindings.cpp 
        libsegmenter/bindings/SegmenterBindings.cpp 
        libsegmenter/bindings/ColaBindings.cpp 
        libsegmenter/bindings/WindowsBindings.cpp 
    )
    target_link_libraries(bindings PRIVATE Segmenter::Segmenter)
    install(TARGETS bindings LIBRARY DESTINATION libsegmenter)
endif()
