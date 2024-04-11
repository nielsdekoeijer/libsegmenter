cmake_minimum_required(VERSION 3.10)
project(Segmenter VERSION 1.0 LANGUAGES CXX)
set(NAMESPACE Segmenter)

add_library(${NAMESPACE}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/libsegmenter)
