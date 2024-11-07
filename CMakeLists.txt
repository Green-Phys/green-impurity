cmake_minimum_required(VERSION 3.20)
project(GREEN-ED C CXX)
set(CMAKE_CXX_STANDARD 17)

################GREEN RELEASE VERSION####################
set(GREEN_RELEASE origin/main)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(GreenDeps)

add_green_dependency(green-ndarray)
add_green_dependency(green-symmetry)
add_green_dependency(green-grids)
find_package(Eigen3 REQUIRED)

add_subdirectory(deps)
add_subdirectory(src)

add_library(GREEN::IMPURITY ALIAS impurity)

option(Build_Tests "Build tests" ON)
if (Build_Tests)
    enable_testing()
    add_subdirectory(test)
endif(Build_Tests)
