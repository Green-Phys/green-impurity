#CMakeLists.txt
#
#Author : Fabian Meyer
#Created On : 30 Mar 2022
#License : MIT

add_library(lsqcpp INTERFACE) 
target_include_directories(lsqcpp INTERFACE "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(lsqcpp INTERFACE Eigen3::Eigen) 
add_library(lsqcpp::lsqcpp ALIAS lsqcpp)

install(DIRECTORY lsqcpp TYPE INCLUDE)
