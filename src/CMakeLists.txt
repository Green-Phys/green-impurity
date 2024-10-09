project(ed)

add_library(ed INTERFACE)
include_directories(.)
target_include_directories(ed INTERFACE .)
target_link_libraries(ed INTERFACE GREEN::NDARRAY lsqcpp::lsqcpp)
