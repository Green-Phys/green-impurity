project(impurity)

add_library(impurity INTERFACE)
include_directories(.)
target_include_directories(impurity INTERFACE .)
target_link_libraries(impurity INTERFACE GREEN::NDARRAY GREEN::GRIDS GREEN::SYMMETRY lsqcpp::lsqcpp)
