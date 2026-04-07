add_library(impurity gw_impurity_solver.cpp)
target_include_directories(impurity PUBLIC .)
target_link_libraries(impurity PUBLIC GREEN::NDARRAY GREEN::UTILS GREEN::GRIDS GREEN::SYMMETRY lsqcpp::lsqcpp)
