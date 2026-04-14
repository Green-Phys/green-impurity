add_library(impurity
    bath_fitting.cpp
    ed_impurity_solver.cpp
    inchworm_impurity_solver.cpp
    gw_impurity_solver.cpp
    impurity_solver.cpp
)
target_include_directories(impurity PUBLIC .)
target_link_libraries(impurity PUBLIC GREEN::NDARRAY GREEN::UTILS GREEN::GRIDS GREEN::SYMMETRY lsqcpp::lsqcpp)
