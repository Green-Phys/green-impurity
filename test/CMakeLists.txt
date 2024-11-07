project(ed-tests)

Include(FetchContent)

FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.3 # or a later release
)

FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_executable(impurity_test bath_fitting_test.cpp
        impurity_solver_test.cpp)
target_compile_definitions(impurity_test PRIVATE TEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data"
        GRID_PATH="${grids_SOURCE_DIR}/data")
target_link_libraries(impurity_test PRIVATE Catch2::Catch2WithMain GREEN::IMPURITY)

include(CTest)
include(Catch)
catch_discover_tests(impurity_test)
