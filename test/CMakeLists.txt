project(ed-tests)

Include(FetchContent)

FetchContent_Declare(Catch2 
                     GIT_REPOSITORY https://github.com/catchorg/Catch2.git
                     GIT_TAG v3.5.3 # or a later release
                     )

FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_executable(ed_test bath_fitting_test.cpp)
target_link_libraries(ed_test PRIVATE Catch2::Catch2WithMain GREEN::ED)

include(CTest)
include(Catch)
catch_discover_tests(ed_test)
