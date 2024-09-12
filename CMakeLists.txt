cmake_minimum_required(VERSION 3.10)

project(lsoda-cxx-modern VERSION 0.0.1)

############################################################
# Compiler
############################################################

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall -Wextra -O3 -march=native)

############################################################
# Create a library
############################################################

include_directories(include)

############################################################
# Create tests
############################################################
# Enable testing for the project
enable_testing()

add_executable(test_idamax test/test_idamax.cpp)
add_test(NAME test_idamax
    COMMAND test_idamax)

add_executable(test_dgefa_dgesl test/test_dgefa_dgesl.cpp)
add_test(NAME test_dgefa_dgesl
    COMMAND test_dgefa_dgesl)

add_executable(test_vmnorm_fnorm test/test_vmnorm_fnorm.cpp)
add_test(NAME test_vmnorm_fnorm
    COMMAND test_vmnorm_fnorm)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_idamax PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(test_dgefa_dgesl PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(test_vmnorm_fnorm PUBLIC OpenMP::OpenMP_CXX)
endif()
