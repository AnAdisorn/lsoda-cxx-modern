cmake_minimum_required(VERSION 3.10)

project(lsoda-cxx-modern VERSION 0.0.1)

############################################################
# Compiler
############################################################

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall -Wextra -O3)

############################################################
# Create a library
############################################################

include_directories(include)

############################################################
# Create tests
############################################################
# Enable testing for the project
enable_testing()

add_executable(test_daxpy test/test_daxpy.cpp)
add_test(NAME test_daxpy
    COMMAND test_daxpy)

add_executable(test_ddot test/test_ddot.cpp)
add_test(NAME test_ddot
    COMMAND test_ddot)

add_executable(test_dscal test/test_dscal.cpp)
add_test(NAME test_dscal
    COMMAND test_dscal)

add_executable(test_idamax test/test_idamax.cpp)
add_test(NAME test_idamax
    COMMAND test_idamax)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_daxpy PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(test_ddot PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(test_dscal PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(test_idamax PUBLIC OpenMP::OpenMP_CXX)
endif()
