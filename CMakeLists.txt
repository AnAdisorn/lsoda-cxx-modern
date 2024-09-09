cmake_minimum_required(VERSION 3.10)

project(lsoda-cxx-modern VERSION 0.0.1)

############################################################
# Compiler
############################################################

set(CMAKE_CXX_STANDARD 11)

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

# test daxpy
add_executable(test_daxpy test/test_daxpy.cpp)
add_test(NAME test_daxpy
         COMMAND test_daxpy)

# test daxpy
add_executable(test_ddot test/test_ddot.cpp)
add_test(NAME test_ddot
         COMMAND test_ddot)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_daxpy PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(test_ddot PUBLIC OpenMP::OpenMP_CXX)
endif()
