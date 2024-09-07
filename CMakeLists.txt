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

add_library(lsoda lib/lsoda.hpp)

############################################################
# Create tests
############################################################
# Enable testing for the project
enable_testing()

# Test creating tensor
add_executable(test_graphene_fn tests/test_graphene_fn.cc)
target_link_libraries(test_graphene_fn graphene_fn)
add_test(NAME test_graphene_fn
         COMMAND test_graphene_fn)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(polarization PUBLIC OpenMP::OpenMP_CXX)
endif()
