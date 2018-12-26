cmake_minimum_required(VERSION 3.10)
project(Image_based_Tree_Modeling)

set(CMAKE_CXX_STANDARD 11)

add_executable(Image_based_Tree_Modeling main.cpp globalmatting.h globalmatting.cpp guidedfilter.h guidedfilter.cpp)