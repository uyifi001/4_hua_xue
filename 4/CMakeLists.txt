cmake_minimum_required(VERSION 3.16)
project(chemo_mechanical_dealii LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(chemo_mechanical_dealii dealii_mirror.cpp)
target_compile_options(chemo_mechanical_dealii PRIVATE -O2 -Wall -Wextra -Wpedantic)
