cmake_minimum_required(VERSION 3.17)
project(compress_binding)

set(CMAKE_CXX_STANDARD 14)

find_package(pybind11 CONFIG REQUIRED)

# Create an extension module
pybind11_add_module(compress_binding main.cpp)