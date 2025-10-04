cmake_minimum_required(VERSION 3.10)
project(linear_regression VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(linear_regression ../src/linear_regression.cpp)

find_package(xtensor CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
target_link_libraries(linear_regression PRIVATE xtensor Boost::program_options)