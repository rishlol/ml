cmake_minimum_required(VERSION 3.10)
set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "vckpt toolchain file")

project(ml VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(xtensor CONFIG REQUIRED)
find_package(xtensor-blas CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
link_libraries(xtensor xtensor-blas Boost::program_options)

add_executable(linear_regression lin_reg.cpp src/LinearRegression.cpp src/Dataset.cpp)
target_include_directories(linear_regression PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
