cmake_minimum_required(VERSION 3.15)
set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "vckpg toolchain file")

project(ml VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(xtensor CONFIG REQUIRED)
find_package(xtensor-blas CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)

add_executable(linear_regression lin_reg.cpp src/LinearRegression.cpp src/Dataset.cpp)
target_link_libraries(linear_regression PRIVATE xtensor xtensor-blas Boost::program_options)
target_include_directories(linear_regression PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(support_vector_machine svm.cpp src/SupportVectorMachine.cpp src/Dataset.cpp)
target_link_libraries(support_vector_machine PRIVATE xtensor xtensor-blas Boost::program_options)
target_include_directories(support_vector_machine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(perceptron perc.cpp src/Perceptron.cpp src/Dataset.cpp)
target_link_libraries(perceptron PRIVATE xtensor xtensor-blas Boost::program_options)
target_include_directories(perceptron PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
