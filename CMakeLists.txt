cmake_minimum_required(VERSION 3.5)
project(ipddp)

find_package(PythonLibs REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)

# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -Wall -Wextra -Wpedantic -DDEBUG")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")

include_directories(
    ipddp/
)

# autodiff requires a c++17 enabled compiler
set(CMAKE_CXX_STANDARD 17)          # ensure cmake instructs compiler to use C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON) # ensure the C++ standard given before is actually used
# set(CMAKE_CXX_EXTENSIONS OFF)       # avoid compile flags of the type -std=gnu++1z added by cmake

include_directories(
    eigen/
    matplotlibcpp/
    autodiff/
    ${PYTHON_INCLUDE_DIRS}
    example/model/
)

add_executable(${PROJECT_NAME}
    example/main.cpp
)

target_compile_options(${PROJECT_NAME} PRIVATE -O2)

target_link_libraries(${PROJECT_NAME}
    ${PYTHON_LIBRARIES}
)