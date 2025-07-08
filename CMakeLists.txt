cmake_minimum_required(VERSION 3.12)
project(alipddp)

find_package(Python3 COMPONENTS Development REQUIRED)
# find_package(OpenMP REQUIRED)

# autodiff requires a c++17 enabled compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/alipddp/
    ${CMAKE_SOURCE_DIR}/problem_maker/
    ${CMAKE_SOURCE_DIR}/eigen/
    ${CMAKE_SOURCE_DIR}/matplotlib-cpp/
    ${Python3_INCLUDE_DIRS}
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -Wpedantic -DDEBUG -fno-omit-frame-pointer")
    # fsanitize=address is not compatible with shared pointer
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -Wpedantic -DDEBUG -fsanitize=address -fno-omit-frame-pointer")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()


add_library(alipddp STATIC alipddp/alipddp/alipddp.cpp)
target_include_directories(alipddp PRIVATE ${INCLUDE_DIRS})

# # Examples
# if (BUILD_EXAMPLE)
foreach(EXAMPLE rocket_2d
                point_mass_3d_basic
                point_mass_3d_simple)
    add_executable(${EXAMPLE} problem_examples/${EXAMPLE}.cpp)
    target_include_directories(${EXAMPLE} PRIVATE ${INCLUDE_DIRS})
    target_compile_options(${EXAMPLE} PRIVATE -O3)
    target_link_libraries(${EXAMPLE} PRIVATE ${Python3_LIBRARIES} alipddp)
endforeach()
# endif()