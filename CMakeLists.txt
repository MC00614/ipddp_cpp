cmake_minimum_required(VERSION 3.12)
project(ipddp)

find_package(Python3 COMPONENTS Development REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
# find_package(OpenMP REQUIRED)

# autodiff requires a c++17 enabled compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/ipddp/
    ${CMAKE_SOURCE_DIR}/eigen/
    ${CMAKE_SOURCE_DIR}/matplotlib-cpp/
    ${CMAKE_SOURCE_DIR}/autodiff/
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/example/model/
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -Wpedantic -DDEBUG -fno-omit-frame-pointer")
    # fsanitize=address is not compatible with shared pointer
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -Wpedantic -DDEBUG -fsanitize=address -fno-omit-frame-pointer")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

# Main Executable
add_executable(${PROJECT_NAME} example/main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE -O2)
target_link_libraries(${PROJECT_NAME} PRIVATE ${Python3_LIBRARIES})
# target_link_libraries(${PROJECT_NAME} PRIVATE ${Python3_LIBRARIES} OpenMP::OpenMP_CXX)

# Examples
if (BUILD_EXAMPLE)
    foreach(EXAMPLE rocket2d rocket3d point3d drone3d rpo)
        add_executable(${EXAMPLE} example/${EXAMPLE}.cpp)
        target_include_directories(${EXAMPLE} PRIVATE ${INCLUDE_DIRS})
        target_compile_options(${EXAMPLE} PRIVATE -O2)
        target_link_libraries(${EXAMPLE} PRIVATE ${Python3_LIBRARIES})
    endforeach()
endif()