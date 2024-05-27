cmake_minimum_required(VERSION 3.5)
project(soc_ipddp)

find_package(PythonLibs REQUIRED)

include_directories(
    soc_ipddp/
    model/
    eigen/
    matplotlibcpp/
    ${PYTHON_INCLUDE_DIRS}
    )

add_executable(${PROJECT_NAME}
    src/main.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${PYTHON_LIBRARIES}
)