set_target_properties(alipddp_library PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(alipddp
    src/function/bind_function.cpp
    src/problem/bind_problem.cpp
    src/solver/bind_param.cpp
    src/solver/bind_alipddp.cpp
    src/bind.cpp
)

target_link_libraries(alipddp
    PRIVATE alipddp_library
)
target_include_directories(alipddp PRIVATE ${INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)

