find_package(pybind11 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Ensure static lib is built as PIC
set_target_properties(alipddp_library PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(alipddp
    src/bind.cpp

    src/solver/bind_alipddp.cpp
    src/solver/bind_param.cpp
)

target_link_libraries(alipddp
    PRIVATE alipddp_library
)
target_include_directories(alipddp PRIVATE ${INCLUDE_DIRS})
