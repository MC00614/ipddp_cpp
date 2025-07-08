#pragma once

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "param.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bindParam(py::module& m);