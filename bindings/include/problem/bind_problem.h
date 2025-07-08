#pragma once

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "optimal_control_problem.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bindProblem(py::module& m);
