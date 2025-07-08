#pragma once

#include "function/override/py_dynamics.h"
#include "function/override/py_stage_constraint.h"
#include "function/override/py_stage_cost.h"
#include "function/override/py_terminal_constraint.h"
#include "function/override/py_terminal_cost.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bindFunction(py::module& m);
