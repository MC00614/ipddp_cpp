#pragma once

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "alipddp/alipddp.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bindALIPDDP(py::module& m);