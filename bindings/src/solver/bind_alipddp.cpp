#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "alipddp/alipddp.h"
#include "solver/bind_alipddp.h"

namespace py = pybind11;

void bindALIPDDP(py::module& m) {
    py::class_<ALIPDDP<double>>(m, "ALIPDDP")
        .def(py::init<OptimalControlProblem<double>&>(), py::arg("ocp"))
        .def("init", &ALIPDDP<double>::init)
        .def("solve", &ALIPDDP<double>::solve)
        .def("getResX", &ALIPDDP<double>::getResX)
        .def("getResU", &ALIPDDP<double>::getResU)
        .def("getAllCost", &ALIPDDP<double>::getAllCost);
}