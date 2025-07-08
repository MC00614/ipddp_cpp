#include "solver/bind_alipddp.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bindALIPDDP(py::module& m) {
    using Solver = ALIPDDP<double>;

    py::class_<Solver>(m, "ALIPDDP")
        .def(py::init<OptimalControlProblem<double>&>(), py::arg("ocp"), py::keep_alive<1, 2>())
        // .def(py::init<std::shared_ptr<OCP>>(), py::arg("ocp"))
        .def("init", &Solver::init)
        .def("solve", &Solver::solve)
        .def("getResX", &Solver::getResX)
        .def("getResU", &Solver::getResU)
        .def("getAllCost", &Solver::getAllCost);
}