#include "solver/bind_param.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bindParam(py::module& m) {
    py::class_<Param>(m, "Param")
        .def(py::init<>())
        .def_readwrite("max_iter", &Param::max_iter)
        .def_readwrite("max_inner_iter", &Param::max_inner_iter)
        .def_readwrite("tolerance", &Param::tolerance)
        .def_readwrite("mu", &Param::mu)
        .def_readwrite("muT", &Param::muT)
        .def_readwrite("mu_mul", &Param::mu_mul)
        .def_readwrite("mu_exp", &Param::mu_exp)
        .def_readwrite("mu_min", &Param::mu_min)
        .def_readwrite("rho", &Param::rho)
        .def_readwrite("rho_mul", &Param::rho_mul)
        .def_readwrite("rho_max", &Param::rho_max)
        .def_readwrite("max_step_iter", &Param::max_step_iter)
        .def_readwrite("reg1_exp", &Param::reg1_exp)
        .def_readwrite("reg1_min", &Param::reg1_min)
        .def_readwrite("reg2_exp", &Param::reg2_exp)
        .def_readwrite("reg2_min", &Param::reg2_min)
        .def_readwrite("max_regularization", &Param::max_regularization)
        .def_readwrite("corr_p_min", &Param::corr_p_min)
        .def_readwrite("corr_d_min", &Param::corr_d_min)
        .def_readwrite("corr_p_mul", &Param::corr_p_mul)
        .def_readwrite("corr_d_mul", &Param::corr_d_mul)
        .def_readwrite("max_inertia_correction", &Param::max_inertia_correction)
        .def_readwrite("forward_early_termination", &Param::forward_early_termination)
        .def_readwrite("forward_filter", &Param::forward_filter)
        .def_readwrite("forward_cost_threshold", &Param::forward_cost_threshold)
        .def_readwrite("is_quaternion_in_state", &Param::is_quaternion_in_state)
        .def_readwrite("quaternion_idx", &Param::quaternion_idx);
}
