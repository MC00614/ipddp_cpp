#include "solver/bind_alipddp.h"
#include "solver/bind_param.h"
#include "problem/bind_problem.h"
#include "function/bind_function.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(alipddp, m) {
    bindFunction(m);
    bindProblem(m);
    bindParam(m);
    bindALIPDDP(m);
}