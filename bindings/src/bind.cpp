#include <pybind11/pybind11.h>

#include "solver/bind_alipddp.h"
#include "solver/bind_param.h"
#include "problem/bind_problem.h"
#include "function/bind_function.h"

namespace py = pybind11;

PYBIND11_MODULE(alipddp, m) {
    bindALIPDDP(m);
    bindParam(m);
    bindProblem(m);
    bindFunction(m);
}