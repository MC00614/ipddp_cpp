#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "solver/bind_alipddp.h"
#include "solver/bind_param.h"

PYBIND11_MODULE(alipddp, m) {
    bindALIPDDP(m);
    bindParam(m);
}