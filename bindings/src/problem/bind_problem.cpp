#include "problem/bind_problem.h"

void bindProblem(py::module& m) {
    py::class_<OptimalControlProblem<double>>(m, "OptimalControlProblem")
        .def(py::init<int>())
        .def("setInitialState", py::overload_cast<int, const Vector<double>&>(&OptimalControlProblem<double>::setInitialState))
        .def("setInitialState", py::overload_cast<const Vector<double>&>(&OptimalControlProblem<double>::setInitialState))
        .def("setInitialControl", py::overload_cast<int, const Vector<double>&>(&OptimalControlProblem<double>::setInitialControl))
        .def("setInitialControl", py::overload_cast<const Vector<double>&>(&OptimalControlProblem<double>::setInitialControl))
        .def("setStageDynamics", py::overload_cast<int, std::shared_ptr<DiscreteDynamicsBase<double>>>(&OptimalControlProblem<double>::setStageDynamics))
        .def("setStageDynamics", py::overload_cast<std::shared_ptr<DiscreteDynamicsBase<double>>>(&OptimalControlProblem<double>::setStageDynamics))
        .def("setStageCost", py::overload_cast<int, std::shared_ptr<StageCostBase<double>>>(&OptimalControlProblem<double>::setStageCost))
        .def("setStageCost", py::overload_cast<std::shared_ptr<StageCostBase<double>>>(&OptimalControlProblem<double>::setStageCost))
        .def("setTerminalCost", &OptimalControlProblem<double>::setTerminalCost)
        .def("addStageConstraint", py::overload_cast<int, std::shared_ptr<StageConstraintBase<double>>>(&OptimalControlProblem<double>::addStageConstraint))
        .def("addStageConstraint", py::overload_cast<std::shared_ptr<StageConstraintBase<double>>>(&OptimalControlProblem<double>::addStageConstraint))
        .def("addTerminalConstraint", &OptimalControlProblem<double>::addTerminalConstraint)

        .def("getHorizon", &OptimalControlProblem<double>::getHorizon)
        .def("getInitialState", &OptimalControlProblem<double>::getInitialState, py::return_value_policy::reference_internal)
        .def("getInitialControl", &OptimalControlProblem<double>::getInitialControl, py::return_value_policy::reference_internal)
        .def("getDynamics", &OptimalControlProblem<double>::getDynamics, py::return_value_policy::reference_internal)
        .def("getStageCost", &OptimalControlProblem<double>::getStageCost, py::return_value_policy::reference_internal)
        .def("getTerminalCost", &OptimalControlProblem<double>::getTerminalCost, py::return_value_policy::reference_internal)
        .def("getStageConstraints", &OptimalControlProblem<double>::getStageConstraints, py::return_value_policy::reference_internal)
        .def("getTerminalConstraints", &OptimalControlProblem<double>::getTerminalConstraints, py::return_value_policy::reference_internal);

    py::enum_<ConstraintType>(m, "ConstraintType")
        .value("NO", ConstraintType::NO)
        .value("SOC", ConstraintType::SOC)
        .value("EQ", ConstraintType::EQ)
        .export_values();
}
