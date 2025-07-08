#include "function/bind_function.h"
#include "optimal_control_problem.h"


#include <pybind11/stl.h>
#include <pybind11/eigen.h>

void bindFunction(py::module& m) {
    // Dynamics
    py::class_<DiscreteDynamicsBase<double>, PyDiscreteDynamics<double>, std::shared_ptr<DiscreteDynamicsBase<double>>>(m, "DiscreteDynamicsBase")
        .def(py::init<>())
        .def("f", &DiscreteDynamicsBase<double>::f)
        .def("fx", &DiscreteDynamicsBase<double>::fx)
        .def("fu", &DiscreteDynamicsBase<double>::fu)
        .def("getDimX", &DiscreteDynamicsBase<double>::getDimX)
        .def("getDimU", &DiscreteDynamicsBase<double>::getDimU)
        .def("getDT", &DiscreteDynamicsBase<double>::getDT);

    py::class_<LinearDiscreteDynamics<double>, DiscreteDynamicsBase<double>, std::shared_ptr<LinearDiscreteDynamics<double>>>(m, "LinearDiscreteDynamics")
        .def(py::init<>())
        .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&>(), py::arg("A"), py::arg("B"))
        .def("setA", &LinearDiscreteDynamics<double>::setA)
        .def("setB", &LinearDiscreteDynamics<double>::setB)
        .def("setC", &LinearDiscreteDynamics<double>::setC);

    // Stage Cost
    py::class_<StageCostBase<double>, PyStageCost, std::shared_ptr<StageCostBase<double>>>(m, "StageCostBase")
        .def("q", &StageCostBase<double>::q)
        .def("qx", &StageCostBase<double>::qx)
        .def("qu", &StageCostBase<double>::qu)
        .def("qxx", &StageCostBase<double>::qxx)
        .def("quu", &StageCostBase<double>::quu)
        .def("qxu", &StageCostBase<double>::qxu);

    py::class_<QuadraticStageCost<double>, StageCostBase<double>, std::shared_ptr<QuadraticStageCost<double>>>(m, "QuadraticStageCost")
        .def(py::init<>())
        .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&>(), py::arg("Q"), py::arg("R"))
        .def("setQ", &QuadraticStageCost<double>::setQ)
        .def("setR", &QuadraticStageCost<double>::setR);

    py::class_<ScalarQuadraticStageCost<double>, StageCostBase<double>, std::shared_ptr<ScalarQuadraticStageCost<double>>>(m, "ScalarQuadraticStageCost")
        .def(py::init<>())
        .def(py::init<const double&, const double&>(), py::arg("Q"), py::arg("R"))
        .def("setQ", &ScalarQuadraticStageCost<double>::setQ)
        .def("setR", &ScalarQuadraticStageCost<double>::setR);


    // Terminal Cost
    py::class_<TerminalCostBase<double>, PyTerminalCost, std::shared_ptr<TerminalCostBase<double>>>(m, "TerminalCostBase")
        .def(py::init<>())
        .def("p", &TerminalCostBase<double>::p)
        .def("px", &TerminalCostBase<double>::px)
        .def("pxx", &TerminalCostBase<double>::pxx);

    py::class_<QuadraticTerminalCost<double>, TerminalCostBase<double>, std::shared_ptr<QuadraticTerminalCost<double>>>(m, "QuadraticTerminalCost")
        .def(py::init<>())
        .def(py::init<const Eigen::MatrixXd&>(), py::arg("Qf"))
        .def("setQf", &QuadraticTerminalCost<double>::setQf);

    py::class_<ScalarQuadraticTerminalCost<double>, TerminalCostBase<double>, std::shared_ptr<ScalarQuadraticTerminalCost<double>>>(m, "ScalarQuadraticTerminalCost")
        .def(py::init<>())
        .def(py::init<const double&>(), py::arg("Qf"))
        .def("setQf", &ScalarQuadraticTerminalCost<double>::setQf);


    // Stage Constraint
    py::class_<StageConstraintBase<double>, PyStageConstraint, std::shared_ptr<StageConstraintBase<double>>>(m, "StageConstraintBase")
        .def(py::init<>())
        .def("c", &StageConstraintBase<double>::c)
        .def("cx", &StageConstraintBase<double>::cx)
        .def("cu", &StageConstraintBase<double>::cu)
        .def("setConstraintType", &StageConstraintBase<double>::setConstraintType)
        .def("setDimC", &StageConstraintBase<double>::setDimC)
        .def("getConstraintType", &StageConstraintBase<double>::getConstraintType)
        .def("getDimC", &StageConstraintBase<double>::getDimC);

    py::class_<LinearStageConstraint<double>, StageConstraintBase<double>, std::shared_ptr<LinearStageConstraint<double>>>(m, "LinearStageConstraint")
        .def(py::init<>())
        .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::VectorXd&, ConstraintType>(),
            py::arg("Cx"), py::arg("Cu"), py::arg("c0"), py::arg("constraint_type"))
        .def("setCx", &LinearStageConstraint<double>::setCx)
        .def("setCu", &LinearStageConstraint<double>::setCu)
        .def("setC0", &LinearStageConstraint<double>::setC0);


    // Terminal Constraint
    py::class_<TerminalConstraintBase<double>, PyTerminalConstraint, std::shared_ptr<TerminalConstraintBase<double>>>(m, "TerminalConstraintBase")
        .def(py::init<>())
        .def("cT", &TerminalConstraintBase<double>::cT)
        .def("cTx", &TerminalConstraintBase<double>::cTx)
        .def("setConstraintType", &TerminalConstraintBase<double>::setConstraintType)
        .def("setDimCT", &TerminalConstraintBase<double>::setDimCT)
        .def("getConstraintType", &TerminalConstraintBase<double>::getConstraintType)
        .def("getDimCT", &TerminalConstraintBase<double>::getDimCT);

    py::class_<LinearTerminalConstraint<double>, TerminalConstraintBase<double>, std::shared_ptr<LinearTerminalConstraint<double>>>(m, "LinearTerminalConstraint")
        .def(py::init<>())
        .def(py::init<const Eigen::MatrixXd&, const Eigen::VectorXd&, ConstraintType>(),
            py::arg("CTx"), py::arg("cT0"), py::arg("constraint_type"))
        .def("setCTx", &LinearTerminalConstraint<double>::setCTx)
        .def("setcT0", &LinearTerminalConstraint<double>::setcT0);
}
