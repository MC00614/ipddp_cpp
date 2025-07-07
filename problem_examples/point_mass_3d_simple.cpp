#include "optimal_control_problem.h"
#include "ipddp.h"

#include <cmath>
#include <chrono>

int main() {
    using Scalar = double;
    
    int N = 100;
    OptimalControlProblem<Scalar> problem(N);


    Matrix<Scalar> A = Matrix<Scalar>::Zero(6,6);
    Matrix<Scalar> B = Matrix<Scalar>::Zero(6,3);
    double dt = 0.1;
    A.block(0,0,3,3) = Matrix<Scalar>::Identity(3,3);
    A.block(0,3,3,3) = dt * Matrix<Scalar>::Identity(3,3);
    A.block(3,3,3,3) = Matrix<Scalar>::Identity(3,3);
    B.block(3,0,3,3) = dt * Matrix<Scalar>::Identity(3,3);
    auto dynamics = std::make_shared<LinearDiscreteDynamics<Scalar>>(A, B);
    Vector<Scalar> c = Vector<Scalar>::Zero(6);
    c(5) = -9.81 * dt;
    dynamics->setC(c);
    problem.setStageDynamics(dynamics);


    Matrix<Scalar> Q = Matrix<Scalar>::Zero(6,6);
    Matrix<Scalar> R = 1e-6 * Matrix<Scalar>::Identity(3,3);
    auto stage_cost = std::make_shared<QuadraticStageCost<Scalar>>(Q, R);
    problem.setStageCost(stage_cost);


    Matrix<Scalar> QT = Matrix<Scalar>::Zero(6,6);
    auto terminal_cost = std::make_shared<QuadraticTerminalCost<Scalar>>(QT);
    problem.setTerminalCost(terminal_cost);


    Matrix<Scalar> CT = Matrix<Scalar>::Identity(6,6);
    Vector<Scalar> c0T = Vector<Scalar>::Zero(6);
    c0T(2) = -1.0;
    auto terminal_constraint = std::make_shared<LinearTerminalConstraint<Scalar>>(CT, c0T, ConstraintType::EQ);
    problem.addTerminalConstraint(terminal_constraint);

    Eigen::VectorXd x0(6);
    x0 << 5.0, 5.0, 10.0, -0.2, -0.2, -0.3;
    problem.setInitialState(0, x0);

    Eigen::VectorXd u0(3);
    u0 << 0.0, 0.0, 9.81;
    problem.setInitialControl(u0);


    Param param;
    ALIPDDP<Scalar> solver(problem);
    solver.init(param);
    solver.solve();

    // Result
    auto Xres = solver.getResX();
    auto Ures = solver.getResU();

    std::cout << "Final state:\n" << Xres[N].transpose() << std::endl;
    std::cout << "Final input:\n" << Ures[N-1].transpose() << std::endl;

    return 0;
}