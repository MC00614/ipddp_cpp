#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"

#include <cmath>
#include <chrono>

template <typename Scalar>
class TempConstraint : public StageConstraintBase<Scalar> {
private:
    Scalar umax;
    Scalar umin;

public:
    TempConstraint() {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 4;

        umax = 10.0;
        umin = -10.0;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Vector<Scalar> c_n(4);
        c_n(0) = umax - u(0);
        c_n(1) = u(0) - umin;
        c_n(2) = u(1);
        c_n(3) = u(2);
        return -c_n;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(4, 2);
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(4, 3);
        J(0, 0) = -1.0;
        J(1, 0) = 1.0;
        J(2, 1) = 1.0;
        J(3, 2) = 1.0;
        return -J;
    }
};

template <typename Scalar>
class EqualityConstraint : public StageConstraintBase<Scalar> {
public:
    EqualityConstraint() {
        this->constraint_type = ConstraintType::EQ;
        this->dim_c = 1;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Vector<Scalar> c_n(1);
        c_n(0) = u(1) - u(2) - u(0) * x(1);
        return c_n;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J(1, 2);
        J.setZero();
        J(0, 1) = - u(0);
        return J;
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J(1, 3);
        J.setZero();
        J(0, 0) = - x(1);
        J(0, 1) = 1.0;
        J(0, 2) = - 1.0;
        return J;
    }
};

int main() {
    using Scalar = double;

    int N = 100;
    OptimalControlProblem<Scalar> problem(N);

    Vector<Scalar> x0(2);
    x0 << 0.0, 0.0;
    problem.setInitialState(0, x0);

    Vector<Scalar> u0(3);
    u0 << 0.01, 0.01, 0.01;
    problem.setInitialControl(u0);

     Matrix<Scalar> A(2, 2);
    A << 1.0, 0.05,
         0.0, 1.0;
    Matrix<Scalar> B(2, 3);
    B.setZero();
    B(1, 0) = 0.05;
    problem.setStageDynamics(std::make_shared<LinearDiscreteDynamics<Scalar>>(A, B));

    Vector<Scalar> Q = Vector<Scalar>::Zero(2);
    Vector<Scalar> R = Vector<Scalar>::Zero(3);
    R(1) = 0.05;
    R(2) = 0.05;
    auto stage_cost = std::make_shared<VectorStageCost<Scalar>>(Q, R);
    problem.setStageCost(stage_cost);
    
    Matrix<Scalar> QT = Matrix<Scalar>::Zero(2,2);
    QT(0, 0) = 1000.0;
    QT(1, 1) = 1000.0;
    Vector<Scalar> x_ref = Vector<Scalar>::Zero(2);
    x_ref(0) = 1.0;
    auto terminal_cost = std::make_shared<ErrorQuadraticTerminalCost<Scalar>>(QT, x_ref);
    problem.setTerminalCost(terminal_cost);

    problem.addStageConstraint(std::make_shared<EqualityConstraint<Scalar>>());
    problem.addStageConstraint(std::make_shared<TempConstraint<Scalar>>());

    Param param;
    param.max_inner_iter = 10;
    param.rho_mul = 1000.0;
    // param.forward_cost_threshold = 1.0;
    // param.mu_mul = 0.01;

    clock_t start = clock();
    ALIPDDP<Scalar> solver(problem);
    solver.init(param);
    solver.solve();

    clock_t finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "\nIn Total : " << duration << " Seconds" << std::endl;

    // Parse Result
    std::vector<Eigen::VectorXd> X_result = solver.getResX();
    std::vector<Eigen::VectorXd> U_result = solver.getResU();
    std::vector<double> all_cost = solver.getAllCost();

    std::cout << "X_result = " << std::endl;
    for (int k = 0; k < problem.getHorizon() + 1; ++k) {
        std::cout << X_result[k].transpose() << std::endl;
    }

    std::cout << "U_result = " << std::endl;
    for (int k = 0; k < problem.getHorizon(); ++k) {
        std::cout << U_result[k].transpose() << std::endl;
    }

    std::cout<<"X_last = \n"<<X_result[problem.getHorizon()].transpose()<<std::endl;
    std::cout<<"U_last = \n"<<U_result[problem.getHorizon() - 1].transpose()<<std::endl;

    return 0;
}