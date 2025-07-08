#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"

#include <cmath>
#include <chrono>

// Dynamics
template <typename Scalar>
class Rocket2D : public DiscreteDynamicsBase<Scalar> {
private:
    Scalar mass = 10.0;
    Scalar I = 3.33;
    Scalar l = 0.7;
    Scalar gravity = -9.81;

public:
    Rocket2D() {
        this->dim_x = 6;
        this->dim_u = 2;
        this->dt = 0.035;
    }

    Vector<Scalar> f(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Vector<Scalar> x_n(6);

        const Scalar& c = cos(x(4));
        const Scalar& s = sin(x(4));

        x_n(0) = x(0) + this->dt * x(2);
        x_n(1) = x(1) + this->dt * x(3);
        x_n(2) = x(2) + this->dt * (gravity + (c * u(0) - s * u(1)) / mass);
        x_n(3) = x(3) + this->dt * ((s * u(0) + c * u(1)) / mass);
        x_n(4) = x(4) + this->dt * x(5);
        x_n(5) = x(5) - this->dt * (l / I) * u(1);

        return x_n;
    }

    Matrix<Scalar> fx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> Fx = Matrix<Scalar>::Identity(6, 6);

        const Scalar c = cos(x(4));
        const Scalar s = sin(x(4));

        Fx(0, 2) += this->dt;
        Fx(1, 3) += this->dt;
        Fx(4, 5) += this->dt;
        Fx(2, 4) += this->dt * (-s * u(0) - c * u(1)) / mass;
        Fx(3, 4) += this->dt * (c * u(0) - s * u(1)) / mass;

        return Fx;
    }

    Matrix<Scalar> fu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> Fu = Matrix<Scalar>::Zero(6, 2);

        const Scalar c = cos(x(4));
        const Scalar s = sin(x(4));

        Fu(2, 0) =   this->dt * c / mass;
        Fu(2, 1) = - this->dt * s / mass;
        Fu(3, 0) =   this->dt * s / mass;
        Fu(3, 1) =   this->dt * c / mass;
        Fu(5, 1) = - this->dt * l / I;

        return Fu;
    }
};

template <typename Scalar>
class MaxInput : public StageConstraintBase<Scalar> {
private:
    Scalar umax;

public:
    MaxInput() {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 1;

        Scalar mass = 10.0;
        Scalar gravity = 9.81;
        umax = mass * gravity * 1.1;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Vector<Scalar> c_n(1);
        c_n(0) = umax - u.norm();
        return -c_n;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(1, x.size());
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, u.size());
        J.row(0) = -u.transpose() / u.norm();
        return -J;
    }
};

int main() {
    using Scalar = double;

    int N = 170;
    OptimalControlProblem<Scalar> problem(N);


    problem.setStageDynamics(std::make_shared<Rocket2D<Scalar>>());


    // Matrix<Scalar> Q = Matrix<Scalar>::Zero(6,6);
    // Matrix<Scalar> R = 1e-4 * Matrix<Scalar>::Identity(2,2);
    // problem.setStageCost(std::make_shared<QuadraticStageCost<Scalar>>(Q, R));
    // Matrix<Scalar> QT = Matrix<Scalar>::Zero(6,6);
    // problem.setTerminalCost(std::make_shared<QuadraticTerminalCost<Scalar>>(QT));

    Scalar Q = 0.0;
    Scalar R = 1e-4;
    problem.setStageCost(std::make_shared<ScalarQuadraticStageCost<Scalar>>(Q, R));
    Scalar QT = 0.0;
    problem.setTerminalCost(std::make_shared<ScalarQuadraticTerminalCost<Scalar>>(QT));


    Matrix<Scalar> inputcone_cx = Matrix<Scalar>::Zero(2, 6);
    Matrix<Scalar> inputcone_cu = Matrix<Scalar>::Zero(2, 2);
    inputcone_cu(0, 0) = - std::tan(20.0 * M_PI / 180.0);
    inputcone_cu(1, 1) = - 1.0;
    auto inputcone_c0 = Vector<Scalar>::Zero(2);
    problem.addStageConstraint(std::make_shared<LinearStageConstraint<Scalar>>(
        inputcone_cx, inputcone_cu, inputcone_c0, ConstraintType::SOC));

    
    Matrix<Scalar> glideslope_cx = Matrix<Scalar>::Zero(2, 6);
    glideslope_cx(0, 0) = - std::tan(45.0 * M_PI / 180.0);
    glideslope_cx(1, 1) = - 1.0;
    Matrix<Scalar> glideslope_cu = Matrix<Scalar>::Zero(2, 2);
    auto glideslope_c0 = Vector<Scalar>::Zero(2);
    problem.addStageConstraint(std::make_shared<LinearStageConstraint<Scalar>>(
        glideslope_cx, glideslope_cu, glideslope_c0, ConstraintType::SOC));


    Matrix<Scalar> CT = Matrix<Scalar>::Identity(6,6);
    Vector<Scalar> c0T = Vector<Scalar>::Zero(6);
    c0T(0) = - 1.0;
    problem.addTerminalConstraint(std::make_shared<LinearTerminalConstraint<Scalar>>(
        CT, c0T, ConstraintType::EQ));

    
    problem.addStageConstraint(std::make_shared<MaxInput<Scalar>>());


    Vector<Scalar> x0 = Vector<Scalar>::Zero(6);
    x0(0) = 4.0;
    x0(1) = 3.0;
    problem.setInitialState(0, x0);
    Vector<Scalar> u0 = Vector<Scalar>::Zero(2);
    u0(0) = 9.81 * 10.0;
    problem.setInitialControl(u0);


    Param param;

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

    // std::cout << "X_result = " << std::endl;
    // for (int k = 0; k < problem.N+1; ++k) {
    //     std::cout << X_result[k].transpose() << std::endl;
    // }

    // std::cout << "U_result = " << std::endl;
    // for (int k = 0; k < problem.N; ++k) {
    //     std::cout << U_result[k].transpose() << std::endl;
    // }

    std::cout<<"X_last = \n"<<X_result[problem.getHorizon()].transpose()<<std::endl;
    std::cout<<"U_last = \n"<<U_result[problem.getHorizon() - 1].transpose()<<std::endl;

    return 0;
}