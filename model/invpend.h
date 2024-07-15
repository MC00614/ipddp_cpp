#include "model_base.h"

#include <eigen3/Eigen/Dense>

class InvPend : public ModelBase {
public:
    InvPend();
    ~InvPend();
};

InvPend::InvPend() {
    // Stage Count
    N = 500;

    // Dimensions
    dim_x = 2;
    dim_u = 1;
    dim_c = 2;

    // Status Setting
    X = Eigen::MatrixXd::Zero(dim_x, N+1);
    X(0,0) = -M_PI;
    X(1,0) = 0.0;

    // U = 0.02*Eigen::MatrixXd::Random(dim_u, N) - Eigen::MatrixXd::Constant(dim_u, N, 0.01);
    U = Eigen::MatrixXd::Zero(dim_u, N);
    U(0,0) = 0.0;

    Y = 0.01*Eigen::MatrixXd::Ones(dim_c, N);

    S = 0.1*Eigen::MatrixXd::Ones(dim_c, N);

    // Discrete Time System
    f = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> Eigen::VectorXd {
        const double h = 0.05;
        Eigen::VectorXd x_n(x.size());
        x_n(0) = x(0) + h * x(1);
        x_n(1) = x(1) + h * std::sin(x(0)) + h * u(0);
        return x_n;
    };

    // Stage Cost Function
    q = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> double {
        return 0.025 * (x.squaredNorm() + u.squaredNorm());
    };

    // Terminal Cost Function
    p = [this](const Eigen::VectorXd& x) -> double {
        return 5.0 * x.squaredNorm();
    };

    // Constraint
    c = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> Eigen::VectorXd {
        Eigen::VectorXd c_n(x.size());
        c_n(0) = u(0) - 0.25;
        c_n(1) = -u(0) - 0.25;
        return c_n;
    };
}

InvPend::~InvPend() {
}