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

    // Status Setting
    X = Eigen::MatrixXd::Zero(dim_x, N+1);
    X(0,0) = -M_PI;
    X(1,0) = 0.0;
    X(0,N) = 0,0;
    X(1,N) = 0.0;
    U = Eigen::MatrixXd::Zero(dim_u, N);
    U(0,0) = 0.0;

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
}

InvPend::~InvPend() {
}