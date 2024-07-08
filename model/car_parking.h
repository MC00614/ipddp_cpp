#include "model_base.h"

#include <eigen3/Eigen/Dense>

class CarParking : public ModelBase {
public:
    CarParking();
    ~CarParking();
};

CarParking::CarParking() {
    // Stage Count
    N = 500;

    // Dimensions
    dim_x = 4;
    dim_u = 2;

    // Status Setting
    X = Eigen::MatrixXd::Zero(dim_x, N+1);
    X(0,0) = 1.0;
    X(1,0) = 1.0;
    X(2,0) = 3*M_PI_2;
    X(3,0) = 0.0;
    X(0,N) = 0,0;
    X(1,N) = 0.0;
    X(2,N) = 0.0;
    X(3,N) = 0.0;
    U = Eigen::MatrixXd::Zero(dim_u, N);
    U(0,0) = 0.0;
    U(1,0) = 0.0;

    // Discrete Time System
    f = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> Eigen::VectorXd {
        const double h = 0.03;
        const double d = 2.0;
        Eigen::VectorXd x_n(x.size());
        double b = d + h*x(3)*std::cos(u(0)) - std::sqrt(d*d - h*h*x(3)*x(3)*std::sin(u(0))*std::sin(u(0)));
        x_n(0) = x(0) + b*std::cos(x(2));
        x_n(1) = x(1) + b*std::sin(x(2));
        x_n(2) = x(2) + std::asin((h*x(3)/d)*std::sin(u(0)));
        x_n(3) = x(3) + h*u(1);
        return x_n;
    };

    // Stage Cost Function
    q = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> double {
        return 0.01 * (u(0)*u(0) + 0.01*u(1)*u(1)) +
            0.01 * (std::sqrt(x(0) * x(0) + 0.1 * 0.1) - 0.1) +
            0.01 * (std::sqrt(x(1) * x(1) + 0.1 * 0.1) - 0.1);
    };

    // Terminal Cost Function
    p = [this](const Eigen::VectorXd& x) -> double {
        return std::sqrt(x(0) * x(0) + 0.1 * 0.1) - 0.1 +
           std::sqrt(x(1) * x(1) + 0.1 * 0.1) - 0.1 +
           std::sqrt(x(2) * x(2) + 0.01 * 0.01) - 0.01 +
           std::sqrt(x(3) * x(3) + 1.0 * 1.0) - 1.0;
    };
}

CarParking::~CarParking() {
}