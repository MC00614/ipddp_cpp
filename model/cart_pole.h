#include "model_base.h"

#include <eigen3/Eigen/Dense>

class CartPole : public ModelBase {
public:
    CartPole();
    ~CartPole();
};

CartPole::CartPole() {
    // Stage Count
    N = 500;

    // Dimensions
    dim_x = 5;
    dim_u = 1;

    // Status Setting
    X = Eigen::MatrixXd::Zero(dim_x, N+1);
    X(0,0) = 0.0;
    X(1,0) = 0.0;
    X(2,0) = std::sin(M_PI);
    X(3,0) = std::cos(M_PI);
    X(4,0) = 0.0;
    X(0,N) = 0,0;
    X(1,N) = 0.0;
    X(2,N) = std::sin(0.0);
    X(3,N) = std::cos(0.0);
    X(4,N) = 0.0;
    U = Eigen::MatrixXd::Zero(dim_u, N);
    U(0,0) = 0.0;

    // Discrete Time System
    f = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> Eigen::VectorXd {
        const double G = 9.80665;  // gravitational constant
        const double mc = 1.0;  // mass of the cart
        const double mp = 0.1;  // mass of the pole
        const double l = 1.0;   // length of the pole
        const double dt = 0.05; // time step

        double x_ = x(0);
        double x_dot = x(1);
        double sin_theta = x(2);
        double cos_theta = x(3);
        double theta_dot = x(4);

        double F = std::tanh(u(0));

        // Eq. (23)
        double temp = (F + mp * l * theta_dot * theta_dot * sin_theta) / (mc + mp);
        double numerator = G * sin_theta - cos_theta * temp;
        double denominator = l * (4.0 / 3.0 - mp * cos_theta * cos_theta / (mc + mp));
        double theta_dot_dot = numerator / denominator;

        // Eq. (24)
        double x_dot_dot = temp - mp * l * theta_dot_dot * cos_theta / (mc + mp);

        // Deaugment state for dynamics.
        double theta = std::atan2(sin_theta, cos_theta);
        double next_theta = theta + theta_dot * dt;

        Eigen::VectorXd x_n(5);
        x_n(0) = x_ + x_dot * dt;
        x_n(1) = x_dot + x_dot_dot * dt;
        x_n(2) = std::sin(next_theta);
        x_n(3) = std::cos(next_theta);
        x_n(4) = theta_dot + theta_dot_dot * dt;
        return x_n;
    };

    // Stage Cost Function
    q = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> double {
        return x(0) * x(0) +
                x(2) * x(2) +
                (x(3)-1.0) * (x(3)-1.0) +
                0.1 * u(0) * u(0);
    };

    // Terminal Cost Function
    p = [this](const Eigen::VectorXd& x) -> double {
        return 100.0 * (x(0) * x(0) +
                        x(1) * x(1) +
                        x(2) * x(2) +
                        (x(3)-1.0) * (x(3)-1.0) +
                        x(4) * x(4));
    };
}

CartPole::~CartPole() {
}