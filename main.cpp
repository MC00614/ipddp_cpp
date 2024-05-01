#include <eigen3/Eigen/Dense>
#include <iostream>

// constexpr int dim_x = 3;
// constexpr int dim_u = 1;
// const Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(dim_x, dim_x);
// const Eigen::MatrixXd R = Eigen::MatrixXd::Identity(dim_u, dim_u);

// Discrete Time System
template <typename State, typename Input, typename Output>
Output f(State x, Input u) {
    auto f;
    f(0,0) = 0.025 * ((x.transpose() * x) + (u.transpose() * u))
    f(1,0) = 
    return f;
}

// Stage Cost Function
template <typename State, typename Input, typename Output>
Output q(State x, Input u) {
    auto q = 0.025 * ((x.transpose() * x) + (u.transpose() * u))
    return q(0,0);
}

// Terminal Cost Function
template <typename State, typename Input, typename Output>
Output p(State x, Input u) {
	auto p = (5 * (x.transpose() * x));
    return p(0,0)
}

int main() {
    Eigen::MatrixXd x(2,1);
    x(0,0) = -M_PI;
    x(1,0) = 0;

    Eigen::MatrixXd u(1,1);
    u(0,0) = 0.0;
    auto sum1 = q<Eigen::MatrixXd, Eigen::MatrixXd, double> (x, u);
    auto sum2 = p<Eigen::MatrixXd, Eigen::MatrixXd, double> (x, u);

    

    std::cout << "123" << std::endl;
    std::cout << sum1 << std::endl;
    std::cout << sum2 << std::endl;
    return 0;
}