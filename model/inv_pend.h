#include <eigen3/Eigen/Dense>

// Discrete Time System
Eigen::VectorXd f(Eigen::VectorXd x0, Eigen::VectorXd u) {
    const double h = 0.05;
    Eigen::VectorXd x1(x0.rows(),x0.cols());
    x1(0) = x0(0) + h * x0(1);
    x1(1) = x0(1) + h * std::sin(x0(0)) + h * u(0);
    return x1;
}

// Stage Cost Function
double q(Eigen::VectorXd x, Eigen::VectorXd u) {
    double q;
    q =  0.025 * (x.squaredNorm() + u.squaredNorm());
    return q;
}

// Terminal Cost Function
double p(Eigen::VectorXd x) {
    double p;
    p = 5 * x.squaredNorm();
	return p;
}
