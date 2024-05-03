#include "soc_ipddp.h"

#include <eigen3/Eigen/Dense>
#include <iostream>

#include <cmath>

// Discrete Time System
Eigen::MatrixXd f(Eigen::VectorXd x0, Eigen::VectorXd u) {
    const double h = 0.05;
    Eigen::MatrixXd x1(x0.rows(),x0.cols());
    x1(0) = x0(0) + h * x0(1);
    x1(1) = x0(1) + h * std::sin(x0(0)) + h * u(0);
    return x1;
}

// Stage Cost Function
double q(Eigen::VectorXd x, Eigen::VectorXd u) {
    return 0.025 * (x.squaredNorm() + u.squaredNorm());
}

// Terminal Cost Function
double p(Eigen::VectorXd x, Eigen::VectorXd u) {
	return 5 * x.squaredNorm();
}

int main() {
    Eigen::VectorXd x0(2);
    x0(0,0) = -M_PI;
    x0(1,0) = 0;

    Eigen::VectorXd u0(1);
    u0(0,0) = 0.0;

    SOC_IPDDP soc_ipddp;
    soc_ipddp.init(500);
    soc_ipddp.setSystemF(f);
    soc_ipddp.setStageCostQ(q);
    soc_ipddp.setTerminalCostP(p);
    soc_ipddp.solve();

    Eigen::MatrixXd X = soc_ipddp.getX();
    Eigen::MatrixXd U = soc_ipddp.getU();

    return 0;
}