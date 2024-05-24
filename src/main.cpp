#include "soc_ipddp.h"

#include <eigen3/Eigen/Dense>
#include <iostream>

#include <cmath>

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

int main() {
    int N = 500;

    int dim_x = 2;
    int dim_u = 1;

    Eigen::MatrixXd X(dim_x, N);
    X(0,0) = -M_PI;
    X(1,0) = 0.0;
    X(0,N-1) = 0,0;
    X(1,N-1) = 0.0;
    Eigen::MatrixXd U(dim_u, N);
    U(0,0) = 0.0;
    
    SOC_IPDDP soc_ipddp;
    soc_ipddp.init(N, 1000, X, U);
    soc_ipddp.setSystemModel(f);
    soc_ipddp.setStageCost(q);
    soc_ipddp.setTerminalCost(p);

    soc_ipddp.solve();

    // Eigen::MatrixXd X = soc_ipddp.getX();
    // Eigen::MatrixXd U = soc_ipddp.getU();

    return 0;
}