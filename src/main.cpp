#include "soc_ipddp.h"

#include <eigen3/Eigen/Dense>
#include <iostream>

#include <cmath>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

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
    // Intitial Setting
    int N = 500;

    int dim_x = 2;
    int dim_u = 1;

    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(dim_x, N+1);
    X(0,0) = -M_PI;
    X(1,0) = 0.0;
    X(0,N) = 0,0;
    X(1,N) = 0.0;
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(dim_u, N);
    U(0,0) = 0.0;
    
    // Solver
    SOC_IPDDP soc_ipddp;
    soc_ipddp.setSystemModel(f);
    soc_ipddp.setStageCost(q);
    soc_ipddp.setTerminalCost(p);
    soc_ipddp.init(N, 100, 1e-2, X, U);

    soc_ipddp.solve();

    // Result
    Eigen::MatrixXd X_result = soc_ipddp.getX();
    Eigen::MatrixXd U_result = soc_ipddp.getU();
    std::vector<double> all_cost = soc_ipddp.getAllCost();


    // // // // // // // // // // // // // // // // // // // // // // // // // 
    //  VISUALIZATION  // VISUALIZATION  // VISUALIZATION  // VISUALIZATION // 
    // // // // // // // // // // // // // // // // // // // // // // // // // 
    std::vector<std::vector<double>> X_INIT(X_result.rows(), std::vector<double>(X_result.cols()));
    std::vector<std::vector<double>> X_RES(X_result.rows(), std::vector<double>(X_result.cols()));
    for (int i = 0; i < X_result.rows(); ++i) {
        for (int j = 0; j < X_result.cols(); ++j) {
            X_INIT[i][j] = X(i,j);
            X_RES[i][j] = X_result(i, j);
        }
    }

    std::vector<std::vector<double>> U_INIT(U_result.rows(), std::vector<double>(U_result.cols()));
    std::vector<std::vector<double>> U_RES(U_result.rows(), std::vector<double>(U_result.cols()));
    for (int i = 0; i < U_result.rows(); ++i) {
        for (int j = 0; j < U_result.cols(); ++j) {
            U_INIT[i][j] = U(i,j);
            U_RES[i][j] = U_result(i, j);
        }
    }

    for (size_t i = 0; i < X_RES.size(); ++i) {
        plt::subplot(dim_x + dim_u + 1, 1, i + 1);
        plt::plot(X_INIT[i], {{"label", "Init"}});
        plt::plot(X_RES[i], {{"label", "Result"}});
        plt::title("X_result Dimension " + std::to_string(i));
        plt::legend();
    }

    for (size_t i = 0; i < U_RES.size(); ++i) {
        plt::subplot(dim_x + dim_u + 1, 1, dim_x + 1 + i);
        plt::plot(U_INIT[i], {{"label", "Init"}});
        plt::plot(U_RES[i], {{"label", "Result"}});
        plt::title("U_result Dimension " + std::to_string(i));
        plt::legend();
    }

    plt::subplot(dim_x + dim_u + 1, 1, dim_x + dim_u + 1);
    plt::plot(all_cost, {{"label", "Cost"}});
    plt::title("Cost");
    plt::legend();

    plt::show();
    // // // // // // // // // // // // // // // // // // // // // // // // // 
    //  VISUALIZATION  // VISUALIZATION  // VISUALIZATION  // VISUALIZATION // 
    // // // // // // // // // // // // // // // // // // // // // // // // // 

    return 0;
}