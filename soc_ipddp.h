#include <eigen3/Eigen/Dense>
#include <iostream>

#include <cmath>

template<typename Func>
Eigen::MatrixXd calculateJacobian(Func f, Eigen::VectorXd x, Eigen::VectorXd u, double eps = 1e-5) {
    Eigen::MatrixXd fx = f(x, u);
    Eigen::MatrixXd jacobian(fx.size(), x.size());
    for (int i = 0; i < x.size(); ++i) {
        Eigen::VectorXd xl = x;
        Eigen::VectorXd xr = x;
        xl(i) -= eps;
        xr(i) += eps;
        Eigen::MatrixXd fxl = f(xl, u);
        Eigen::MatrixXd fxr = f(xr, u);
        jacobian.col(i) = (fxr - fxl) / (2 * eps); 
    }
    return jacobian;

    // Eigen::MatrixXd grad(x.rows(), x.rows());
    // for (int i = 0; i < x.size(); ++i) {
    //     Value x_eps = x;
    //     x_eps(i) += eps;
    //     grad(i) = (f(x_eps, u) - f(x, u)) / eps;
    // }
    // return grad;
}