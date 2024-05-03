#include <eigen3/Eigen/Dense>
#include <iostream>

#include <cmath>

template<typename Func>
Eigen::MatrixXd calculateJacobian(Func f, Eigen::VectorXd x, Eigen::VectorXd u, double eps = 1e-5) {
    Eigen::MatrixXd fx = f(x, u);
    Eigen::MatrixXd jacobian(fx.size(), x.size() + u.size());
    for (int i = 0; i < x.size(); ++i) {
        Eigen::VectorXd xl = x;
        Eigen::VectorXd xr = x;
        xl(i) -= eps;
        xr(i) += eps;
        jacobian.col(i) = (f(xr, u) - f(xl, u)) / (2 * eps);
    }
    for (int j = x.size(); j < x.size() + u.size(); ++j) {
        Eigen::VectorXd ul = u;
        Eigen::VectorXd ur = u;
        ul(j - x.size()) -= eps;
        ur(j - x.size()) += eps;
        jacobian.col(j) = (f(x, ur) - f(x, ul)) / (2 * eps);
    }
    return jacobian;
}

template<typename Func>
Eigen::MatrixXd calculateFullHessian(Func f, Eigen::VectorXd x, Eigen::VectorXd u, double eps = 1e-5) {
    int n = x.size();
    int m = u.size();
    int dim = n + m;
    Eigen::MatrixXd hessian(dim, dim);

    Eigen::VectorXd z = Eigen::VectorXd::Zero(dim);
    z.head(n) = x;
    z.tail(m) = u;

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            Eigen::VectorXd zij = z;
            Eigen::VectorXd zpijp = z;
            Eigen::VectorXd zpijm = z;
            Eigen::VectorXd zimjp = z;
            Eigen::VectorXd zimjm = z;

            zpijp(i) += eps; zpijp(j) += eps;
            zpijm(i) += eps; zpijm(j) -= eps;
            zimjp(i) -= eps; zimjp(j) += eps;
            zimjm(i) -= eps; zimjm(j) -= eps;

            double fpijp = f(zpijp.head(n), zpijp.tail(m));
            double fpijm = f(zpijm.head(n), zpijm.tail(m));
            double fimjp = f(zimjp.head(n), zimjp.tail(m));
            double fimjm = f(zimjm.head(n), zimjm.tail(m));

            hessian(i, j) = (fpijp - fpijm - fimjp + fimjm) / (4 * eps * eps);
        }
    }

    return hessian;
}