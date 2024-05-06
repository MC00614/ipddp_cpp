#include <eigen3/Eigen/Dense>
#include <functional>
#include <vector>
#include <cmath>

template<typename Func>
Eigen::MatrixXd calculateJacobian(Func f, Eigen::VectorXd x, Eigen::VectorXd u, double eps = 1e-5) {
    Eigen::MatrixXd F = f(x, u);
    Eigen::MatrixXd jacobian(F.rows(), x.size() + u.size());
    for (int i = 0; i < x.size(); ++i) {
        Eigen::VectorXd xm = x;
        Eigen::VectorXd xp = x;
        xm(i) -= eps;
        xp(i) += eps;
        jacobian.col(i) = (f(xp, u) - f(xm, u)) / (2 * eps);
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
Eigen::MatrixXd calculateJacobian(Func f, Eigen::VectorXd x, double eps = 1e-5) {
    Eigen::MatrixXd F = f(x);
    Eigen::MatrixXd jacobian(F.rows(), x.size());
    for (int i = 0; i < x.size(); ++i) {
        Eigen::VectorXd xm = x;
        Eigen::VectorXd xp = x;
        xm(i) -= eps;
        xp(i) += eps;
        jacobian.col(i) = (f(xp) - f(xm)) / (2 * eps);
    }
    return jacobian;
}

template<typename Func>
std::vector<Eigen::MatrixXd> calculateHessian(Func f, Eigen::VectorXd x, double eps = 1e-5) {    
    int n = x.size();
    int m = f(x).size();
    std::vector<Eigen::MatrixXd> hessians(m, Eigen::MatrixXd(n, n));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = j; k < n; ++k) {
                Eigen::VectorXd x_jkp = x, x_jkm = x, x_jmk = x, x_jmm = x;
                x_jkp(j) += eps; x_jkp(k) += eps;
                x_jkm(j) += eps; x_jkm(k) -= eps;
                x_jmk(j) -= eps; x_jmk(k) += eps;
                x_jmm(j) -= eps; x_jmm(k) -= eps;

                Eigen::VectorXd f_jkp = f(x_jkp);
                Eigen::VectorXd f_jkm = f(x_jkm);
                Eigen::VectorXd f_jmk = f(x_jmk);
                Eigen::VectorXd f_jmm = f(x_jmm);

                double second_derivative = (f_jkp(i) - f_jkm(i) - f_jmk(i) + f_jmm(i)) / (4 * eps * eps);
                hessians[i](j, k) = second_derivative;
                if (j != k) {
                    hessians[i](k, j) = second_derivative;
                }
            }
        }
    }
    return hessians;
}

template<typename Func>
std::vector<Eigen::MatrixXd> calculateHessian(Func f, Eigen::VectorXd x, Eigen::VectorXd u, double eps = 1e-5) {    
    int nx = x.size();
    int nu = u.size();
    int m = f(x, u).size();
    std::vector<Eigen::MatrixXd> hessians(m, Eigen::MatrixXd(nx + nu, nx + nu));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < nx + nu; ++j) {
            for (int k = j; k < nx + nu; ++k) {
                Eigen::VectorXd x_jp = x, x_jm = x;
                Eigen::VectorXd u_kp = u, u_km = u;

                if (j < nx) {
                    x_jp(j) += eps; x_jm(j) -= eps;
                } else {
                    u_kp(j-nx) += eps; u_km(j-nx) -= eps;
                }

                if (k < nx) {
                    x_jp(k) += eps; x_jm(k) -= eps;
                } else {
                    u_kp(k-nx) += eps; u_km(k-nx) -= eps;
                }

                Eigen::VectorXd f_jkp = f(x_jp, u_kp);
                Eigen::VectorXd f_jkm = f(x_jp, u_km);
                Eigen::VectorXd f_jmk = f(x_jm, u_kp);
                Eigen::VectorXd f_jmm = f(x_jm, u_km);

                double second_derivative = (f_jkp(i) - f_jkm(i) - f_jmk(i) + f_jmm(i)) / (4 * eps * eps);
                hessians[i](j, k) = second_derivative;
                if (j != k) {
                    hessians[i](k, j) = second_derivative;
                }
            }
        }
    }

    return hessians;
}
