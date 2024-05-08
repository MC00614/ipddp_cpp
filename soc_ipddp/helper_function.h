#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <functional>
#include <vector>
#include <cmath>

template<typename Func>
Eigen::MatrixXd calculateJacobianX(Func f, Eigen::VectorXd x, Eigen::VectorXd u, double eps = 1e-5) {
    int dim_x = x.size();
    int dim_f = f(x, u).size();
    Eigen::MatrixXd jacobian(dim_f, dim_x);
    for (int i = 0; i < dim_x; ++i) {
        Eigen::VectorXd x_m = x;
        Eigen::VectorXd x_p = x;
        x_m(i) -= eps;
        x_p(i) += eps;
        jacobian.col(i) = (f(x_p, u) - f(x_m, u)) / (2 * eps);
    }
    return jacobian;
}

template<typename Func>
Eigen::MatrixXd calculateJacobianU(Func f, Eigen::VectorXd x, Eigen::VectorXd u, double eps = 1e-5) {
    int dim_u = u.size();
    int dim_f = f(x, u).size();
    Eigen::MatrixXd jacobian(dim_f, dim_u);
    for (int i = 0; i < dim_u; ++i) {
        Eigen::VectorXd u_m = u;
        Eigen::VectorXd u_p = u;
        u_m(i) -= eps;
        u_p(i) += eps;
        jacobian.col(i) = (f(x, u_p) - f(x, u_m)) / (2 * eps);
    }
    return jacobian;
}

template<typename Func>
Eigen::MatrixXd calculateJacobian(Func f, Eigen::VectorXd x, Eigen::VectorXd u, double eps = 1e-5) {
    int dim_x = x.size();
    int dim_u = u.size();
    int dim_f = f(x, u).size();
    Eigen::MatrixXd jacobian(dim_f, dim_x + dim_u);
    for (int i = 0; i < dim_x; ++i) {
        Eigen::VectorXd x_m = x;
        Eigen::VectorXd x_p = x;
        x_m(i) -= eps;
        x_p(i) += eps;
        jacobian.col(i) = (f(x_p, u) - f(x_m, u)) / (2 * eps);
    }
    for (int j = dim_x; j < dim_x + dim_u; ++j) {
        Eigen::VectorXd u_m = u;
        Eigen::VectorXd u_p = u;
        u_m(j - dim_x) -= eps;
        u_p(j - dim_x) += eps;
        jacobian.col(j) = (f(x, u_p) - f(x, u_m)) / (2 * eps);
    }
    return jacobian;
}

template<typename Func>
Eigen::MatrixXd calculateJacobian(Func f, Eigen::VectorXd x, double eps = 1e-5) {
    int dim_x = x.size();
    int dim_f = f(x).size();
    Eigen::MatrixXd jacobian(dim_f, dim_x);
    for (int i = 0; i < dim_x; ++i) {
        Eigen::VectorXd x_m = x;
        Eigen::VectorXd x_p = x;
        x_m(i) -= eps;
        x_p(i) += eps;
        jacobian.col(i) = (f(x_p) - f(x_m)) / (2 * eps);
    }
    return jacobian;
}

template<typename Func>
Eigen::MatrixXd calculateHessianXX(Func f, Eigen::VectorXd x, Eigen::VectorXd u, double eps = 1e-5) {    
    int dim_x = x.size();
    int dim_f = f(x, u).size();
    Eigen::MatrixXd hessians(dim_f * dim_x, dim_f * dim_x);
    for (int j = 0; j < dim_x; ++j) {
        for (int k = j; k < dim_x; ++k) {
            Eigen::VectorXd x_pp = x;
            Eigen::VectorXd x_pm = x;
            Eigen::VectorXd x_mp = x;
            Eigen::VectorXd x_mm = x;
            x_pp(j) += eps; x_pp(k) += eps;
            x_pm(j) += eps; x_pm(k) -= eps;
            x_mp(j) -= eps; x_mp(k) += eps;
            x_mm(j) -= eps; x_mm(k) -= eps;

            Eigen::VectorXd second_derivative = (f(x_pp, u) - f(x_pm, u) - f(x_mp, u) + f(x_mm, u)) / (4 * eps * eps);
            
            for (int i = 0; i < dim_f; ++i) {
                hessians(i * dim_x + j, i * dim_x + k) = second_derivative(i);
                if (j != k) {
                    hessians(i * dim_x + k, i * dim_x + j) = second_derivative(i);
                }
            }
        }
    }
    return hessians;
}

template<typename Func>
Eigen::Tensor<double, 3> calculateHessian(Func f, Eigen::VectorXd x, double eps = 1e-5) {    
    int n = x.size();
    int m = f(x).size();
    Eigen::Tensor<double, 3> hessians(m, n, n);

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
                hessians(i, j, k) = second_derivative;
                if (j != k) {
                    hessians(i, j, k) = second_derivative;
                }
            }
        }
    }
    return hessians;
}

template<typename Func>
Eigen::Tensor<double, 3> calculateHessian(Func f, Eigen::VectorXd x, Eigen::VectorXd u, double eps = 1e-5) {    
    int nx = x.size();
    int nu = u.size();
    int m = f(x, u).size();
    Eigen::Tensor<double, 3> hessians(m, nx + nu, nx + nu);

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
                hessians(i, j, k) = second_derivative;
                if (j != k) {
                    hessians(i, k, j) = second_derivative;
                }
            }
        }
    }

    return hessians;
}
