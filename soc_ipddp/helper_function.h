#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <functional>
#include <vector>
#include <cmath>

#include <iostream>

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
    Eigen::MatrixXd hessians(dim_f * dim_x, dim_x);
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
                std::cout<<i * dim_x + j<<" "<<k<<std::endl;
                hessians(i * dim_x + j, k) = second_derivative(i);
                if (j != k) {
                    std::cout<<i * dim_x + k<<" "<<j<<std::endl;
                    hessians(i * dim_x + k, j) = second_derivative(i);
                }
            }
        }
    }
    return hessians;
}

template<typename Func>
Eigen::MatrixXd calculateHessianUU(Func f, Eigen::VectorXd x, Eigen::VectorXd u, double eps = 1e-5) {    
    int dim_u = u.size();
    int dim_f = f(x, u).size();
    Eigen::MatrixXd hessians(dim_f * dim_u, dim_u);
    for (int j = 0; j < dim_u; ++j) {
        for (int k = j; k < dim_u; ++k) {
            Eigen::VectorXd u_pp = u;
            Eigen::VectorXd u_pm = u;
            Eigen::VectorXd u_mp = u;
            Eigen::VectorXd u_mm = u;
            u_pp(j) += eps; u_pp(k) += eps;
            u_pm(j) += eps; u_pm(k) -= eps;
            u_mp(j) -= eps; u_mp(k) += eps;
            u_mm(j) -= eps; u_mm(k) -= eps;

            Eigen::VectorXd second_derivative = (f(x, u_pp) - f(x, u_pm) - f(x, u_mp) + f(x, u_mm)) / (4 * eps * eps);
            
            for (int i = 0; i < dim_f; ++i) {
                hessians(i * dim_u + j, k) = second_derivative(i);
                if (j != k) {
                    hessians(i * dim_u + k, j) = second_derivative(i);
                }
            }
        }
    }
    return hessians;
}

template<typename Func>
Eigen::MatrixXd calculateHessianXU(Func f, Eigen::VectorXd x, Eigen::VectorXd u, double eps = 1e-5) {    
    int dim_x = x.size();
    int dim_u = u.size();
    int dim_f = f(x, u).size();
    Eigen::MatrixXd hessians(dim_f * dim_u, dim_f * dim_x);
    for (int j = 0; j < dim_x; ++j) {
        for (int k = 0; k < dim_u; ++k) {
            Eigen::VectorXd x_p = x;
            Eigen::VectorXd x_m = x;
            Eigen::VectorXd u_p = u;
            Eigen::VectorXd u_m = u;
            x_p(j) += eps;
            x_m(j) -= eps;
            u_p(k) += eps;
            u_m(k) -= eps;

            Eigen::VectorXd second_derivative = (f(x_p, u_p) - f(x_p, u_m) - f(x_m, u_p) + f(x_m, u_m)) / (4 * eps * eps);
            
            for (int i = 0; i < dim_f; ++i) {
                hessians(i * dim_f + j, i * dim_f + k) = second_derivative(i);
                if (j != k) {
                    hessians(i * dim_f + k, i * dim_f + j) = second_derivative(i);
                }
            }
        }
    }
    return hessians;
}

template<typename Func>
Eigen::MatrixXd calculateHessian(Func f, Eigen::VectorXd x, double eps = 1e-5) {    
    int dim_x = x.size();
    int dim_f = f(x).size();
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

            Eigen::VectorXd second_derivative = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps * eps);
            
            for (int i = 0; i < dim_f; ++i) {
                hessians(i * dim_f + j, i * dim_f + k) = second_derivative(i);
                if (j != k) {
                    hessians(i * dim_f + k, i * dim_f + j) = second_derivative(i);
                }
            }
        }
    }
    return hessians;
}
