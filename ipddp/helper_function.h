#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

#include <functional>
#include <vector>
#include <cmath>

#include <iostream>


// tensdot = @(a,b) permute(sum(bsxfun(@times,a,b),1), [2 3 1]);
inline Eigen::MatrixXd tensdot(const Eigen::VectorXd& vector, const Eigen::Tensor<double, 3>& tensor) {
    int dim1 = tensor.dimension(0);
    int dim2 = tensor.dimension(1);
    int dim3 = tensor.dimension(2);

    Eigen::MatrixXd result(dim2, dim3);
    result.setZero();

    for (int i = 0; i < dim1; ++i) {
        Eigen::MatrixXd tensor_slice(dim2, dim3);
        for (int j = 0; j < dim2; ++j) {
            for (int k = 0; k < dim3; ++k) {
                tensor_slice(j, k) = tensor(i, j, k);
            }
        }
        result += vector(i) * tensor_slice;
    }

    return result;
}

template<typename Func1, typename Func2>
inline void vectorHessian(Eigen::Tensor<double, 3> &hessians, Func1 f, Func2 fs, VectorXdual2nd x, VectorXdual2nd u, const std::string& variable, double eps = 1e-5) {
    int rows = x.size();
    int cols1;
    int cols2;
    Eigen::MatrixXd second_derivative;
    
    if (variable == "xx") {
        cols1 = x.size();
        cols2 = x.size();
        for (int i = 0; i < rows; ++i) {
            second_derivative = hessian(fs[i], wrt(x), at(x,u));
            for (int j = 0; j < cols1; ++j) {
                for (int k = j; k < cols2; ++k) {
                    hessians(i, j, k) = second_derivative(j,k);
                    if (j != k) {
                        hessians(i, k, j) = second_derivative(j,k);
                    }
                }
            }
        }
    }
    else if (variable == "uu") {
        cols1 = u.size();
        cols2 = u.size();
        for (int i = 0; i < rows; ++i) {
            second_derivative = hessian(fs[i], wrt(x), at(x,u));
            for (int j = 0; j < cols1; ++j) {
                for (int k = j; k < cols2; ++k) {
                    hessians(i, j, k) = second_derivative(j,k);
                    if (j != k) {
                        hessians(i, k, j) = second_derivative(j,k);
                    }
                }
            }
        }
    }
    else if (variable == "xu") {
        cols1 = x.size();
        cols2 = u.size();
        for (int k = 0; k < cols2; ++k) {
            VectorXdual2nd u_p = u;
            VectorXdual2nd u_m = u;
            u_p(k) += eps;
            u_m(k) -= eps;

            second_derivative = (jacobian(f, wrt(x), at(x,u_p)) - jacobian(f, wrt(x), at(x,u_m))) / (2 * eps);

            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols1; ++j) {
                    hessians(i, j, k) = second_derivative(i,j);
                }
            }
        }
    }
    else {throw std::invalid_argument("Invalid variable. Use 'xx', 'uu' or 'xu'.");}

    return;
}

template<typename Func>
inline void scalarHessian(Eigen::MatrixXd &hessians, Func f, VectorXdual2nd x, VectorXdual2nd u, const std::string& variable, double eps = 1e-5) {
    int rows;
    int cols;

    if (variable == "xu") {
        rows = x.size();
        cols = u.size();
        VectorXdual2nd u_p(cols);
        VectorXdual2nd u_m(cols);
        for (int k = 0; k < cols; ++k) {
            u_p = u;
            u_m = u;
            u_p(k) += eps;
            u_m(k) -= eps;
            hessians.col(k) = (gradient(f, wrt(x), at(x,u_p)) - gradient(f, wrt(x), at(x,u_m))) / (2 * eps);
        }
    }
    else {throw std::invalid_argument("Invalid variable. Use 'xu'. \n If you want to calculate hessian of 'xx' or 'uu', use autodiff function.");}

    return;
}
