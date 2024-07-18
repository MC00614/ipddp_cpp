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
inline Eigen::Tensor<double, 3> vectorHessian(Func1 f, Func2 fs, VectorXdual2nd x, VectorXdual2nd u, const std::string& variable, double eps = 1e-5) {
    int rows = f(x, u).size();
    int cols1;
    int cols2;
    Eigen::Tensor<double, 3> hessians;
    
    if (variable == "xx") {
        cols1 = x.size();
        cols2 = x.size();
        hessians.resize(rows, cols1, cols2);
        for (int i = 0; i < rows; ++i) {
            Eigen::MatrixXd second_derivative = hessian(fs[i], wrt(x), at(x,u));
            for (int j = 0; j < cols1; ++j) {
                for (int k = 0; k < cols2; ++k) {
                    hessians(i, j, k) = second_derivative(j,k);
                }
            }
        }
    }
    else if (variable == "uu") {
        cols1 = u.size();
        cols2 = u.size();
        hessians.resize(rows, cols1, cols2);
        for (int i = 0; i < rows; ++i) {
            Eigen::MatrixXd second_derivative = hessian(fs[i], wrt(x), at(x,u));
            for (int j = 0; j < cols1; ++j) {
                for (int k = 0; k < cols2; ++k) {
                    hessians(i, j, k) = second_derivative(j,k);
                }
            }
        }
    }
    else if (variable == "xu") {
        cols1 = x.size();
        cols2 = u.size();
        hessians.resize(rows, cols1, cols2);
        for (int k = 0; k < cols2; ++k) {
            VectorXdual2nd u_p = u;
            VectorXdual2nd u_m = u;
            u_p(k) += eps;
            u_m(k) -= eps;

            Eigen::MatrixXd second_derivative = (jacobian(f, wrt(x), at(x,u_p)) - jacobian(f, wrt(x), at(x,u_m))) / (2 * eps);

            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols1; ++j) {
                    hessians(i, j, k) = second_derivative(i,j);
                }
            }
        }
    }
    else {throw std::invalid_argument("Invalid variable. Use 'xx', 'uu' or 'xu'.");}

    return hessians;
}

template<typename Func>
inline Eigen::MatrixXd scalarHessian(Func f, VectorXdual2nd x, VectorXdual2nd u, const std::string& variable, double eps = 1e-5) {
    int rows;
    int cols;
    Eigen::MatrixXd hessians;

    if (variable == "xu") {
        rows = x.size();
        cols = u.size();
        hessians.resize(rows, cols);
        for (int k = 0; k < cols; ++k) {
            VectorXdual2nd u_p = u;
            VectorXdual2nd u_m = u;
            u_p(k) += eps;
            u_m(k) -= eps;

            Eigen::MatrixXd second_derivative = (gradient(f, wrt(x), at(x,u_p)) - gradient(f, wrt(x), at(x,u_m))) / (2 * eps);
            for (int j = 0; j < rows; ++j) {
                hessians(j, k) = second_derivative(j);
            }
        }
    }
    else {throw std::invalid_argument("Invalid variable. Use 'xu'. \n If you want to calculate hessian of 'xx' or 'uu', use hessian function.");}

    return hessians;
}

/////////////////////////////////////////////////////////

// template<typename Func>
// inline Eigen::MatrixXd vectorJacobian(Func f, const Eigen::VectorXd& x, const Eigen::VectorXd& u, const std::string& variable, double eps = 1e-5) {
//     int rows = f(x, u).size();
//     int cols;
//     Eigen::VectorXd m;
//     Eigen::VectorXd p;
//     Eigen::MatrixXd jacobian;

//     if (variable == "x") {
//         cols = x.size();
//         jacobian.resize(rows, cols);
//         for (int i = 0; i < cols; ++i) {
//             m = x;
//             p = x;
//             m(i) -= eps;
//             p(i) += eps;
//             jacobian.col(i) = (f(p, u) - f(m, u)) / (2 * eps);
//         }
//     } 
//     else if (variable == "u") {
//         cols = u.size();
//         jacobian.resize(rows, cols);
//         for (int i = 0; i < cols; ++i) {
//             m = u;
//             p = u;
//             m(i) -= eps;
//             p(i) += eps;
//             jacobian.col(i) = (f(x, p) - f(x, m)) / (2 * eps);
//         }
//     } 
//     else {
//         throw std::invalid_argument("Invalid variable. Use 'x' or 'u'.");
//     }
//     return jacobian;
// }

// template<typename Func>
// inline Eigen::VectorXd scalarJacobian(Func f, const Eigen::VectorXd& x, const Eigen::VectorXd& u, const std::string& variable, double eps = 1e-5) {
//     int rows;
//     int cols = 1;
//     Eigen::VectorXd m;
//     Eigen::VectorXd p;
//     Eigen::VectorXd jacobian;

//     if (variable == "x") {
//         rows = x.size();
//         jacobian.resize(rows, cols);
//         for (int i = 0; i < rows; ++i) {
//             m = x;
//             p = x;
//             m(i) -= eps;
//             p(i) += eps;
//             jacobian(i,0) = (f(p, u) - f(m, u)) / (2 * eps);
//         }
//     } 
//     else if (variable == "u") {
//         rows = u.size();
//         jacobian.resize(rows, cols);
//         for (int i = 0; i < rows; ++i) {
//             m = u;
//             p = u;
//             m(i) -= eps;
//             p(i) += eps;
//             jacobian(i,0) = (f(x, p) - f(x, m)) / (2 * eps);
//         }
//     }
//     else {throw std::invalid_argument("Invalid variable. Use 'x' or 'u'.");}

//     return jacobian;
// }

// template<typename Func>
// inline Eigen::VectorXd scalarJacobian(Func f, const Eigen::VectorXd& x, double eps = 1e-5) {
//     int rows;
//     int cols = 1;
//     Eigen::VectorXd m;
//     Eigen::VectorXd p;
//     Eigen::VectorXd jacobian;
//     rows = x.size();
//     jacobian.resize(rows, cols);
//     for (int i = 0; i < rows; ++i) {
//         m = x;
//         p = x;
//         m(i) -= eps;
//         p(i) += eps;
//         jacobian(i,0) = (f(p) - f(m)) / (2 * eps);
//     }

//     return jacobian;
// }

// template<typename Func>
// inline Eigen::Tensor<double, 3> vectorHessian(Func f, const Eigen::VectorXd& x, const Eigen::VectorXd& u, const std::string& variable, double eps = 1e-5) {
//     int rows = f(x, u).size();
//     int cols1;
//     int cols2;
//     Eigen::Tensor<double, 3> hessians;
    
//     if (variable == "xx") {
//         cols1 = x.size();
//         cols2 = x.size();
//         hessians.resize(rows, cols1, cols2);
//         for (int j = 0; j < cols1; ++j) {
//             for (int k = j; k < cols2; ++k) {
//                 Eigen::VectorXd x_pp = x;
//                 Eigen::VectorXd x_pm = x;
//                 Eigen::VectorXd x_mp = x;
//                 Eigen::VectorXd x_mm = x;
//                 x_pp(j) += eps; x_pp(k) += eps;
//                 x_pm(j) += eps; x_pm(k) -= eps;
//                 x_mp(j) -= eps; x_mp(k) += eps;
//                 x_mm(j) -= eps; x_mm(k) -= eps;

//                 Eigen::VectorXd second_derivative = (f(x_pp, u) - f(x_pm, u) - f(x_mp, u) + f(x_mm, u)) / (4 * eps * eps);
//                 for (int i = 0; i < rows; ++i) {
//                     hessians(i, j, k) = second_derivative(i);
//                     if (j != k) {
//                         hessians(i, k, j) = second_derivative(i);
//                     }
//                 }
//             }
//         }
//     }
//     else if (variable == "xu") {
//         cols1 = x.size();
//         cols2 = u.size();
//         hessians.resize(rows, cols1, cols2);
//         for (int j = 0; j < cols1; ++j) {
//             for (int k = 0; k < cols2; ++k) {
//                 Eigen::VectorXd x_p = x;
//                 Eigen::VectorXd x_m = x;
//                 Eigen::VectorXd u_p = u;
//                 Eigen::VectorXd u_m = u;
//                 x_p(j) += eps;
//                 x_m(j) -= eps;
//                 u_p(k) += eps;
//                 u_m(k) -= eps;

//                 Eigen::VectorXd second_derivative = (f(x_p, u_p) - f(x_p, u_m) - f(x_m, u_p) + f(x_m, u_m)) / (4 * eps * eps);

//                 for (int i = 0; i < rows; ++i) {
//                     hessians(i, j, k) = second_derivative(i);
//                     // if (j != k) {
//                     //     hessians(i, k, j) = second_derivative(i);
//                     // }
//                 }
//             }
//         }
//     }
//     else if (variable == "uu") {
//         cols1 = u.size();
//         cols2 = u.size();
//         hessians.resize(rows, cols1, cols2);
//         for (int j = 0; j < cols1; ++j) {
//             for (int k = j; k < cols2; ++k) {
//                 Eigen::VectorXd u_pp = u;
//                 Eigen::VectorXd u_pm = u;
//                 Eigen::VectorXd u_mp = u;
//                 Eigen::VectorXd u_mm = u;
//                 u_pp(j) += eps; u_pp(k) += eps;
//                 u_pm(j) += eps; u_pm(k) -= eps;
//                 u_mp(j) -= eps; u_mp(k) += eps;
//                 u_mm(j) -= eps; u_mm(k) -= eps;

//                 Eigen::VectorXd second_derivative = (f(x, u_pp) - f(x, u_pm) - f(x, u_mp) + f(x, u_mm)) / (4 * eps * eps);

//                 for (int i = 0; i < rows; ++i) {
//                     hessians(i, j, k) = second_derivative(i);
//                     if (j != k) {
//                         hessians(i, k, j) = second_derivative(i);
//                     }
//                 }
//             }
//         }
//     }
//     else {throw std::invalid_argument("Invalid variable. Use 'xx', 'uu' or 'xu'.");}

//     return hessians;
// }

// template<typename Func>
// inline Eigen::MatrixXd scalarHessian(Func f, const Eigen::VectorXd& x, const Eigen::VectorXd& u, const std::string& variable, double eps = 1e-5) {
//     int rows;
//     int cols;
//     Eigen::MatrixXd hessians;
    
//     if (variable == "xx") {
//         rows = x.size();
//         cols = x.size();
//         hessians.resize(rows, cols);
//         for (int j = 0; j < rows; ++j) {
//             for (int k = j; k < cols; ++k) {
//                 Eigen::VectorXd x_pp = x;
//                 Eigen::VectorXd x_pm = x;
//                 Eigen::VectorXd x_mp = x;
//                 Eigen::VectorXd x_mm = x;
//                 x_pp(j) += eps; x_pp(k) += eps;
//                 x_pm(j) += eps; x_pm(k) -= eps;
//                 x_mp(j) -= eps; x_mp(k) += eps;
//                 x_mm(j) -= eps; x_mm(k) -= eps;

//                 double second_derivative = (f(x_pp, u) - f(x_pm, u) - f(x_mp, u) + f(x_mm, u)) / (4 * eps * eps);
//                 hessians(j, k) = second_derivative;
//                 if (j != k) {
//                     hessians(k, j) = second_derivative;
//                 }
//             }
//         }
//     }
//     else if (variable == "xu") {
//         rows = x.size();
//         cols = u.size();
//         hessians.resize(rows, cols);
//         for (int j = 0; j < rows; ++j) {
//             for (int k = 0; k < cols; ++k) {
//                 Eigen::VectorXd x_p = x;
//                 Eigen::VectorXd x_m = x;
//                 Eigen::VectorXd u_p = u;
//                 Eigen::VectorXd u_m = u;
//                 x_p(j) += eps;
//                 x_m(j) -= eps;
//                 u_p(k) += eps;
//                 u_m(k) -= eps;

//                 double second_derivative = (f(x_p, u_p) - f(x_p, u_m) - f(x_m, u_p) + f(x_m, u_m)) / (4 * eps * eps);
//                 hessians(j, k) = second_derivative;
//                 // if (j != k) {
//                 //     hessians(k, j) = second_derivative;
//                 // }
//             }
//         }
//     }
//     else if (variable == "uu") {
//         rows = u.size();
//         cols = u.size();
//         hessians.resize(rows, cols);
//         for (int j = 0; j < rows; ++j) {
//             for (int k = j; k < cols; ++k) {
//                 Eigen::VectorXd u_pp = u;
//                 Eigen::VectorXd u_pm = u;
//                 Eigen::VectorXd u_mp = u;
//                 Eigen::VectorXd u_mm = u;
//                 u_pp(j) += eps; u_pp(k) += eps;
//                 u_pm(j) += eps; u_pm(k) -= eps;
//                 u_mp(j) -= eps; u_mp(k) += eps;
//                 u_mm(j) -= eps; u_mm(k) -= eps;

//                 double second_derivative = (f(x, u_pp) - f(x, u_pm) - f(x, u_mp) + f(x, u_mm)) / (4 * eps * eps);
//                 hessians(j, k) = second_derivative;
//                 if (j != k) {
//                     hessians(k, j) = second_derivative;
//                 }
//             }
//         }
//     }
//     else {throw std::invalid_argument("Invalid variable. Use 'xx', 'uu' or 'xu'.");}

//     return hessians;
// }

// template<typename Func>
// inline Eigen::MatrixXd scalarHessian(Func f, const Eigen::VectorXd& x, double eps = 1e-5) {
//     int rows;
//     int cols;
//     Eigen::MatrixXd hessians;
//     rows = x.size();
//     cols = x.size();
//     hessians.resize(rows, cols);
//     for (int j = 0; j < rows; ++j) {
//         for (int k = j; k < cols; ++k) {
//             Eigen::VectorXd x_pp = x;
//             Eigen::VectorXd x_pm = x;
//             Eigen::VectorXd x_mp = x;
//             Eigen::VectorXd x_mm = x;
//             x_pp(j) += eps; x_pp(k) += eps;
//             x_pm(j) += eps; x_pm(k) -= eps;
//             x_mp(j) -= eps; x_mp(k) += eps;
//             x_mm(j) -= eps; x_mm(k) -= eps;

//             double second_derivative = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps * eps);
//             hessians(j, k) = second_derivative;
//             if (j != k) {
//                 hessians(k, j) = second_derivative;
//             }
//         }
//     }

//     return hessians;
// }

/////////////////////////////////////////////////////////////////////////////


// template<typename Func>
// Eigen::MatrixXd calculateJacobianX(Func f, Eigen::VectorXd x, Eigen::VectorXd u, double eps = 1e-5) {
//     int dim_x = x.size();
//     int dim_f = f(x, u).size();
//     Eigen::MatrixXd jacobian(dim_f, dim_x);
//     for (int i = 0; i < dim_x; ++i) {
//         Eigen::VectorXd x_m = x;
//         Eigen::VectorXd x_p = x;
//         x_m(i) -= eps;
//         x_p(i) += eps;
//         jacobian.col(i) = (f(x_p, u) - f(x_m, u)) / (2 * eps);
//     }
//     return jacobian;
// }

// template<typename Func>
// Eigen::MatrixXd calculateJacobianU(Func f, Eigen::VectorXd x, Eigen::VectorXd u, double eps = 1e-5) {
//     int dim_u = u.size();
//     int dim_f = f(x, u).size();
//     Eigen::MatrixXd jacobian(dim_f, dim_u);
//     for (int i = 0; i < dim_u; ++i) {
//         Eigen::VectorXd u_m = u;
//         Eigen::VectorXd u_p = u;
//         u_m(i) -= eps;
//         u_p(i) += eps;
//         jacobian.col(i) = (f(x, u_p) - f(x, u_m)) / (2 * eps);
//     }
//     return jacobian;
// }

// template<typename Func>
// Eigen::MatrixXd calculateJacobian(Func f, Eigen::VectorXd x, Eigen::VectorXd u, double eps = 1e-5) {
//     int dim_x = x.size();
//     int dim_u = u.size();
//     int dim_f = f(x, u).size();
//     Eigen::MatrixXd jacobian(dim_f, dim_x + dim_u);
//     for (int i = 0; i < dim_x; ++i) {
//         Eigen::VectorXd x_m = x;
//         Eigen::VectorXd x_p = x;
//         x_m(i) -= eps;
//         x_p(i) += eps;
//         jacobian.col(i) = (f(x_p, u) - f(x_m, u)) / (2 * eps);
//     }
//     for (int j = dim_x; j < dim_x + dim_u; ++j) {
//         Eigen::VectorXd u_m = u;
//         Eigen::VectorXd u_p = u;
//         u_m(j - dim_x) -= eps;
//         u_p(j - dim_x) += eps;
//         jacobian.col(j) = (f(x, u_p) - f(x, u_m)) / (2 * eps);
//     }
//     return jacobian;
// }

// template<typename Func>
// Eigen::MatrixXd calculateJacobian(Func f, Eigen::VectorXd x, double eps = 1e-5) {
//     int dim_x = x.size();
//     int dim_f = f(x).size();
//     Eigen::MatrixXd jacobian(dim_f, dim_x);
//     for (int i = 0; i < dim_x; ++i) {
//         Eigen::VectorXd x_m = x;
//         Eigen::VectorXd x_p = x;
//         x_m(i) -= eps;
//         x_p(i) += eps;
//         jacobian.col(i) = (f(x_p) - f(x_m)) / (2 * eps);
//     }
//     return jacobian;
// }

// template<typename Func>
// Eigen::MatrixXd calculateHessianXX(Func f, Eigen::VectorXd x, Eigen::VectorXd u, double eps = 1e-5) {    
//     int dim_x = x.size();
//     int dim_f = f(x, u).size();
//     Eigen::MatrixXd hessians(dim_f * dim_x, dim_x);
//     for (int j = 0; j < dim_x; ++j) {
//         for (int k = j; k < dim_x; ++k) {
//             Eigen::VectorXd x_pp = x;
//             Eigen::VectorXd x_pm = x;
//             Eigen::VectorXd x_mp = x;
//             Eigen::VectorXd x_mm = x;
//             x_pp(j) += eps; x_pp(k) += eps;
//             x_pm(j) += eps; x_pm(k) -= eps;
//             x_mp(j) -= eps; x_mp(k) += eps;
//             x_mm(j) -= eps; x_mm(k) -= eps;

//             Eigen::VectorXd second_derivative = (f(x_pp, u) - f(x_pm, u) - f(x_mp, u) + f(x_mm, u)) / (4 * eps * eps);
//             for (int i = 0; i < dim_f; ++i) {
//                 hessians(i * dim_x + j, k) = second_derivative(i);
//                 if (j != k) {
//                     hessians(i * dim_x + k, j) = second_derivative(i);
//                 }
//             }
//         }
//     }
//     return hessians;
// }

// template<typename Func>
// Eigen::MatrixXd calculateHessianUU(Func f, Eigen::VectorXd x, Eigen::VectorXd u, double eps = 1e-5) {    
//     int dim_u = u.size();
//     int dim_f = f(x, u).size();
//     Eigen::MatrixXd hessians(dim_f * dim_u, dim_u);
//     for (int j = 0; j < dim_u; ++j) {
//         for (int k = j; k < dim_u; ++k) {
//             Eigen::VectorXd u_pp = u;
//             Eigen::VectorXd u_pm = u;
//             Eigen::VectorXd u_mp = u;
//             Eigen::VectorXd u_mm = u;
//             u_pp(j) += eps; u_pp(k) += eps;
//             u_pm(j) += eps; u_pm(k) -= eps;
//             u_mp(j) -= eps; u_mp(k) += eps;
//             u_mm(j) -= eps; u_mm(k) -= eps;

//             Eigen::VectorXd second_derivative = (f(x, u_pp) - f(x, u_pm) - f(x, u_mp) + f(x, u_mm)) / (4 * eps * eps);
            
//             for (int i = 0; i < dim_f; ++i) {
//                 hessians(i * dim_u + j, k) = second_derivative(i);
//                 if (j != k) {
//                     hessians(i * dim_u + k, j) = second_derivative(i);
//                 }
//             }
//         }
//     }
//     return hessians;
// }

// template<typename Func>
// Eigen::MatrixXd calculateHessianXU(Func f, Eigen::VectorXd x, Eigen::VectorXd u, double eps = 1e-5) {    
//     int dim_x = x.size();
//     int dim_u = u.size();
//     int dim_f = f(x, u).size();
//     Eigen::MatrixXd hessians(dim_f * dim_u, dim_f * dim_x);
//     for (int j = 0; j < dim_x; ++j) {
//         for (int k = 0; k < dim_u; ++k) {
//             Eigen::VectorXd x_p = x;
//             Eigen::VectorXd x_m = x;
//             Eigen::VectorXd u_p = u;
//             Eigen::VectorXd u_m = u;
//             x_p(j) += eps;
//             x_m(j) -= eps;
//             u_p(k) += eps;
//             u_m(k) -= eps;

//             Eigen::VectorXd second_derivative = (f(x_p, u_p) - f(x_p, u_m) - f(x_m, u_p) + f(x_m, u_m)) / (4 * eps * eps);
            
//             for (int i = 0; i < dim_f; ++i) {
//                 hessians(i * dim_f + j, i * dim_f + k) = second_derivative(i);
//                 if (j != k) {
//                     hessians(i * dim_f + k, i * dim_f + j) = second_derivative(i);
//                 }
//             }
//         }
//     }
//     return hessians;
// }

// template<typename Func>
// Eigen::MatrixXd calculateHessian(Func f, Eigen::VectorXd x, double eps = 1e-5) {    
//     int dim_x = x.size();
//     int dim_f = f(x).size();
//     Eigen::MatrixXd hessians(dim_f * dim_x, dim_f * dim_x);
//     for (int j = 0; j < dim_x; ++j) {
//         for (int k = j; k < dim_x; ++k) {
//             Eigen::VectorXd x_pp = x;
//             Eigen::VectorXd x_pm = x;
//             Eigen::VectorXd x_mp = x;
//             Eigen::VectorXd x_mm = x;
//             x_pp(j) += eps; x_pp(k) += eps;
//             x_pm(j) += eps; x_pm(k) -= eps;
//             x_mp(j) -= eps; x_mp(k) += eps;
//             x_mm(j) -= eps; x_mm(k) -= eps;

//             Eigen::VectorXd second_derivative = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps * eps);
            
//             for (int i = 0; i < dim_f; ++i) {
//                 hessians(i * dim_f + j, i * dim_f + k) = second_derivative(i);
//                 if (j != k) {
//                     hessians(i * dim_f + k, i * dim_f + j) = second_derivative(i);
//                 }
//             }
//         }
//     }
//     return hessians;
// }
