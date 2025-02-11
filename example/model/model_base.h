#pragma once

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

#include <eigen3/Eigen/Dense>

class ModelBase {
public:
    ModelBase();
    ~ModelBase();

    int N;
    int dim_x;
    int dim_u;
    int dim_g = 0;
    int dim_h = 0;
    int dim_c = 0;
    int dim_rn;
    std::vector<int> dim_hs;
    
    Eigen::VectorXd x0;
    Eigen::MatrixXd X_init;
    Eigen::MatrixXd U_init;
    Eigen::MatrixXd Y_init;
    Eigen::MatrixXd S_init;

    // Discrete Time System
    std::function<VectorXdual2nd(VectorXdual2nd, VectorXdual2nd)> f;
    // Stage Cost Function
    std::function<dual2nd(VectorXdual2nd, VectorXdual2nd)> q;
    // Terminal Cost Function
    std::function<dual2nd(VectorXdual2nd)> p;
    // Nonnegative Orthant Constraint Mapping
    std::function<VectorXdual2nd(VectorXdual2nd, VectorXdual2nd)> g;
    // Connic Constraint Mapping
    std::function<VectorXdual2nd(VectorXdual2nd, VectorXdual2nd)> h;
    std::vector<std::function<VectorXdual2nd(VectorXdual2nd, VectorXdual2nd)>> hs;
    // Constraint Stack
    std::function<VectorXdual2nd(VectorXdual2nd, VectorXdual2nd)> c;

    // Differential Functions
    virtual Eigen::MatrixXd fx(VectorXdual2nd& x, VectorXdual2nd& u) {
        return jacobian(f, wrt(x), at(x, u));
    }
    virtual Eigen::MatrixXd fu(VectorXdual2nd& x, VectorXdual2nd& u) {
        return jacobian(f, wrt(u), at(x, u));
    }
    virtual Eigen::VectorXd px(VectorXdual2nd& x) {
        return gradient(p, wrt(x), at(x));
    }
    virtual Eigen::MatrixXd pxx(VectorXdual2nd& x) {
        return hessian(p, wrt(x), at(x));
    }
    virtual Eigen::VectorXd qx(VectorXdual2nd& x, VectorXdual2nd& u) {
        return gradient(q, wrt(x), at(x, u));
    }
    virtual Eigen::VectorXd qu(VectorXdual2nd& x, VectorXdual2nd& u) {
        return gradient(q, wrt(u), at(x, u));
    }
    virtual Eigen::MatrixXd qdd(VectorXdual2nd& x, VectorXdual2nd& u) {
        return hessian(q, wrt(x, u), at(x, u));
    }
    virtual Eigen::MatrixXd cx(VectorXdual2nd& x, VectorXdual2nd& u) {
        return jacobian(c, wrt(x), at(x, u));
    }
    virtual Eigen::MatrixXd cu(VectorXdual2nd& x, VectorXdual2nd& u) {
        return jacobian(c, wrt(u), at(x, u));
    }

    // Operator for Quaternion
    virtual Eigen::VectorXd perturb(const Eigen::VectorXd& xn, const Eigen::VectorXd& x) {
        return xn - x;
    } 
};

ModelBase::ModelBase() {
};

ModelBase::~ModelBase() {
};