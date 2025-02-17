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
    int dim_ec = 0;

    int dim_gT = 0;
    int dim_hT = 0;
    int dim_cT = 0;
    int dim_ecT = 0;

    int dim_rn;
    std::vector<int> dim_hs;
    std::vector<int> dim_hTs;
    
    Eigen::VectorXd x0;
    Eigen::MatrixXd X_init; // State Vector
    Eigen::MatrixXd U_init; // Input Vector
    Eigen::MatrixXd M_init; // Equality Lagrangian
    Eigen::MatrixXd S_init; // Inequality Lagrangianx
    Eigen::MatrixXd Y_init; // Slack Varible

    Eigen::VectorXd MT_init; // Equality Lagrangian (Terminal)
    Eigen::VectorXd ST_init; // Inequality Lagrangian (Terminal)
    Eigen::VectorXd YT_init; // Slack Varible (Terminal)

    // Dynamics and Cost Function
    std::function<VectorXdual2nd(VectorXdual2nd, VectorXdual2nd)> f; // Discrete Time System
    std::function<dual2nd(VectorXdual2nd, VectorXdual2nd)> q; // Stage Cost Function

    // Stage (State and Input)
    std::function<VectorXdual2nd(VectorXdual2nd, VectorXdual2nd)> g; // Nonnegative Orthant Constraint Mapping
    std::function<VectorXdual2nd(VectorXdual2nd, VectorXdual2nd)> h; // Connic Constraint Mapping
    std::vector<std::function<VectorXdual2nd(VectorXdual2nd, VectorXdual2nd)>> hs; // Connic Constraint Stack
    std::function<VectorXdual2nd(VectorXdual2nd, VectorXdual2nd)> c; // Inequality Constraint Stack
    std::function<VectorXdual2nd(VectorXdual2nd, VectorXdual2nd)> ec; // Equality Constraint Mapping
    
    // Terminal (State)
    std::function<dual2nd(VectorXdual2nd)> p; // Cost Function (Terminal)
    std::function<VectorXdual2nd(VectorXdual2nd)> gT; // Nonnegative Orthant Constraint Mapping (Terminal)
    std::function<VectorXdual2nd(VectorXdual2nd)> hT; // Connic Constraint Mapping (Terminal)
    std::vector<std::function<VectorXdual2nd(VectorXdual2nd)>> hTs; // Connic Constraint Stack (Terminal)
    std::function<VectorXdual2nd(VectorXdual2nd)> cT; // Inequality Constraint Stack (Terminal)
    std::function<VectorXdual2nd(VectorXdual2nd)> ecT; // Equality Constraint Mapping (Terminal)

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
    virtual Eigen::MatrixXd cTx(VectorXdual2nd& x) {
        return jacobian(cT, wrt(x), at(x));
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