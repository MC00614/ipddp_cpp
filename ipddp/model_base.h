#pragma once

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

#include <eigen3/Eigen/Dense>

class ModelBase {
public:
    ModelBase();
    ~ModelBase();

    int N; // Number of Time Steps
    int dim_x; // State Dimension
    int dim_u; // Input Dimension
    int dim_g = 0; // Nonnegative Orthant Constraint Dimension
    int dim_h = 0; // Connic Constraint Dimension
    int dim_c = 0; // Inequality Constraint Dimension
    int dim_ec = 0; // Equality Constraint Dimension

    int dim_gT = 0; // Nonnegative Orthant Constraint Dimension (Terminal)
    int dim_hT = 0; // Connic Constraint Dimension (Terminal)
    int dim_cT = 0; // Inequality Constraint Dimension (Terminal)
    int dim_ecT = 0; // Equality Constraint Dimension (Terminal)

    int dim_rn; // Reduced State Dimension (for Quaternion)
    std::vector<int> dim_hs; // Connic Constraint Dimension Stack
    std::vector<int> dim_hTs; // Connic Constraint Dimension Stack (Terminal)
    
    Eigen::MatrixXd X_init; // State Vector Initial Guess
    Eigen::MatrixXd U_init; // Input Vector Initial Guess
    Eigen::MatrixXd Z_init; // Equality Lagrangian Initial Guess
    Eigen::MatrixXd R_init; // Equality Lagrangian Initial Guess
    Eigen::MatrixXd S_init; // Inequality Lagrangianx Initial Guess
    Eigen::MatrixXd Y_init; // Inequality Slack Varible Initial Guess

    Eigen::VectorXd ZT_init; // Equality Lagrangian Initial Guess (Terminal)
    Eigen::VectorXd RT_init; // Equality Lagrangian Initial Guess (Terminal)
    Eigen::VectorXd ST_init; // Inequality Lagrangian Initial Guess (Terminal)
    Eigen::VectorXd YT_init; // Inequality Slack Varible Initial Guess (Terminal)

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
    virtual Eigen::MatrixXd ecx(VectorXdual2nd& x, VectorXdual2nd& u) {
        return jacobian(ec, wrt(x), at(x, u));
    }
    virtual Eigen::MatrixXd ecu(VectorXdual2nd& x, VectorXdual2nd& u) {
        return jacobian(ec, wrt(u), at(x, u));
    }
    virtual Eigen::MatrixXd cTx(VectorXdual2nd& x) {
        return jacobian(cT, wrt(x), at(x));
    }
    virtual Eigen::MatrixXd ecTx(VectorXdual2nd& x) {
        return jacobian(ecT, wrt(x), at(x));
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