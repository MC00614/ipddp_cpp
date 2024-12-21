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
    int dim_g;
    int dim_h;
    int dim_h2;
    int dim_c;
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
    std::function<VectorXdual2nd(VectorXdual2nd, VectorXdual2nd)> h2;
    // Constraint Stack
    std::function<VectorXdual2nd(VectorXdual2nd, VectorXdual2nd)> c;
};

ModelBase::ModelBase() {
};

ModelBase::~ModelBase() {
};


