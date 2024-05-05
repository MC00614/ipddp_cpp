#include "helper_function.h"

#include <eigen3/Eigen/Dense>
#include <functional>
#include <cmath>

#include <iostream>

class SOC_IPDDP {
public:
    SOC_IPDDP();
    ~SOC_IPDDP();

    void init(int N, int max_iter, Eigen::MatrixXd X, Eigen::MatrixXd U);

    template<typename Func>
    void setSystemF(Func f);
    template<typename Func>
    void setStageCostQ(Func q);
    template<typename Func>
    void setTerminalCostP(Func p);

    void solve();

    Eigen::MatrixXd getX();
    Eigen::MatrixXd getU();

private:
    int N;
    int max_iter;
    int dim_x;
    int dim_u;
    Eigen::MatrixXd X;
    Eigen::MatrixXd U;

    // Discrete Time System
    std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> f;
    // Stage Cost Function
    std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> q;
    // Terminal Cost Function
    std::function<Eigen::MatrixXd(Eigen::VectorXd)> p;
    void backwardPass();
    void forwardPass();
    void regulate();
};


SOC_IPDDP::SOC_IPDDP() {
}

SOC_IPDDP::~SOC_IPDDP() {
}

void SOC_IPDDP::init(int N, int max_iter, Eigen::MatrixXd X, Eigen::MatrixXd U) {
    this->N = N;
    this->max_iter = max_iter;
    this->X = X;
    this->U = U;

    this->dim_x = X.rows();
    this->dim_u = U.rows();
}


template<typename Func>
void SOC_IPDDP::setSystemF(Func f) {
    this->f = f;
}

template<typename Func>
void SOC_IPDDP::setStageCostQ(Func q) {
    this->q = q;
}

template<typename Func>
void SOC_IPDDP::setTerminalCostP(Func p) {
    this->p = p;
}

void SOC_IPDDP::solve() {
    int iter = 0;

    while (iter++ < this->max_iter) {
        this->backwardPass();
        break;
        this->forwardPass();
        this->regulate();
    }
    
}

void SOC_IPDDP::backwardPass() {    
    Eigen::MatrixXd fd;
    std::vector<Eigen::MatrixXd> fdd;
    Eigen::MatrixXd qd;
    std::vector<Eigen::MatrixXd> qdd;

    Eigen::MatrixXd Qx;
    Eigen::MatrixXd Qu;
    std::vector<Eigen::MatrixXd> Qxx;
    std::vector<Eigen::MatrixXd> Quu;
    std::vector<Eigen::MatrixXd> Qux;
    Eigen::MatrixXd Vx = calculateJacobian(p, X.col(N-1));
    std::vector<Eigen::MatrixXd> Vxx = calculateHessian(p, X.col(N-1));

    for (int t = N-1; t >= 0; --t) {
        fd = calculateJacobian(f, X.col(t), U.col(t));
        fdd = calculateHessian(f, X.col(t), U.col(t));
        qd = calculateJacobian(q, X.col(t), U.col(t));
        qdd = calculateHessian(q, X.col(t), U.col(t));


        Qx = qd.block(0,0,1,1);
        std::cout<<"qd : "<<qd<<std::endl;
        std::cout<<"Qx : "<<Qx<<std::endl;
        
    }

}

void SOC_IPDDP::forwardPass() {}

void SOC_IPDDP::regulate() {}


Eigen::MatrixXd getX() {

}

Eigen::MatrixXd getU() {
    
}