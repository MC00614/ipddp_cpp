#include "helper_function.h"

#include <eigen3/Eigen/Dense>
#include <functional>
#include <cmath>

#include <iostream>

class SOC_IPDDP {
public:
    SOC_IPDDP();
    ~SOC_IPDDP();

    void init(int N);

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
    int n;
    std::function<Eigen::MatrixXd()> f;
    std::function<double()> q;
    std::function<double()> p;

    void backwardPass();
    void forwardPass();
    void regulate();
};


SOC_IPDDP::SOC_IPDDP() {
}

SOC_IPDDP::~SOC_IPDDP() {
}

void SOC_IPDDP::init(int n) {
    this->n = n;
}


template<typename Func>
void SOC_IPDDP::setSystemF(Func f) {
    
}

template<typename Func>
void SOC_IPDDP::setStageCostQ(Func f) {

}

template<typename Func>
void SOC_IPDDP::setTerminalCostP(Func f) {

}

void SOC_IPDDP::solve() {
    while (1)
    {
        this->backwardPass();
        this->forwardPass();
        this->regulate();
    }
    
}

void SOC_IPDDP::backwardPass() {}

void SOC_IPDDP::forwardPass() {}

void SOC_IPDDP::regulate() {}


Eigen::MatrixXd getX() {

}

Eigen::MatrixXd getU() {
    
}