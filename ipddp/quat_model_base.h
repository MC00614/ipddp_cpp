#pragma once

#include "model_base.h"

class QuatModelBase : public ModelBase {
public:
    QuatModelBase(int q_idx);
    ~QuatModelBase();

    int q_idx;
    int q_dim = 4;
    Eigen::Matrix<double, 4, 3> H;

    Vector4dual2nd Phi(Vector3dual2nd w) {
        Vector4dual2nd phi;
        phi << 1, w;
        // phi.normalize();
        phi /= sqrt(1 + w.squaredNorm());
        return phi;
    }

    Matrix4dual2nd Lq(Vector4dual2nd q) {
        Matrix4dual2nd lq;
        lq << q(0), -q(1), -q(2), -q(3),
        q(1),  q(0), -q(3),  q(2),
        q(2),  q(3),  q(0), -q(1),
        q(3), -q(2),  q(1),  q(0);
        return lq;
    };

    inline Eigen::MatrixXd GG(const VectorXdual2nd& x, const VectorXdual2nd& u) {
        return Lq(f(x,u).segment(q_idx, 4)).cast<double>() * H;
    }

    inline Eigen::MatrixXd G(const VectorXdual2nd& x) {
        return Lq(x.segment(q_idx, 4)).cast<double>() * H;
    }

    inline Eigen::MatrixXd EE(const VectorXdual2nd& x, const VectorXdual2nd& u) {
        Eigen::MatrixXd ee = Eigen::MatrixXd::Identity(dim_x, dim_rn);
        ee.block(q_idx, q_idx, 4, 3) = GG(x,u);
        return ee;
    }

    inline Eigen::MatrixXd E(const VectorXdual2nd& x) {
        Eigen::MatrixXd ee = Eigen::MatrixXd::Identity(dim_x, dim_rn);
        ee.block(q_idx, q_idx, 4, 3) = G(x);
        return ee;
    }

    inline Eigen::MatrixXd Id(const VectorXdual2nd& x, const double &fq_q) {
        Eigen::MatrixXd id = Eigen::MatrixXd::Zero(dim_rn, dim_rn);
        id.block(q_idx, q_idx, 3, 3) = fq_q * Eigen::MatrixXd::Identity(3,3);
        return id;
    }

    virtual Eigen::MatrixXd fx(VectorXdual2nd& x, VectorXdual2nd& u) override {
        return EE(x,u).transpose() * jacobian(f, wrt(x), at(x,u)) * E(x);
    }
    virtual Eigen::MatrixXd fu(VectorXdual2nd& x, VectorXdual2nd& u) override {
        return EE(x,u).transpose() * jacobian(f, wrt(u), at(x,u));
    }
    virtual Eigen::VectorXd px(VectorXdual2nd& x) override {
        return E(x).transpose() * gradient(p, wrt(x), at(x));
    }
    virtual Eigen::MatrixXd pxx(VectorXdual2nd& x) {
        Eigen::VectorXd px = gradient(p, wrt(x), at(x));
        double pqq = (px.segment(q_idx, q_dim).transpose() * x.segment(q_idx, q_dim)).cast<double>()(0);
        return E(x).transpose() * hessian(p, wrt(x), at(x)) * E(x) - Id(x, pqq);
    }
    virtual Eigen::VectorXd qx(VectorXdual2nd& x, VectorXdual2nd& u) override {
        return E(x).transpose() * gradient(q, wrt(x), at(x,u));
    }
    virtual Eigen::MatrixXd qdd(VectorXdual2nd& x, VectorXdual2nd& u) override{
        Eigen::VectorXd qx = gradient(q, wrt(x), at(x, u));
        double qqq = (qx.segment(q_idx, q_dim).transpose() * x.segment(q_idx, q_dim)).cast<double>()(0);
        Eigen::MatrixXd qdd = hessian(q, wrt(x, u), at(x, u));
        Eigen::MatrixXd quat_qdd = Eigen::MatrixXd::Zero(dim_rn + dim_u, dim_rn + dim_u);
        quat_qdd.topLeftCorner(dim_rn, dim_rn) = E(x).transpose() * qdd.topLeftCorner(dim_x, dim_x) * E(x) - Id(x, qqq);
        quat_qdd.block(0, dim_rn, dim_rn, dim_u) = E(x).transpose() * qdd.block(0, dim_x, dim_x, dim_u);
        quat_qdd.bottomRightCorner(dim_u, dim_u) = qdd.bottomRightCorner(dim_u, dim_u);
        return quat_qdd;
    }
    virtual Eigen::MatrixXd cx(VectorXdual2nd& x, VectorXdual2nd& u) override{
        return jacobian(c, wrt(x), at(x, u)) * E(x);
    }
    virtual Eigen::MatrixXd ecx(VectorXdual2nd& x, VectorXdual2nd& u) override{
        return jacobian(ec, wrt(x), at(x, u)) * E(x);
    }
    virtual Eigen::MatrixXd cTx(VectorXdual2nd& x) override{
        return jacobian(cT, wrt(x), at(x)) * E(x);
    }
    virtual Eigen::MatrixXd ecTx(VectorXdual2nd& x) override{
        return jacobian(ecT, wrt(x), at(x)) * E(x);
    }
    virtual Eigen::VectorXd perturb(const Eigen::VectorXd& xn, const Eigen::VectorXd& x) override{
        Eigen::VectorXd dx(dim_rn);
        Eigen::VectorXd q_qn = Lq(x.segment(q_idx, q_dim)).cast<double>().transpose() * xn.segment(q_idx, q_dim);
        dx << xn.segment(0,q_idx) - x.segment(0,q_idx),
            q_qn.segment(1,3)/q_qn(0);
        return dx;
    }
};

QuatModelBase::QuatModelBase(int q_idx) : q_idx(q_idx) {
    H << 0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1;
}

QuatModelBase::~QuatModelBase() {
}

/*
#pragma once

#include "model_base.h"

class QuatModelBase : public ModelBase {
public:
    QuatModelBase(int q_idx);
    ~QuatModelBase();

    int q_idx;
    int q_dim = 4;
    Eigen::Matrix<double, 4, 3> H;

    Vector4dual2nd Phi(Vector3dual2nd w) {
        Vector4dual2nd phi;
        phi << 1, w;
        phi /= sqrt(1 + w.squaredNorm());
        return phi;
    }

    Matrix4dual2nd Lq(Vector4dual2nd q) const {
        Matrix4dual2nd lq;
        lq << q(0), -q(1), -q(2), -q(3),
              q(1),  q(0), -q(3),  q(2),
              q(2),  q(3),  q(0), -q(1),
              q(3), -q(2),  q(1),  q(0);
        return lq;
    };

    inline Eigen::MatrixXd GG(const VectorXdual2nd& x, const VectorXdual2nd& u) const {
        return Lq(f(x,u).segment(q_idx, 4)).cast<double>() * H;
    }

    inline Eigen::MatrixXd G(const VectorXdual2nd& x) const {
        return Lq(x.segment(q_idx, 4)).cast<double>() * H;
    }

    inline void calcEE(const VectorXdual2nd& fv, Eigen::MatrixXd& ee) const {
        ee.setIdentity(dim_x, dim_rn);
        ee.block(q_idx, q_idx, 4, 3) = Lq(fv.segment(q_idx, 4)).cast<double>() * H;
    }

    inline Eigen::MatrixXd EE(const VectorXdual2nd& x, const VectorXdual2nd& u) {
        Eigen::MatrixXd ee(dim_x, dim_rn);
        calcEE(f(x,u), ee);
        return ee;
    }

    inline void calcE(const VectorXdual2nd& x, Eigen::MatrixXd& ee) const {
        ee.setIdentity(dim_x, dim_rn);
        ee.block(q_idx, q_idx, 4, 3) = G(x);
    }

    inline Eigen::MatrixXd E(const VectorXdual2nd& x) {
        Eigen::MatrixXd ee(dim_x, dim_rn);
        calcE(x, ee);
        return ee;
    }

    inline Eigen::MatrixXd Id(const double &fq_q) const {
        Eigen::MatrixXd id = Eigen::MatrixXd::Zero(dim_rn, dim_rn);
        id.block(q_idx, q_idx, 3, 3) = fq_q * Eigen::MatrixXd::Identity(3,3);
        return id;
    }

    virtual Eigen::MatrixXd fx(VectorXdual2nd& x, VectorXdual2nd& u) override {
        VectorXdual2nd fv;
        Eigen::MatrixXd Jx = jacobian(f, wrt(x), at(x,u), fv);
        Eigen::MatrixXd E_x(dim_x, dim_rn), EE_xu(dim_x, dim_rn);
        calcE(x, E_x);
        calcEE(fv, EE_xu);
        return EE_xu.transpose() * Jx * E_x;
    }

    virtual Eigen::MatrixXd fu(VectorXdual2nd& x, VectorXdual2nd& u) override {
        VectorXdual2nd fv;
        Eigen::MatrixXd Ju = jacobian(f, wrt(u), at(x,u), fv);
        Eigen::MatrixXd EE_xu(dim_x, dim_rn);
        calcEE(fv, EE_xu);
        return EE_xu.transpose() * Ju;
    }

    virtual Eigen::VectorXd px(VectorXdual2nd& x) override {
        Eigen::MatrixXd E_x(dim_x, dim_rn);
        calcE(x, E_x);
        return E_x.transpose() * gradient(p, wrt(x), at(x));
    }

    virtual Eigen::MatrixXd pxx(VectorXdual2nd& x) override {
        Eigen::MatrixXd E_x(dim_x, dim_rn);
        calcE(x, E_x);
        Eigen::VectorXd pxv = gradient(p, wrt(x), at(x));
        double pqq = (pxv.segment(q_idx, q_dim).transpose() * x.segment(q_idx, q_dim)).cast<double>()(0);
        return E_x.transpose() * hessian(p, wrt(x), at(x)) * E_x - Id(pqq);
    }

    virtual Eigen::VectorXd qx(VectorXdual2nd& x, VectorXdual2nd& u) override {
        Eigen::MatrixXd E_x(dim_x, dim_rn);
        calcE(x, E_x);
        return E_x.transpose() * gradient(q, wrt(x), at(x,u));
    }

    virtual Eigen::MatrixXd qdd(VectorXdual2nd& x, VectorXdual2nd& u) override {
        Eigen::MatrixXd E_x(dim_x, dim_rn);
        calcE(x, E_x);
        Eigen::VectorXd qxv = gradient(q, wrt(x), at(x, u));
        double qqq = (qxv.segment(q_idx, q_dim).transpose() * x.segment(q_idx, q_dim)).cast<double>()(0);
        Eigen::MatrixXd qddm = hessian(q, wrt(x, u), at(x, u));
        Eigen::MatrixXd quat_qdd = Eigen::MatrixXd::Zero(dim_rn + dim_u, dim_rn + dim_u);
        quat_qdd.topLeftCorner(dim_rn, dim_rn) = E_x.transpose() * qddm.topLeftCorner(dim_x, dim_x) * E_x - Id(qqq);
        quat_qdd.block(0, dim_rn, dim_rn, dim_u) = E_x.transpose() * qddm.block(0, dim_x, dim_x, dim_u);
        quat_qdd.bottomRightCorner(dim_u, dim_u) = qddm.bottomRightCorner(dim_u, dim_u);
        return quat_qdd;
    }

    virtual Eigen::MatrixXd cx(VectorXdual2nd& x, VectorXdual2nd& u) override {
        Eigen::MatrixXd E_x(dim_x, dim_rn);
        calcE(x, E_x);
        return jacobian(c, wrt(x), at(x, u)) * E_x;
    }

    virtual Eigen::MatrixXd ecx(VectorXdual2nd& x, VectorXdual2nd& u) override {
        Eigen::MatrixXd E_x(dim_x, dim_rn);
        calcE(x, E_x);
        return jacobian(ec, wrt(x), at(x, u)) * E_x;
    }

    virtual Eigen::MatrixXd cTx(VectorXdual2nd& x) override {
        Eigen::MatrixXd E_x(dim_x, dim_rn);
        calcE(x, E_x);
        return jacobian(cT, wrt(x), at(x)) * E_x;
    }

    virtual Eigen::MatrixXd ecTx(VectorXdual2nd& x) override {
        Eigen::MatrixXd E_x(dim_x, dim_rn);
        calcE(x, E_x);
        return jacobian(ecT, wrt(x), at(x)) * E_x;
    }

    virtual Eigen::VectorXd perturb(const Eigen::VectorXd& xn, const Eigen::VectorXd& x) override {
        Eigen::VectorXd dx(dim_rn);
        Eigen::VectorXd q_qn = Lq(x.segment(q_idx, q_dim)).cast<double>().transpose() * xn.segment(q_idx, q_dim);
        dx << xn.segment(0, q_idx) - x.segment(0, q_idx),
              q_qn.segment(1, 3) / q_qn(0);
        return dx;
    }
};

QuatModelBase::QuatModelBase(int q_idx) : q_idx(q_idx) {
    H << 0, 0, 0,
         1, 0, 0,
         0, 1, 0,
         0, 0, 1;
}

QuatModelBase::~QuatModelBase() {
}

*/