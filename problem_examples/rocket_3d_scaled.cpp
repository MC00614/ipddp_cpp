#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"

#include <cmath>
#include <chrono>

#define m_scale 30000.0
#define r_scale 559.016994375

// Dynamics
template <typename Scalar>
class ScaledRocket3D : public DiscreteDynamicsBase<Scalar> {
public:
    Scalar m_wet = m_scale;
    Scalar m_dry = 22000.0;
    Scalar alpha_m = 1.0 / (282.0 * 9.81);
    Eigen::Vector3d gravity;
    Eigen::Matrix3d J_B;
    Eigen::Matrix3d J_B_inv;
    Eigen::Vector3d L_thrust;
    Eigen::Matrix<Scalar, 3, 3> skew_L;
    Eigen::Matrix4d dL_dq[4];
    
    ScaledRocket3D() {
        this->dim_x = 14;
        this->dim_u = 3;
        this->dt = 0.1;

        gravity << 0.0, 0.0, -9.81;

        J_B << 4000000., 0, 0,
                0, 4000000.0, 0,
                0, 0, 100000.0;
                
        L_thrust << 0, 0, -14.0;
                
        dL_dq[0] << 1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1;

        dL_dq[1] << 0, -1, 0, 0,
                    1, 0, 0, 0,
                    0, 0, 0, -1,
                    0, 0, 1, 0;

        dL_dq[2] << 0, 0, -1, 0,
                    0, 0, 0, 1,
                    1, 0, 0, 0,
                    0, -1, 0, 0;

        dL_dq[3] << 0, 0, 0, -1,
                    0, 0, 1, 0,
                    0, -1, 0, 0,
                    1, 0, 0, 0;

        alpha_m *= r_scale;
        L_thrust /= r_scale;
        gravity /= r_scale;
        J_B /= (m_scale * (r_scale * r_scale));
        J_B_inv = J_B.inverse();
        m_wet /= m_scale;
        m_dry /= m_scale;

        J_B_inv = J_B.inverse();
        skew_L = skewSymmetric(L_thrust);
    }

    static inline Eigen::Matrix<Scalar, 4, 1> Phi(const Eigen::Matrix<Scalar, 3, 1>& w_dt) {
        Eigen::Matrix<Scalar, 4, 1> phi;
        phi << 1.0, w_dt;
        phi /= std::sqrt(1.0 + w_dt.squaredNorm());
        return phi;
    };

    static inline Eigen::Matrix<Scalar, 3, 3> calcC(const Eigen::Vector4d& q) {
        Eigen::Matrix<Scalar, 3, 3> C;
        C << 1 - 2 * (q(2) * q(2) + q(3) * q(3)), 2 * (q(1) * q(2) - q(0) * q(3)), 2 * (q(1) * q(3) + q(0) * q(2)),
             2 * (q(1) * q(2) + q(0) * q(3)), 1 - 2 * (q(1) * q(1) + q(3) * q(3)), 2 * (q(2) * q(3) - q(0) * q(1)),
             2 * (q(1) * q(3) - q(0) * q(2)), 2 * (q(2) * q(3) + q(0) * q(1)), 1 - 2 * (q(1) * q(1) + q(2) * q(2));
        return C;
    };

    static inline Eigen::Matrix<Scalar, 4, 4> calcL(const Eigen::Vector4d& q) {
        Eigen::Matrix<Scalar, 4, 4> L;
        L << q(0), -q(1), -q(2), -q(3),
             q(1),  q(0), -q(3),  q(2),
             q(2),  q(3),  q(0), -q(1),
             q(3), -q(2),  q(1),  q(0);
        return L;
    };

    static inline Eigen::Matrix<Scalar,3,3> skewSymmetric(const Eigen::Matrix<Scalar, 3, 1>& v) {
        Eigen::Matrix<Scalar,3,3> skew_m;
        skew_m <<    0, -v(2),  v(1),
             v(2),     0, -v(0),
            -v(1),  v(0),     0;
        return skew_m;
    }

    Vector<Scalar> f(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Vector<Scalar> x_n(this->dim_x);

        Scalar m = x(0);
        Eigen::Vector3d r = x.segment(1, 3);
        Eigen::Vector3d v = x.segment(4, 3);
        Eigen::Vector3d w = x.segment(7, 3);
        Eigen::Vector4d q = x.segment(10, 4);

        Eigen::Vector3d u_ = u;

        Eigen::Matrix<Scalar, 3, 3> C = calcC(q);

        Eigen::Matrix<Scalar, 4, 4> Omega;
        Omega <<  0,    -w(0), -w(1), -w(2),
                  w(0),  0,     w(2), -w(1),
                  w(1), -w(2),  0,     w(0),
                  w(2),  w(1), -w(0),  0;

        x_n(0) = m + this->dt * (- alpha_m * u.norm());
        x_n.segment(1, 3) = r + this->dt * v;
        x_n.segment(4, 3) = v + this->dt * (gravity + (C * u) / m);
        x_n.segment(7, 3) = w + this->dt * (J_B_inv * (L_thrust.cross(u_) - w.cross(J_B * w)));

        Eigen::Matrix<Scalar, 4, 4> L = calcL(q);

        Eigen::Matrix<Scalar, 4, 1> q_next = L * Phi(w * this->dt / 2.0);
        q_next.normalize();
        x_n.segment(10, 4) = q_next;

        return x_n;
    }

    Matrix<Scalar> fx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> Fx = Matrix<Scalar>::Identity(this->dim_x, this->dim_x);

        Scalar m = x(0);
        Eigen::Vector3d r = x.segment(1, 3);
        Eigen::Vector3d v = x.segment(4, 3);
        Eigen::Vector3d w = x.segment(7, 3);
        Eigen::Vector4d q = x.segment(10, 4);

        Fx.block(1, 4, 3, 3).setIdentity();
        Fx.block(1, 4, 3, 3) *= this->dt;

        Eigen::Matrix<Scalar, 3, 3> Jw_cross = skewSymmetric(J_B * w);
        Eigen::Matrix<Scalar, 3, 3> w_cross = skewSymmetric(w);
        Fx.block(7, 7, 3, 3) += this->dt * J_B_inv * (Jw_cross - w_cross * J_B);

        Eigen::Matrix<Scalar, 3, 3> dCdq0, dCdq1, dCdq2, dCdq3;
        dCdq0 << 0, -2 * q(3),  2 * q(2),
                2 * q(3), 0, -2 * q(1),
                -2 * q(2), 2 * q(1), 0;
        dCdq1 << 0, 2 * q(2), 2 * q(3),
                2 * q(2), -4 * q(1), -2 * q(0),
                2 * q(3), 2 * q(0), -4 * q(1);
        dCdq2 << -4 * q(2), 2 * q(1), 2 * q(0),
                2 * q(1), 0, 2 * q(3),
                -2 * q(0), 2 * q(3), -4 * q(2);
        dCdq3 << -4 * q(3), -2 * q(0), 2 * q(1),
                2 * q(0), -4 * q(3), 2 * q(2),
                2 * q(1), 2 * q(2), 0;
        Fx.block(4, 10, 3, 1) = this->dt * dCdq0 * u / m;
        Fx.block(4, 11, 3, 1) = this->dt * dCdq1 * u / m;
        Fx.block(4, 12, 3, 1) = this->dt * dCdq2 * u / m;
        Fx.block(4, 13, 3, 1) = this->dt * dCdq3 * u / m;

        Eigen::Matrix<Scalar, 3, 3> C = calcC(q);
        Fx.block(4, 0, 3, 1) = - this->dt * (C * u) / (m * m);

        Eigen::Matrix<Scalar,3,3> I = Eigen::Matrix<Scalar,3,3>::Identity();
        Scalar w2 = w.squaredNorm();
        Scalar alpha = this->dt / 2.0;
        Scalar denom = std::sqrt(1.0 + alpha * alpha * w2);
        Eigen::Matrix<Scalar,3,3> dphi_dt_w = alpha / denom * (I - (alpha * alpha / (1.0 + w2)) * (w * w.transpose()));

        Eigen::Matrix<Scalar,4,3> dphi_dw = Eigen::Matrix<Scalar,4,3>::Zero();
        dphi_dw.block(1,0,3,3) = dphi_dt_w;

        Eigen::Matrix<Scalar,4,4> L = calcL(q);
        Fx.block(10,7,4,3) = L * dphi_dw;

        Eigen::Matrix<Scalar, 4, 1> phi = Phi(w * this->dt / 2.0);
        for (int i = 0; i < 4; ++i) {
            Fx.block(10, 10 + i, 4, 1) = dL_dq[i] * phi;
        }

        return Fx;
    }

    Matrix<Scalar> fu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> Fu = Matrix<Scalar>::Zero(this->dim_x, this->dim_u);

        Scalar m = x(0);
        Eigen::Vector4d q = x.segment(10, 4);

        Fu.block(0, 0, 1, 3) = - this->dt * alpha_m * u.transpose() / u.norm();

        Eigen::Matrix<Scalar, 3, 3> C = calcC(q);

        Fu.block(4, 0, 3, 3) = this->dt * C / m;

        Fu.block(7, 0, 3, 3) = this->dt * J_B_inv * skew_L;

        return Fu;
    }
};

template <typename Scalar>
class MinMaxInput : public StageConstraintBase<Scalar> {
private:
    Scalar umax;
    Scalar umin;

public:
    MinMaxInput() {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 2;

        Scalar mass = 10.0;
        Scalar gravity = 9.81;
        umax = mass * 9.81 * 1.1;
        umin = mass * 9.81 * 0.6;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Vector<Scalar> c_n(2);
        c_n(0) = umax - u.norm();
        c_n(1) = u.norm() - umin;
        return -c_n;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(2, x.size());
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(2, u.size());
        J.row(0) = -u.transpose() / u.norm();
        J.row(1) = u.transpose() / u.norm();
        return -J;
    }
};

int main() {
    using Scalar = double;

    int N = 30;
    OptimalControlProblem<Scalar> problem(N);


    problem.setStageDynamics(std::make_shared<ScaledRocket3D<Scalar>>());

    Scalar Q = 0.0;
    Scalar R = 1e-6;
    problem.setStageCost(std::make_shared<ScalarQuadraticStageCost<Scalar>>(Q, R));
    Scalar QT = 0.0;
    problem.setTerminalCost(std::make_shared<ScalarQuadraticTerminalCost<Scalar>>(QT));


    Matrix<Scalar> inputcone_cx = Matrix<Scalar>::Zero(3, 14);
    Matrix<Scalar> inputcone_cu = Matrix<Scalar>::Zero(3, 3);
    inputcone_cu(0, 2) = - std::tan(7.0 * M_PI / 180.0);
    inputcone_cu(1, 0) = - 1.0;
    inputcone_cu(2, 1) = - 1.0;
    auto inputcone_c0 = Vector<Scalar>::Zero(3);
    problem.addStageConstraint(std::make_shared<LinearStageConstraint<Scalar>>(
        inputcone_cx, inputcone_cu, inputcone_c0, ConstraintType::SOC));

    
    Matrix<Scalar> glideslope_cx = Matrix<Scalar>::Zero(3, 14);
    glideslope_cx(0, 3) = - std::tan(60.0 * M_PI / 180.0);
    glideslope_cx(1, 1) = - 1.0;
    glideslope_cx(2, 2) = - 1.0;
    Matrix<Scalar> glideslope_cu = Matrix<Scalar>::Zero(3, 3);
    auto glideslope_c0 = Vector<Scalar>::Zero(3);
    problem.addStageConstraint(std::make_shared<LinearStageConstraint<Scalar>>(
        glideslope_cx, glideslope_cu, glideslope_c0, ConstraintType::SOC));


    // problem.addStageConstraint(std::make_shared<MinMaxInput<Scalar>>());


    // Matrix<Scalar> CT = Matrix<Scalar>::Zero(13, 14);
    // CT.block(0, 1, 13, 13).setIdentity();
    // Vector<Scalar> c0T = Vector<Scalar>::Zero(13);
    // c0T(2) = - 1.0;
    // c0T(9) = - 1.0;
    // c0T.segment(0, 6) /= r_scale;
    // problem.addTerminalConstraint(std::make_shared<LinearTerminalConstraint<Scalar>>(
    //     CT, c0T, ConstraintType::EQ));


    Vector<Scalar> x0 = Vector<Scalar>::Zero(14);
    x0(0) = 30000.0;
    x0(1) = 150.0;
    x0(2) = 200.0;
    x0(3) = 500.0;
    // x0(3) = -2.0;
    // x0(4) = -2.5;
    // x0(5) = 3.0;
    x0(10) = 1.0; // Quaternion
    x0(0) /= m_scale;
    x0.segment(1, 6) /= r_scale;
    problem.setInitialState(0, x0);

    Vector<Scalar> u0 = Vector<Scalar>::Zero(3);
    u0(2) = 800000.0 * 0.7 / 2.0;
    u0 /= (m_scale * r_scale);
    problem.setInitialControl(u0);

    Param param;
    param.is_quaternion_in_state = true;
    param.quaternion_idx = 10; // Quaternion starts at index 10

    param.rho_max = 1e8;
    param.mu_min = 1e-8;

    clock_t start = clock();
    ALIPDDP<Scalar> solver(problem);
    solver.init(param);
    solver.solve();

    clock_t finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "\nIn Total : " << duration << " Seconds" << std::endl;

    // Parse Result
    std::vector<Eigen::VectorXd> X_result = solver.getResX();
    std::vector<Eigen::VectorXd> U_result = solver.getResU();
    std::vector<double> all_cost = solver.getAllCost();

    for (int k = 0; k < problem.getHorizon() + 1; ++k) {
        X_result[k](0) *= m_scale;
        X_result[k].segment(1, 6) *= r_scale;
    }

    for (int k = 0; k < problem.getHorizon(); ++k) {
        U_result[k] *= (m_scale * r_scale);
    }

    // std::cout << "X_result = " << std::endl;
    // for (int k = 0; k < problem.getHorizon() + 1; ++k) {
    //     std::cout << X_result[k].transpose() << std::endl;
    // }

    // std::cout << "U_result = " << std::endl;
    // for (int k = 0; k < problem.getHorizon(); ++k) {
    //     std::cout << U_result[k].transpose() << std::endl;
    // }

    std::cout<<"X_last = \n"<<X_result[problem.getHorizon()].transpose()<<std::endl;
    std::cout<<"U_last = \n"<<U_result[problem.getHorizon() - 1].transpose()<<std::endl;

    return 0;
}