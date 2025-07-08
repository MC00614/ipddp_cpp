#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"

#include <cmath>
#include <chrono>

// Dynamics
template <typename Scalar>
class Rocket3D : public DiscreteDynamicsBase<Scalar> {
public:
    Scalar r = 0.4;
    Scalar l = 1.4;
    Scalar dt = 0.1;
    Scalar mass = 10.0;
    Scalar umax, umin;
    Eigen::Vector3d gravity;
    Eigen::Matrix3d J_B, J_B_inv;
    Eigen::Vector3d L_thrust;
    Eigen::Matrix4d dL_dq[4];
    
    Rocket3D() {
        this->dim_x = 13;
        this->dim_u = 3;
        this->dt = dt;

        gravity << 0.0, 0.0, -9.81;
        umax = mass * 9.81 * 1.1;
        umin = mass * 9.81 * 0.6;

        J_B << (1.0/12.0) * mass * (3 * r * r + l * l), 0, 0,
               0, (1.0/12.0) * mass * (3 * r * r + l * l), 0,
               0, 0, 0.5 * mass * r * r;
        J_B_inv = J_B.inverse();

        L_thrust << 0, 0, -l / 2;

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

    }

    static inline Eigen::Matrix<Scalar, 4, 1> Phi(const Eigen::Matrix<Scalar, 3, 1>& w_dt) {
        Eigen::Matrix<Scalar, 4, 1> phi;
        phi << 1.0, w_dt;
        phi /= std::sqrt(1.0 + w_dt.squaredNorm());
        return phi;
    };

    Vector<Scalar> f(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Vector<Scalar> x_n(this->dim_x);

        Eigen::Vector3d r = x.segment(0, 3);
        Eigen::Vector3d v = x.segment(3, 3);
        Eigen::Vector3d w = x.segment(6, 3);
        Eigen::Vector4d q = x.segment(9, 4);

        Eigen::Vector3d u_ = u;

        Eigen::Matrix<Scalar, 3, 3> C;
        C << 1 - 2 * (q(2) * q(2) + q(3) * q(3)), 2 * (q(1) * q(2) - q(0) * q(3)), 2 * (q(1) * q(3) + q(0) * q(2)),
             2 * (q(1) * q(2) + q(0) * q(3)), 1 - 2 * (q(1) * q(1) + q(3) * q(3)), 2 * (q(2) * q(3) - q(0) * q(1)),
             2 * (q(1) * q(3) - q(0) * q(2)), 2 * (q(2) * q(3) + q(0) * q(1)), 1 - 2 * (q(1) * q(1) + q(2) * q(2));

        Eigen::Matrix<Scalar, 4, 4> Omega;
        Omega <<  0,    -w(0), -w(1), -w(2),
                  w(0),  0,     w(2), -w(1),
                  w(1), -w(2),  0,     w(0),
                  w(2),  w(1), -w(0),  0;

        x_n.segment(0, 3) = r + this->dt * v;
        x_n.segment(3, 3) = v + this->dt * (gravity + (C * u) / mass);
        x_n.segment(6, 3) = w + this->dt * (J_B_inv * (L_thrust.cross(u_) - w.cross(J_B * w)));

        Eigen::Matrix<Scalar, 4, 4> L;
        L << q(0), -q(1), -q(2), -q(3),
             q(1),  q(0), -q(3),  q(2),
             q(2),  q(3),  q(0), -q(1),
             q(3), -q(2),  q(1),  q(0);

        Eigen::Matrix<Scalar, 4, 1> q_next = L * Phi(w * this->dt / 2.0);
        q_next.normalize();
        x_n.segment(9, 4) = q_next;

        return x_n;
    }

    Matrix<Scalar> fx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> Fx = Matrix<Scalar>::Identity(this->dim_x, this->dim_x);

        Eigen::Vector3d r = x.segment(0, 3);
        Eigen::Vector3d v = x.segment(3, 3);
        Eigen::Vector3d w = x.segment(6, 3);
        Eigen::Vector4d q = x.segment(9, 4);

        Eigen::Matrix<Scalar, 3, 3> C;
        C << 1 - 2 * (q(2) * q(2) + q(3) * q(3)), 2 * (q(1) * q(2) - q(0) * q(3)), 2 * (q(1) * q(3) + q(0) * q(2)),
            2 * (q(1) * q(2) + q(0) * q(3)), 1 - 2 * (q(1) * q(1) + q(3) * q(3)), 2 * (q(2) * q(3) - q(0) * q(1)),
            2 * (q(1) * q(3) - q(0) * q(2)), 2 * (q(2) * q(3) + q(0) * q(1)), 1 - 2 * (q(1) * q(1) + q(2) * q(2));

        Fx.block(0, 3, 3, 3).setIdentity();
        Fx.block(0, 3, 3, 3) *= this->dt;

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

        Fx.block(3, 9, 3, 1) = this->dt * dCdq0 * u / mass;
        Fx.block(3, 10, 3, 1) = this->dt * dCdq1 * u / mass;
        Fx.block(3, 11, 3, 1) = this->dt * dCdq2 * u / mass;
        Fx.block(3, 12, 3, 1) = this->dt * dCdq3 * u / mass;

        Eigen::Matrix<Scalar, 3, 3> skew_Jw;
        skew_Jw <<      0, -J_B(2,2)*w(1),  J_B(1,1)*w(2),
                J_B(2,2)*w(0),       0, -J_B(0,0)*w(2),
                -J_B(1,1)*w(0), J_B(0,0)*w(1),        0;
        Fx.block(6, 6, 3, 3) -= this->dt * J_B_inv * skew_Jw;

        Eigen::Matrix<Scalar, 3, 3> skew_Lu;
        skew_Lu <<       0, -u(2),  u(1),
                    u(2),     0, -u(0),
                    -u(1),  u(0),     0;

        Eigen::Matrix<Scalar, 3, 3> dLu_dw = -skew_Lu;
        Fx.block(6, 6, 3, 3) += this->dt * J_B_inv * (-skew_Lu);

        Eigen::Matrix<Scalar, 4, 4> L;
        L << q(0), -q(1), -q(2), -q(3),
            q(1),  q(0), -q(3),  q(2),
            q(2),  q(3),  q(0), -q(1),
            q(3), -q(2),  q(1),  q(0);

        Vector<Scalar> w_dt = w * this->dt / 2.0;
        Scalar w_norm2 = w_dt.squaredNorm();
        Scalar alpha = 1.0 / sqrt(1.0 + w_norm2);
        Vector<Scalar> phi(4);
        phi << 1.0, w_dt;
        phi *= alpha;

        Vector<Scalar> z = L * phi;
        Scalar norm_z = z.norm();
        Matrix<Scalar> D_norm = (Matrix<Scalar>::Identity(4, 4) - z * z.transpose() / (norm_z * norm_z)) / norm_z;

        Matrix<Scalar> dPhi_dw(4, 3);
        // dPhi_dw = -(alpha * alpha * alpha / 2.0) * w_dt * phi.transpose();
        dPhi_dw = -(alpha * alpha * alpha / 2.0) * phi * w_dt.transpose();
        dPhi_dw.block(1, 0, 3, 3) += alpha * Matrix<Scalar>::Identity(3, 3);

        Fx.block(9, 6, 4, 3) = D_norm * L * dPhi_dw;

        for (int i = 0; i < 4; ++i)
            Fx.block(9, 9 + i, 4, 1) = D_norm * (dL_dq[i] * phi);

        return Fx;
    }

    Matrix<Scalar> fu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> Fu = Matrix<Scalar>::Zero(this->dim_x, this->dim_u);

        Eigen::Vector4d q = x.segment(9, 4);

        Eigen::Matrix<Scalar, 3, 3> C;
        C << 1 - 2 * (q(2) * q(2) + q(3) * q(3)), 2 * (q(1) * q(2) - q(0) * q(3)), 2 * (q(1) * q(3) + q(0) * q(2)),
            2 * (q(1) * q(2) + q(0) * q(3)), 1 - 2 * (q(1) * q(1) + q(3) * q(3)), 2 * (q(2) * q(3) - q(0) * q(1)),
            2 * (q(1) * q(3) - q(0) * q(2)), 2 * (q(2) * q(3) + q(0) * q(1)), 1 - 2 * (q(1) * q(1) + q(2) * q(2));

        Fu.block(3, 0, 3, 3) = this->dt * C / mass;

        Eigen::Matrix<Scalar, 3, 3> skew_L;
        skew_L <<        0, -L_thrust(2),  L_thrust(1),
                    L_thrust(2),        0, -L_thrust(0),
                -L_thrust(1),  L_thrust(0),        0;

        Fu.block(6, 0, 3, 3) = this->dt * J_B_inv * skew_L;

        return Fu;
    }
};

template <typename Scalar>
class MaxInput : public StageConstraintBase<Scalar> {
private:
    Scalar umax;

public:
    MaxInput() {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 1;

        Scalar mass = 10.0;
        Scalar gravity = 9.81;
        umax = mass * gravity * 1.1;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Vector<Scalar> c_n(1);
        c_n(0) = umax - u.norm();
        return -c_n;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(1, x.size());
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, u.size());
        J.row(0) = -u.transpose() / u.norm();
        return -J;
    }
};

int main() {
    using Scalar = double;

    int N = 170;
    OptimalControlProblem<Scalar> problem(N);


    problem.setStageDynamics(std::make_shared<Rocket3D<Scalar>>());

    Scalar Q = 0.0;
    Scalar R = 1e-4;
    problem.setStageCost(std::make_shared<ScalarQuadraticStageCost<Scalar>>(Q, R));
    Scalar QT = 0.0;
    problem.setTerminalCost(std::make_shared<ScalarQuadraticTerminalCost<Scalar>>(QT));


    // Matrix<Scalar> inputcone_cx = Matrix<Scalar>::Zero(2, 6);
    // Matrix<Scalar> inputcone_cu = Matrix<Scalar>::Zero(2, 2);
    // inputcone_cu(0, 0) = - std::tan(20.0 * M_PI / 180.0);
    // inputcone_cu(1, 1) = - 1.0;
    // auto inputcone_c0 = Vector<Scalar>::Zero(2);
    // problem.addStageConstraint(std::make_shared<LinearStageConstraint<Scalar>>(
    //     inputcone_cx, inputcone_cu, inputcone_c0, ConstraintType::SOC));

    
    // Matrix<Scalar> glideslope_cx = Matrix<Scalar>::Zero(2, 6);
    // glideslope_cx(0, 0) = - std::tan(45.0 * M_PI / 180.0);
    // glideslope_cx(1, 1) = - 1.0;
    // Matrix<Scalar> glideslope_cu = Matrix<Scalar>::Zero(2, 2);
    // auto glideslope_c0 = Vector<Scalar>::Zero(2);
    // problem.addStageConstraint(std::make_shared<LinearStageConstraint<Scalar>>(
    //     glideslope_cx, glideslope_cu, glideslope_c0, ConstraintType::SOC));


    Matrix<Scalar> CT = Matrix<Scalar>::Identity(13,13);
    Vector<Scalar> c0T = Vector<Scalar>::Zero(13);
    c0T(2) = - 1.0;
    c0T(9) = - 1.0;
    problem.addTerminalConstraint(std::make_shared<LinearTerminalConstraint<Scalar>>(
        CT, c0T, ConstraintType::EQ));

    
    // problem.addStageConstraint(std::make_shared<MaxInput<Scalar>>());


    Vector<Scalar> x0 = Vector<Scalar>::Zero(13);
    x0(0) = 4.0;
    x0(1) = 6.0;
    x0(2) = 8.0;
    x0(9) = 1.0; // Quaternion
    problem.setInitialState(0, x0);

    Vector<Scalar> u0 = Vector<Scalar>::Zero(3);
    u0(2) = 9.81 * 10.0;
    problem.setInitialControl(u0);


    Param param;
    param.is_quaternion_in_state = true;
    param.quaternion_idx = 9; // Quaternion starts at index 9

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

    // std::cout << "X_result = " << std::endl;
    // for (int k = 0; k < problem.N+1; ++k) {
    //     std::cout << X_result[k].transpose() << std::endl;
    // }

    // std::cout << "U_result = " << std::endl;
    // for (int k = 0; k < problem.N; ++k) {
    //     std::cout << U_result[k].transpose() << std::endl;
    // }

    std::cout<<"X_last = \n"<<X_result[problem.getHorizon()].transpose()<<std::endl;
    std::cout<<"U_last = \n"<<U_result[problem.getHorizon() - 1].transpose()<<std::endl;

    return 0;
}