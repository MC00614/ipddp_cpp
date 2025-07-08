#include "quat_model_base.h"

class Drone3D : public QuatModelBase {
public:
    Drone3D();
    ~Drone3D();

    double dt;
    double gravity;
    double mass;
    Eigen::Matrix3d J;
    Eigen::Matrix3d J_inv;

    double u_max;
    double w_max;
    Eigen::Vector4d q_desired;

    Eigen::Vector3d obs_cent;
    Eigen::MatrixXd obs_cent_zip;
    double obs_rad;
    Eigen::VectorXd obs_rad_zip;
};

Drone3D::Drone3D() : QuatModelBase(9) { // q index = 9
    dt = 0.1;
    gravity = 9.81;
    mass = 0.027;

    // Inertia
    J.setZero();
    J(0, 0) = 1.43e-5;
    J(1, 1) = 1.43e-5;
    J(2, 2) = 2.89e-5;
    J_inv = J.inverse();

    N = 100;

    // Dimensions
    dim_x = 13; // position(3), velocity(3), angular velocity(3), quaternion(4)
    dim_u = 4;  // thrust, torque_x, torque_y, torque_z

    X_init = Eigen::MatrixXd::Zero(dim_x, N + 1);
    X_init(0,0) = 5.0;
    X_init(1,0) = 5.0;
    X_init(2,0) = 10.0;
    X_init(3,0) = -0.2;
    X_init(4,0) = -0.2;
    X_init(5,0) = -0.3;
    X_init(9, 0) = 1.0;

    U_init = Eigen::MatrixXd::Zero(dim_u, N);
    U_init.row(0) = mass * gravity * Eigen::VectorXd::Ones(N); // hover

    u_max = 2.0 * mass * gravity;
    w_max = 0.5; // max angular velocity

    int obs_cnt = 3;
    int obs_cent_dim = 3;
    int obs_rad_dim = 1;
    obs_cent_zip = Eigen::MatrixXd(obs_cent_dim, obs_cnt);
    obs_rad_zip = Eigen::VectorXd(obs_cnt);

    obs_cent = Eigen::Vector3d::Zero();
    obs_rad = 0.0;

    obs_cent(0) = 5.0;
    obs_cent(1) = 4.0;
    obs_cent(2) = 6.0;
    obs_rad = 3.0;
    obs_cent_zip.col(0) = obs_cent;
    obs_rad_zip(0) = obs_rad;

    obs_cent(0) = 3.0;
    obs_cent(1) = 2.0;
    obs_cent(2) = 5.0;
    obs_rad = 2.0;
    obs_cent_zip.col(1) = obs_cent;
    obs_rad_zip(1) = obs_rad;

    obs_cent(0) = 1.0;
    obs_cent(1) = 4.0;
    obs_cent(2) = 4.0;
    obs_rad = 3.0;
    obs_cent_zip.col(2) = obs_cent;
    obs_rad_zip(2) = obs_rad;

    // Dynamics
    f = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd x_next(dim_x);

        dual2nd m = mass;
        Vector3dual2nd r = x.segment(0,3);
        Vector3dual2nd v = x.segment(3,3);
        Vector3dual2nd w = x.segment(6,3);
        Vector4dual2nd q = x.segment(9,4);

        // Rotation matrix from quaternion
        Matrix3dual2nd C;
        C << 1 - 2*(q(2)*q(2) + q(3)*q(3)), 2*(q(1)*q(2) - q(0)*q(3)),     2*(q(1)*q(3) + q(0)*q(2)),
             2*(q(1)*q(2) + q(0)*q(3)),     1 - 2*(q(1)*q(1) + q(3)*q(3)), 2*(q(2)*q(3) - q(0)*q(1)),
             2*(q(1)*q(3) - q(0)*q(2)),     2*(q(2)*q(3) + q(0)*q(1)),     1 - 2*(q(1)*q(1) + q(2)*q(2));

        Vector3dual2nd thrust_b(0, 0, u(0));
        Vector3dual2nd a = (1.0 / m) * (C * thrust_b);
        a(2) -= gravity;

        Vector3dual2nd torque = u.segment(1, 3);
        Vector3dual2nd w_dot = J_inv * (torque - w.cross(J * w));

        // Quaternion update via Omega
        Vector4dual2nd q_next = Lq(q) * Phi(w * dt / 2);
        q_next.normalize();

        x_next.segment(0,3) = r + dt * v;
        x_next.segment(3,3) = v + dt * a;
        x_next.segment(6,3) = w + dt * w_dot;
        x_next.segment(9,4) = q_next;

        return x_next;
    };

    // Cost
    q = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> dual2nd {
        return 1e-6 * u.squaredNorm();
    };

    p = [this](const VectorXdual2nd& x) -> dual2nd {
        return 0.0;
    };

    // // Constraints
    dim_g = 6;
    g = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd g_n(6);
        g_n(0) = u_max - u(0);
        g_n(1) = w_max*w_max - x.segment(6,3).squaredNorm();
        g_n(2) = 0.0 - x(5);
        g_n(3) = (x.head(3) - obs_cent_zip.col(0)).norm() - obs_rad_zip(0);
        g_n(4) = (x.head(3) - obs_cent_zip.col(1)).norm() - obs_rad_zip(1);
        g_n(5) = (x.head(3) - obs_cent_zip.col(2)).norm() - obs_rad_zip(2);
        return -g_n;
    };

    dim_h = 3;
    h = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        const double state_angmax = tan(45.0 * (M_PI/180.0));
        VectorXdual2nd h_n(3);
        h_n(0) = state_angmax * x(2);
        h_n(1) = x(0);
        h_n(2) = x(1);
        return -h_n;
    };
    hs.push_back(h);
    dim_hs.push_back(dim_h);

    dim_hT = 3;
    hT = [this](const VectorXdual2nd& x) -> VectorXdual2nd {
        const double state_angmax = tan(45.0 * (M_PI/180.0));
        VectorXdual2nd h_n(3);
        h_n(0) = state_angmax * x(2);
        h_n(1) = x(0);
        h_n(2) = x(1);
        return -h_n;
    };
    hTs.push_back(hT);
    dim_hTs.push_back(dim_hT);

    // Terminal constraint
    q_desired << 1, 0, 0, 0;
    dim_ecT = 13;
    ecT = [this](const VectorXdual2nd& x) -> VectorXdual2nd {
        VectorXdual2nd ecT_n(13);
        ecT_n(0) = x(0);
        ecT_n(1) = x(1);
        ecT_n(2) = x(2) - 1.0;
        ecT_n(3) = x(3);
        ecT_n(4) = x(4);
        ecT_n(5) = x(5);
        ecT_n(6) = x(6);
        ecT_n(7) = x(7);
        ecT_n(8) = x(8);
        VectorXdual2nd dq = Lq(x.segment(9, 4)).transpose() * q_desired;
        ecT_n(9) = 1.0 - abs(dq(0));
        ecT_n(10) = dq(1);
        ecT_n(11) = dq(2);
        ecT_n(12) = dq(3);
        return ecT_n;
    };
}

Drone3D::~Drone3D() {}
