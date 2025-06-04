#include "quat_model_base.h"

class Rocket3D : public QuatModelBase {
public:
    Rocket3D();
    ~Rocket3D();

    // User Variable
    double r;
    double l;
    double dt;
    double m_wet;
    double m_dry;
    Eigen::Vector3d gravity;
    double umax;
    double umin;
    double alpha_m;
    double w_B_max;
    double w_B_max_2;
    double tilt_max;
    double tilt_max_2;
    Eigen::MatrixXd U;
    Eigen::Matrix3d J_B;
    Eigen::Matrix3d J_B_inv;
    Eigen::Vector3d L_thrust;
    Eigen::VectorXd x_final;
    Eigen::VectorXd state_scale;
    Eigen::VectorXd input_scale;

    double r_scale;
    double m_scale;
};

Rocket3D::Rocket3D() : QuatModelBase(10) { // q_idx = 10 with mass
// Rocket3D::Rocket3D() : QuatModelBase(9) { // q_idx = 9, q_dim = 4
    dt = 0.5;
    m_wet = 30000.0;
    m_dry = 22000.0;
    gravity << 0.0, 0.0, -9.81;
    umax = 800000.0;
    umin = umax * 0.4;
    J_B << 4000000., 0, 0,
           0, 4000000.0, 0,
           0, 0, 100000.0;
    L_thrust << 0, 0, -14.0;
    alpha_m = 1.0 / (282 * 9.81);
    w_B_max = 90.0 * (M_PI/180.0); // radian
    w_B_max_2 = w_B_max * w_B_max;
    tilt_max = sqrt((1.0 - cos(70.0 * (M_PI/180.0))) / 2.0);
    tilt_max_2 = tilt_max * tilt_max;

    x_final.resize(14);
    x_final << m_dry,
               0.0, 0.0, 1.0,
               0.0, 0.0, -5.0, // ?
               0.0, 0.0, 0.0,
               1.0, 0.0, 0.0, 0.0;

    // Stage Count
    N = 30;

    dim_x = 14;
    X_init = Eigen::MatrixXd::Zero(dim_x, N+1);
    X_init(0,0) = m_wet;
    X_init(1,0) = 150.0;
    X_init(2,0) = 200.0;
    X_init(3,0) = 500.0;
    // X_init(4,0) = -(1.0/15.0) * X_init(1,0);
    // X_init(5,0) = -(1.0/15.0) * X_init(2,0);
    // X_init(6,0) = -20.0;

    // X_init(7,0) = 0.05;
    // X_init(8,0) = 0.05;
    // X_init(9,0) = 0.00;

    // Quaternion
    X_init(10, 0) = 1.0;
    // X_init(10, 0) = 0.95;
    // X_init(11, 0) = 0.1803;
    // X_init(12, 0) = 0.1803;
    // X_init(13, 0) = 0.1803;

    dim_u = 3;
    U_init = Eigen::MatrixXd::Zero(dim_u, N);
    U_init.row(2) = (umax + umin) / 2 * Eigen::VectorXd::Ones(N);

    // Dynamics Scale
    m_scale = m_wet;
    r_scale = X_init.col(0).middleRows(1,3).norm();
    
    alpha_m *= r_scale;
    L_thrust /= r_scale;
    gravity /= r_scale;
    J_B /= (m_scale * (r_scale * r_scale));
    J_B_inv = J_B.inverse();
    m_wet /= m_scale;
    m_dry /= m_scale;
    
    // Constraint Scale
    umax /= (m_scale * r_scale);
    umin /= (m_scale * r_scale);
    x_final.row(0) /= m_scale;
    x_final.middleRows(1,6) /= r_scale;

    // X_nondim
    X_init(0,0) /= m_scale;
    X_init(1,0) /= r_scale;
    X_init(2,0) /= r_scale;
    X_init(3,0) /= r_scale;
    X_init(4,0) /= r_scale;
    X_init(5,0) /= r_scale;
    X_init(6,0) /= r_scale;

    // U_nondim
    // U_init.row(0) /= (m_scale * r_scale);
    // U_init.row(1) /= (m_scale * r_scale);
    U_init.row(2) /= (m_scale * r_scale);

    // Discrete Time System
    f = [this](const VectorXdual2nd& x, const Vector3dual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd x_n = x;
        VectorXdual2nd f_dot(dim_x);

        dual2nd mass = x(0);
        Vector3dual2nd r = x.segment(1,3);
        Vector3dual2nd v = x.segment(4,3);
        Vector3dual2nd w = x.segment(7,3);
        Vector4dual2nd q = x.segment(10,4);

        Matrix3dual2nd C;
        C << 1 - 2 * (q(2) * q(2) + q(3) * q(3)),
            2 * (q(1) * q(2) - q(0) * q(3)),
            2 * (q(1) * q(3) + q(0) * q(2)),
            2 * (q(1) * q(2) + q(0) * q(3)),
            1 - 2 * (q(1) * q(1) + q(3) * q(3)),
            2 * (q(2) * q(3) - q(0) * q(1)),
            2 * (q(1) * q(3) - q(0) * q(2)),
            2 * (q(2) * q(3) + q(0) * q(1)),
            1 - 2 * (q(1) * q(1) + q(2) * q(2));

        Matrix4dual2nd Omega;
        Omega << 0,         -w(0), -w(1), -w(2),
                 w(0),   0,          w(2), -w(1),
                 w(1),  -w(2),   0,         w(0),
                 w(2),   w(1), -w(0),   0;
        
        x_n(0) += dt * (- alpha_m * u.norm());
        x_n.segment(1,3) += dt * (v);
        x_n.segment(4,3) += dt * (gravity + (C * u / mass));
        x_n.segment(7,3) += dt * (J_B_inv * (L_thrust.cross(u) - w.cross(J_B * w)));
        x_n.segment(10,4) = Lq(q) * Phi(w * dt / 2);
        return x_n;
    };

    // Stage Cost Function
    q = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> dual2nd {
        return 0.0;
        // return u.squaredNorm();
    };

    // Terminal Cost Function
    p = [this](const VectorXdual2nd& x) -> dual2nd {
        return 0.0;
    };

    // Nonnegative Orthant Constraint Mapping (ALL)
    dim_g = 5;
    g = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd g_n(5);
        g_n(0) = umax - u.norm();
        g_n(1) = u.norm() - umin;
        g_n(2) = x(0) - m_dry;
        g_n(3) = w_B_max_2 - x.segment(7,3).squaredNorm();
        g_n(4) = tilt_max_2 - x.segment(11,2).squaredNorm();
        return -g_n;
    };

    // Nonnegative Orthant Constraint Mapping (ALL, Terminal)
    dim_gT = 3;
    gT = [this](const VectorXdual2nd& x) -> VectorXdual2nd {
        VectorXdual2nd g_n(3);
        g_n(0) = x(0) - m_dry;
        g_n(1) = w_B_max_2 - x.segment(7,3).squaredNorm();
        g_n(2) = tilt_max_2 - x.segment(11,2).squaredNorm();
        return -g_n;
    };

    // Connic Constraint Mapping (Thrust Angle)
    dim_h = 3;
    h = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        const double input_angmax = tan(7.0 * (M_PI/180.0));
        VectorXdual2nd h_n(3);
        h_n(0) = input_angmax * u(2);
        h_n(1) = u(0);
        h_n(2) = u(1);
        return -h_n;
    };
    hs.push_back(h);
    dim_hs.push_back(dim_h);
    
    // Connic Constraint Mapping (Guidance Cone)
    dim_h = 3;
    h = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        const double state_angmax = tan(60.0 * (M_PI/180.0));
        VectorXdual2nd h_n(3);
        h_n(0) = state_angmax * x(3);
        h_n(1) = x(1);
        h_n(2) = x(2);
        return -h_n;
    };
    hs.push_back(h);
    dim_hs.push_back(dim_h);

    // Terminal State Constraint (Terminal Guidance Cone)
    dim_hT = 3;
    hT = [this](const VectorXdual2nd& x) -> VectorXdual2nd {
        const double state_angmax = tan(60.0 * (M_PI/180.0));
        VectorXdual2nd h_n(3);
        h_n(0) = state_angmax * x(3);
        h_n(1) = x(1);
        h_n(2) = x(2);
        return -h_n;
    };
    hTs.push_back(hT);
    dim_hTs.push_back(dim_hT);

    // // Terminal State Equality Constraint (Full State)
    // dim_ecT = 13;
    // ecT = [this](const VectorXdual2nd& x) -> VectorXdual2nd {
    //     VectorXdual2nd ecT_n(13);
    //     ecT_n = x.bottomRows(13) - x_final.bottomRows(13);
    //     return ecT_n;
    // };

    // Terminal State Equality Constraint (Full State)
    dim_ecT = 10;
    ecT = [this](const VectorXdual2nd& x) -> VectorXdual2nd {
        VectorXdual2nd ecT_n(10);
        ecT_n.segment(0,9) = x.segment(1,9) - x_final.segment(1,9);
        VectorXdual2nd dq = Lq(x.segment(10,4)).transpose() * x_final.segment(10,4);
        ecT_n(9) = 1.0 - abs(dq(0));
        return ecT_n;
    };
}

Rocket3D::~Rocket3D() {
}