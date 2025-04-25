#include "quat_model_base.h"

class Rocket3D : public QuatModelBase {
public:
    Rocket3D();
    ~Rocket3D();

    // User Variable
    double r;
    double l;
    double dt;
    double mass;
    double m_wet;
    double m_dry;
    Eigen::Vector3d gravity;
    double umax;
    double umin;
    double alpha_m;
    Eigen::MatrixXd U;
    Eigen::Matrix3d J_B;
    Eigen::Matrix3d J_B_inv;
    Eigen::Vector3d L_thrust;
    Eigen::Vector4d q_desired;
    Eigen::VectorXd x_final;
    Eigen::VectorXd state_scale;
    Eigen::VectorXd input_scale;

    double r_scale;
    double m_scale;
};

Rocket3D::Rocket3D() : QuatModelBase(9) { // q_idx = 9, q_dim = 4
    dt = 0.5;
    m_wet = 30000.0;
    m_dry = 22000.0;
    mass = m_wet;
    gravity << 0.0, 0.0, -9.81;
    umax = 800000.0;
    umin = umax * 0.4;
    J_B << 4000000., 0, 0,
           0, 4000000.0, 0,
           0, 0, 100000.0;
    L_thrust << 0, 0, -14.0;
    alpha_m = 1 / (282 * 9.81);

    x_final.resize(13);
    x_final << 0.0, 0.0, 0.0,
               0.0, 0.0, -5.0, // ?
               0.0, 0.0, 0.0,
               1.0, 0.0, 0.0, 0.0;


    // Stage Count
    N = 30;

    dim_x = 13;
    X_init = Eigen::MatrixXd::Zero(dim_x, N+1);
    X_init(0,0) = 100.0;
    X_init(1,0) = 200.0;
    X_init(2,0) = 500.0;
    // X_init(3,0) = -(1.0/15.0) * X_init(0,0);
    // X_init(4,0) = -(1.0/15.0) * X_init(1,0);
    X_init(5,0) = -80.0;

    // Quaternion
    X_init(9, 0) = 1.0;
    // X_init(9, 0) = 0.7071;
    // X_init(10, 0) = 0.0;
    // X_init(11, 0) = 0.7071;
    // X_init(12, 0) = 0.0;

    dim_u = 3;
    U_init = Eigen::MatrixXd::Zero(dim_u, N);
    // U_init.row(2) = Eigen::VectorXd::Ones(N);
    U_init.row(2) = (umax + umin) / 2 * Eigen::VectorXd::Ones(N);

    // Dynamics Scale
    m_scale = m_wet;
    r_scale = X_init.topRows(3).col(0).norm();
    
    mass /= m_scale;
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
    x_final.topRows(6) /= r_scale;

    // X_nondim
    // X_init(0,0) /= m_scale; // not yet
    X_init(0,0) /= r_scale;
    X_init(1,0) /= r_scale;
    X_init(2,0) /= r_scale;
    X_init(3,0) /= r_scale;
    X_init(4,0) /= r_scale;
    X_init(5,0) /= r_scale;

    // U_nondim
    // U_init.row(0) /= (m_scale * r_scale);
    // U_init.row(1) /= (m_scale * r_scale);
    U_init.row(2) /= (m_scale * r_scale);

    // Discrete Time System
    f = [this](const VectorXdual2nd& x, const Vector3dual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd x_n = x;
        VectorXdual2nd f_dot(dim_x);

        Vector3dual2nd r = x.segment(0,3);
        Vector3dual2nd v = x.segment(3,3);
        Vector3dual2nd w = x.segment(6,3);
        Vector4dual2nd q = x.segment(9,4);

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
        
        x_n.segment(0,3) += dt * (v);
        x_n.segment(3,3) += dt * (gravity + (C * u / mass));
        x_n.segment(6,3) += dt * (J_B_inv * (L_thrust.cross(u) - w.cross(J_B * w)));
        x_n.segment(9,4) = Lq(q) * Phi(w * dt / 2);
        // x_n.segment(9,4).normalize();
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

    // // Nonnegative Orthant Constraint Mapping (Only Max)
    // dim_g = 1;
    // g = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
    //     VectorXdual2nd g_n(1);
    //     g_n(0) = umax - u.norm();
    //     return -g_n;
    // };

    // Nonnegative Orthant Constraint Mapping (MinMax)
    dim_g = 2;
    g = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd g_n(2);
        g_n(0) = umax - u.norm();
        g_n(1) = u.norm() - umin;
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
        h_n(0) = state_angmax * x(2);
        h_n(1) = x(0);
        h_n(2) = x(1);
        return -h_n;
    };
    hs.push_back(h);
    dim_hs.push_back(dim_h);

    // Terminal State Constraint (Terminal Guidance Cone)
    dim_hT = 3;
    hT = [this](const VectorXdual2nd& x) -> VectorXdual2nd {
        const double state_angmax = tan(60.0 * (M_PI/180.0));
        VectorXdual2nd h_n(3);
        h_n(0) = state_angmax * x(2);
        h_n(1) = x(0);
        h_n(2) = x(1);
        return -h_n;
    };
    hTs.push_back(hT);
    dim_hTs.push_back(dim_hT);

    // Terminal State Equality Constraint (Full State)
    dim_ecT = 13;
    ecT = [this](const VectorXdual2nd& x) -> VectorXdual2nd {
        VectorXdual2nd ecT_n(13);
        ecT_n = x - x_final;
        return ecT_n;
    };

    // // Terminal State Equality Constraint (Position Only)
    // dim_ecT = 1;
    // ecT = [this](const VectorXdual2nd& x) -> VectorXdual2nd {
    //     VectorXdual2nd ecT_n(1);
    //     ecT_n = x.row(2) - x_final.row(2);
    //     return ecT_n;
    // };
}

Rocket3D::~Rocket3D() {
}