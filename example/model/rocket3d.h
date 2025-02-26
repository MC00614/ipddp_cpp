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
    Eigen::Vector3d gravity;
    double umax;
    Eigen::MatrixXd U;
    Eigen::Matrix3d J_B;
    Eigen::Matrix3d J_B_inv;
    Eigen::Vector3d L_thrust;
    // Eigen::Vector4d q_desired;
};

Rocket3D::Rocket3D() : QuatModelBase(9) { // q_idx = 9, q_dim = 4
    // q_desired << 1, 0, 0, 0;

    r = 0.4;
    l = 1.4;
    dt = 0.1;
    mass = 10.0;
    gravity << 0.0, 0.0, -9.81;
    umax = mass*9.81*1.1;
    J_B << (1.0/12.0) * mass * (3 * r * r + l * l), 0, 0,
           0, (1.0/12.0) * mass * (3 * r * r + l * l), 0,
           0, 0, 0.5 * mass * r * r;
    J_B_inv = J_B.inverse();
    L_thrust << 0, 0, -l/2;

    // Stage Count
    N = 100;

    dim_x = 13;
    X_init = Eigen::MatrixXd::Zero(dim_x, N+1);
    X_init(0,0) = 4.0;
    X_init(1,0) = 2.0;
    X_init(2,0) = 8.0;
    // X_init(3,0) = -1.0;
    // X_init(4,0) = 2.0;
    // X_init(5,0) = -1.0;

    // Quaternion
    X_init(9, 0) = 1.0;
    X_init(10, 0) = 0.0;
    X_init(11, 0) = 0.0;
    X_init(12, 0) = 0.0;

    dim_u = 3;
    U_init = Eigen::MatrixXd::Zero(dim_u, N);
    U_init.row(2) = 9.81 * 10.0 * Eigen::VectorXd::Ones(N);

    Y_init = Eigen::MatrixXd::Zero(7, N);
    Y_init.topRows(1) = 15*Eigen::MatrixXd::Ones(1, N);
    Y_init.row(1) = 30*Eigen::VectorXd::Ones(N);
    Y_init.row(4) = 5*Eigen::VectorXd::Ones(N);
    
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
        x_n.segment(9,4).normalize();
        return x_n;
    };

    // Stage Cost Function
    q = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> dual2nd {
        return (2 * 1e-3 * u.squaredNorm());
                + (50 * x.segment(3,6).squaredNorm());
        // return (2 * 1e-3 * u.squaredNorm());
    };

    // Terminal Cost Function
    p = [this](const VectorXdual2nd& x) -> dual2nd {
        return 50 * x.segment(0,3).squaredNorm()
                + (1000 * x.segment(3,6).squaredNorm());
                // + (1 * 1e-0 * x.segment(6,3).squaredNorm())
                // + (1 * 1e-0 * (Lq(q_desired).transpose() * x.segment(q_idx, q_dim)).segment(1,3).squaredNorm());
        // return 5 * 1e-1 * x.segment(0,3).squaredNorm();
    };

    // Nonnegative Orthant Constraint Mapping
    dim_g = 1;
    g = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd g_n(1);
        g_n(0) = umax - u.norm();
        return -g_n;
    };

    // Connic Constraint Mapping (Thrust Angle)
    dim_h = 3;
    h = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        const double input_angmax = tan(20.0 * (M_PI/180.0));
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
        const double state_angmax = tan(45.0 * (M_PI/180.0));
        VectorXdual2nd h_n(3);
        h_n(0) = state_angmax * x(2);
        h_n(1) = x(0);
        h_n(2) = x(1);
        return -h_n;
    };
    hs.push_back(h);
    dim_hs.push_back(dim_h);

    // Terminal State Constraint
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
}

Rocket3D::~Rocket3D() {
}