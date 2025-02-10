#include "model_base.h"

class Rocket3D : public ModelBase {
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
};

Rocket3D::Rocket3D() {
    r = 0.4;
    l = 0.7;
    dt = 0.1;
    mass = 10.0;
    gravity << 0.0, 0.0, -9.81;
    umax = mass*9.81*1.1;
    J_B << (1.0/12.0) * mass * (3 * r * r + l * l), 0, 0,
           0, (1.0/12.0) * mass * (3 * r * r + l * l), 0,
           0, 0, 0.5 * mass * r * r;
    J_B_inv = J_B.inverse();
    L_thrust << 0, 0, -l;

    // Stage Count
    N = 100;

    dim_x = 13;
    X_init = Eigen::MatrixXd::Zero(dim_x, N+1);
    X_init(0,0) = 4.0;
    X_init(1,0) = 6.0;
    X_init(2,0) = 8.0;

    // Quaternion
    X_init(6, 0) = 1.0;
    X_init(7, 0) = 0.0;
    X_init(8, 0) = 0.0;
    X_init(9, 0) = 0.0;

    dim_u = 3;
    U_init = Eigen::MatrixXd::Zero(dim_u, N);
    U_init.row(2) = 9.81 * 10.0 * Eigen::VectorXd::Ones(N);
    
    // Discrete Time System
    f = [this](const VectorXdual2nd& x, const Vector3dual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd x_n(dim_x);
        VectorXdual2nd f_dot(dim_x);
        Matrix3dual2nd C;
        C << 1 - 2 * (x(8) * x(8) + x(9) * x(9)),
            2 * (x(7) * x(8) - x(6) * x(9)),
            2 * (x(7) * x(9) + x(6) * x(8)),
            2 * (x(7) * x(8) + x(6) * x(9)),
            1 - 2 * (x(7) * x(7) + x(9) * x(9)),
            2 * (x(8) * x(9) - x(6) * x(7)),
            2 * (x(7) * x(9) - x(6) * x(8)),
            2 * (x(8) * x(9) + x(6) * x(7)),
            1 - 2 * (x(7) * x(7) + x(8) * x(8));

        Vector3dual2nd w = x.segment(10,3);
        Matrix4dual2nd Omega;
        Omega << 0,         -w(0), -w(1), -w(2),
                w(0),   0,          w(2), -w(1),
                w(1),  -w(2),   0,         w(0),
                w(2),   w(1),  -w(0),  0;
        f_dot << 
            x.segment(3, 3),
            gravity + (C * u / mass),
            0.5 * (Omega * x.segment(6,4)),
            J_B_inv * (L_thrust.cross(u) - w.cross(J_B * w));
        x_n = x + (dt * f_dot);
        // std::cout<<x_n.segment(6,4).normalize();
        return x_n;
    };

    // Stage Cost Function
    q = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> dual2nd {
        return (2 * 1e-3 * u.norm()) + (5 * 1e-3 * x.segment(0,6).norm());
    };

    // Terminal Cost Function
    p = [this](const VectorXdual2nd& x) -> dual2nd {
        // return 0;
        return x.segment(0,6).norm();
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

    // TODO!
    // Terminal State Constraint
}

Rocket3D::~Rocket3D() {
}