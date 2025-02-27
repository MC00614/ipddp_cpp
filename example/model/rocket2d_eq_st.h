#include "model_base.h"

class Rocket2D : public ModelBase {
public:
    Rocket2D();
    ~Rocket2D();

    // User Variable
    double l;
    double dt;
    double mass;
    double I;
    Eigen::VectorXd gravity;
    double umax;
    double umin;
    Eigen::MatrixXd U;
};

Rocket2D::Rocket2D() {
    l = 0.7;
    dt = 0.035;
    mass = 10.0;
    I = 3.33;
    gravity.resize(2);
    gravity << -9.81, 0.0;
    umax = mass*9.81*1.1;
    umin = mass*9.81*0.5;

    // Stage Count
    N = 300;

    dim_x = 6;
    X_init = Eigen::MatrixXd::Zero(dim_x, N+1);
    X_init(0,0) = 10.0;
    X_init(1,0) = 6.0;

    dim_u = 2;
    U_init = Eigen::MatrixXd::Zero(dim_u, N);
    U_init.row(0) = 9.81 * 10.0 * Eigen::VectorXd::Ones(N);
    
    // Discrete Time System
    f = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd x_n(dim_x);
        VectorXdual2nd f_dot(dim_x);
        MatrixXdual2nd U(2,2);
        U << cos(x(4)), -sin(x(4)),
            sin(x(4)),  cos(x(4));
        f_dot << 
            x(2),
            x(3),
            gravity + U * u / mass,
            x(5),
            -(l / I) * u(1);
        x_n = x + (dt * f_dot);
        return x_n;
    };

    // Stage Cost Function
    q = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> dual2nd {
        // return u.squaredNorm();
        return 1e-6 * u.squaredNorm() + 1e-4 * x.segment(2,4).squaredNorm();
    };

    // Terminal Cost Function
    p = [this](const VectorXdual2nd& x) -> dual2nd {
        return 10 * x.segment(2,4).squaredNorm();
    };

    // Nonnegative Orthant Constraint Mapping
    dim_g = 2;
    g = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd g_n(2);
        g_n(0) = umax - u.norm();
        g_n(1) = u.norm() - umin;
        return -g_n;
    };

    // Connic Constraint Mapping (Thrust Angle)
    dim_h = 2;
    h = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        const double input_angmax = tan(20.0 * (M_PI/180.0));
        VectorXdual2nd h_n(2);
        h_n(0) = input_angmax * u(0);
        h_n(1) = u(1);
        return -h_n;
    };
    hs.push_back(h);
    dim_hs.push_back(dim_h);
    
    // Connic Constraint Mapping (Guidance Cone)
    dim_h = 2;
    h = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        const double state_angmax = tan(45.0 * (M_PI/180.0));
        VectorXdual2nd h_n(2);
        h_n(0) = state_angmax * x(0);
        h_n(1) = x(1);
        return -h_n;
    };
    hs.push_back(h);
    dim_hs.push_back(dim_h);

    // Terminal State Conic Constraint
    dim_hT = 2;
    hT = [this](const VectorXdual2nd& x) -> VectorXdual2nd {
        const double state_angmax = tan(45.0 * (M_PI/180.0));
        VectorXdual2nd h_n(2);
        h_n(0) = state_angmax * x(0);
        h_n(1) = x(1);
        return -h_n;
    };
    hTs.push_back(hT);
    dim_hTs.push_back(dim_hT);

    // Terminal State Equality Constraint
    dim_ecT = 2;
    // dim_ecT = 6;
    ecT = [this](const VectorXdual2nd& x) -> VectorXdual2nd {
        VectorXdual2nd ecT_n(2);
        // VectorXdual2nd ecT_n(6);
        ecT_n(0) = x(0) - 1.0;
        ecT_n(1) = x(1);
        // ecT_n(2) = x(2);
        // ecT_n(3) = x(3);
        // ecT_n(4) = x(4);
        // ecT_n(5) = x(5);
        return ecT_n;
    };
}

Rocket2D::~Rocket2D() {
}