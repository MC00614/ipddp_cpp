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
    Eigen::MatrixXd U;
};

Rocket2D::Rocket2D() {
    l = 0.7;
    dt = 0.1;
    mass = 10.0;
    I = 3.33;
    gravity.resize(2);
    gravity << -9.81, 0.0;
    umax = mass*9.81*1.1;

    // Stage Count
    N = 300;

    // Dimensions
    dim_x = 6;
    dim_u = 2;
    dim_g = 1;
    dim_h = 2;

    // Status Setting
    X_init = Eigen::MatrixXd::Zero(dim_x, N+1);
    X_init(0,0) = 20.0;
    X_init(1,0) = 7.0;

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
        return (2 * 1E-5 * u.squaredNorm()) + (x.topRows(2).squaredNorm() * 1e-3 * 5);
    };

    // Terminal Cost Function
    p = [this](const VectorXdual2nd& x) -> dual2nd {
        return 2000.0 * x.norm();
    };

    // Nonnegative Orthant Constraint Mapping
    g = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd g_n(1);
        g_n(0) = u.norm() - umax;
        return g_n;
    };

    // Connic Constraint Mapping
    h = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        const double angmax = tan(20.0 * (M_PI/180.0));
        VectorXdual2nd h_n(2);
        h_n(0) = angmax * u(0);
        h_n(1) = u(1);
        return -h_n;
    };
}

Rocket2D::~Rocket2D() {
}