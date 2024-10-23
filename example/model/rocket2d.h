#include "model_base.h"

class Rocket2D : public ModelBase {
public:
    Rocket2D();
    ~Rocket2D();
};

Rocket2D::Rocket2D() {
    // Stage Count
    N = 300;

    // Dimensions
    dim_x = 6;
    dim_u = 2;
    dim_g = 1;
    dim_h = 2;
    dim_c = dim_g + dim_h;

    // Status Setting
    X = Eigen::MatrixXd::Zero(dim_x, N+1);
    // X(0,0) = 10.0;
    // X(1,0) = 5.0;
    X(0,0) = 20.0;
    X(1,0) = 7.0;

    U = Eigen::MatrixXd::Zero(dim_u, N);
    U.row(0) = 9.81 * 10.0 * Eigen::VectorXd::Ones(N);

    S = 0.01 * Eigen::MatrixXd::Ones(dim_c, N);
    Y = 0.01 * Eigen::MatrixXd::Ones(dim_c, N);
    Eigen::VectorXd y_init(dim_c);
    Eigen::VectorXd s_init(dim_c);
    y_init << 0.01, 0.001, 0.0;
    s_init << 0.01, 0.001, 0.0;
    Y.colwise() = y_init;
    S.colwise() = s_init;
    
    // Discrete Time System
    f = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        const double l = 0.7;
        const double dt = 0.1;
        const double mass = 10.0;
        const double I = 1.0 / 12.0 * mass * pow(2.0, 2);

        VectorXdual2nd x_n(dim_x);
        x_n(0) = x(0) + dt * x(2);
        x_n(1) = x(1) + dt * x(3);
        x_n(2) = x(2) + dt * (-9.81 + (cos(x(4)) * u(0) - sin(x(4)) * u(1)) / mass);
        x_n(3) = x(3) + dt * ((sin(x(4)) * u(0) + cos(x(4)) * u(1)) / mass);
        x_n(4) = x(4) + dt * x(5);
        x_n(5) = x(5) + dt * (-(l / I) * u(1));
        return x_n;
    };

    // Stage Cost Function
    q = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> dual2nd {
        return (2 * 1E-5 * u.squaredNorm()) + ((pow(x(0), 2)+pow(x(1), 2)) * 1e-3 * 5);
    };

    // Terminal Cost Function
    p = [this](const VectorXdual2nd& x) -> dual2nd {
        return 2000.0 * x.norm();
    };

    // Nonnegative Orthant Constraint Mapping
    g = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        const double g = 9.81;
        const double mass = 10.0;
        const double umax = mass*g*1.1;
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
    
    // Constraint Stack
    c = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd c_n(dim_c);
        if (dim_g) {c_n.topRows(dim_g) = g(x,u);}
        if (dim_h) {c_n.bottomRows(dim_h) = h(x,u);}
        return c_n;
    };
}

Rocket2D::~Rocket2D() {
}