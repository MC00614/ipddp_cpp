#include "model_base.h"

class Rocket3D : public ModelBase {
public:
    Rocket3D();
    ~Rocket3D();

    // User Variable
    double l;
    double dt;
    double mass;
    double I;
    Eigen::VectorXd gravity;
    double umax;
    Eigen::MatrixXd U;
    const double angmax = tan(20.0 * (M_PI/180.0));
};

Rocket3D::Rocket3D() {
    l = 0.7;
    dt = 0.1;
    mass = 10.0;
    I = 1.0 / 12.0 * mass * pow(2.0, 2);
    gravity.resize(3);
    gravity << 0.0, 0.0, -9.81;
    umax = mass*9.81*1.1;

    // Stage Count
    N = 300;

    // Dimensions
    dim_x = 12;
    dim_u = 3;
    dim_g = 1;
    dim_h = 3;

    // Status Setting
    X_init = Eigen::MatrixXd::Zero(dim_x, N+1);
    X_init(0,0) = 0.0;
    X_init(1,0) = 0.0;
    X_init(2,0) = 5.0;

    U_init = Eigen::MatrixXd::Zero(dim_u, N);
    U_init.row(2) = 9.81 * 10.0 * Eigen::VectorXd::Ones(N);
    
    // Discrete Time System
    f = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd x_n(dim_x);
        VectorXdual2nd f_dot(dim_x);
        
        MatrixXdual2nd R(3, 3);
        R << 
            cos(x(7)) * cos(x(8)), 
            cos(x(7)) * sin(x(8)), 
            -sin(x(7)),
            sin(x(6)) * sin(x(7)) * cos(x(8)) - cos(x(6)) * sin(x(8)), 
            sin(x(6)) * sin(x(7)) * sin(x(8)) + cos(x(6)) * cos(x(8)), 
            sin(x(6)) * cos(x(7)),
            cos(x(6)) * sin(x(7)) * cos(x(8)) + sin(x(6)) * sin(x(8)), 
            cos(x(6)) * sin(x(7)) * sin(x(8)) - sin(x(6)) * cos(x(8)), 
            cos(x(6)) * cos(x(7));

        f_dot << 
            x(3),
            x(4),
            x(5),
            -gravity + (R * u) / mass,
            x(9),
            x(10),
            x(11),
            (l / I) * u;
        x_n = x + (dt * f_dot);
        return x_n;
    };

    // Stage Cost Function
    q = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> dual2nd {
        return (2 * 1E-5 * u.squaredNorm()) + (x.topRows(3).squaredNorm() * 1e-3 * 5);
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
        VectorXdual2nd h_n(3);
        h_n(0) = angmax * u(2);
        h_n(1) = u(0);
        h_n(2) = u(1);
        return -h_n;
    };
}

Rocket3D::~Rocket3D() {
}