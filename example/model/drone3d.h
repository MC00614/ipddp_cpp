#include "model_base.h"

class Drone3D : public ModelBase {
public:
    Drone3D();
    ~Drone3D();

    // User Variable
    double dt;
    double gravity;
    double umax;
    Eigen::MatrixXd U;
};

Drone3D::Drone3D() {
    dt = 0.1;
    gravity = 9.81;
    umax = gravity * 1.5;

    // Stage Count
    N = 300;

    // Dimensions
    dim_x = 6;
    dim_u = 3;

    // Status Setting
    X_init = Eigen::MatrixXd::Zero(dim_x, N+1);
    X_init(0,0) = 5.0;
    X_init(1,0) = 5.0;
    X_init(2,0) = 5.0;

    U_init = Eigen::MatrixXd::Zero(dim_u, N);
    U_init.row(2) = 9.81 * Eigen::VectorXd::Ones(N);
    
    // Discrete Time System
    f = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd x_n(dim_x);
        VectorXdual2nd f_dot(dim_x);
        f_dot << 
            x(3),
            x(4),
            x(5),
            u(0),
            u(1),
            u(2) - gravity;
        x_n = x + (dt * f_dot);
        return x_n;
    };

    // Stage Cost Function
    q = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> dual2nd {
        return (2 * 1E-5 * u.squaredNorm()) + (x.topRows(2).squaredNorm() * 1e-3 * 5);
    };

    // Terminal Cost Function
    p = [this](const VectorXdual2nd& x) -> dual2nd {
        return 2000 * x.norm();
    };

    // Nonnegative Orthant Constraint Mapping
    dim_g = 3;
    g = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd g_n(3);
        Eigen::Vector3d o1;
        o1 << 1.0, 1.0, 1.0;

        g_n(0) = umax - u.norm();
        g_n(1) = -x(2);

        g_n(2) = (x.segment(0,3)-o1).squaredNorm() - 1;

        return -g_n;
    };

    // Connic Constraint Mapping
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
}

Drone3D::~Drone3D() {
}