#include "model_base.h"

class InvPend : public ModelBase {
public:
    InvPend();
    ~InvPend();
};

InvPend::InvPend() {
    // Stage Count
    N = 500;

    // Dimensions
    dim_x = 2;
    dim_u = 1;
    dim_g = 2;
    dim_h = 2;
    dim_c = dim_g + dim_h;

    // Status Setting
    X = Eigen::MatrixXd::Zero(dim_x, N+1);
    X(0,0) = -M_PI;
    X(1,0) = 0.0;

    // U = 0.02*Eigen::MatrixXd::Random(dim_u, N) - Eigen::MatrixXd::Constant(dim_u, N, 0.01);
    U = Eigen::MatrixXd::Zero(dim_u, N);
    U(0,0) = 0.0;

    Y = 0.01*Eigen::MatrixXd::Ones(dim_c, N);

    S = 0.1*Eigen::MatrixXd::Ones(dim_c, N);
    
    // Discrete Time System
    f = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        const double h = 0.05;
        VectorXdual2nd x_n(dim_x);
        x_n(0) = x(0) + h * x(1);
        x_n(1) = x(1) + h * sin(x(0)) + h * u(0);
        return x_n;
    };

    // Stage Cost Function
    q = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> dual2nd {
        return 0.025 * (x.squaredNorm() + u.squaredNorm());
    };

    // Terminal Cost Function
    p = [this](const VectorXdual2nd& x) -> dual2nd {
        return 5.0 * x.squaredNorm();
    };

    // Nonnegative Orthant Constraint Mapping
    g = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd g_n(dim_g);
        g_n(0) = u(0) - 0.25;
        g_n(1) = -u(0) - 0.25;
        return g_n;
    };

    // Connic Constraint Mapping
    h = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd h_n(dim_h);
        h_n(0) = x(0);
        h_n(1) = x(1);
        return h_n;
    };

    // Constraint Stack
    c = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd c_n(dim_c);
        c_n.topRows(dim_g) = g(x,u);
        c_n.bottomRows(dim_h) = h(x,u);
        return c_n;
    };
}

InvPend::~InvPend() {
}