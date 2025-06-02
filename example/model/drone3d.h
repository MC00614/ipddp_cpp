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
    Eigen::Vector3d obs_cent;
    Eigen::MatrixXd obs_cent_zip;
    double obs_rad;
    Eigen::VectorXd obs_rad_zip;
};

Drone3D::Drone3D() {
    dt = 0.1;
    gravity = 9.81;
    umax = gravity * 1.5;

    int obs_cnt = 3;
    int obs_cent_dim = 3;
    int obs_rad_dim = 1;
    obs_cent_zip = Eigen::MatrixXd(obs_cent_dim, obs_cnt);
    obs_rad_zip = Eigen::VectorXd(obs_cnt);

    obs_cent = Eigen::Vector3d::Zero();
    obs_rad = 0.0;

    obs_cent(0) = 5.0;
    obs_cent(1) = 4.0;
    obs_cent(2) = 6.0;
    obs_rad = 3.0;
    obs_cent_zip.col(0) = obs_cent;
    obs_rad_zip(0) = obs_rad;

    obs_cent(0) = 3.0;
    obs_cent(1) = 2.0;
    obs_cent(2) = 5.0;
    obs_rad = 2.0;
    obs_cent_zip.col(1) = obs_cent;
    obs_rad_zip(1) = obs_rad;

    obs_cent(0) = 1.0;
    obs_cent(1) = 4.0;
    obs_cent(2) = 4.0;
    obs_rad = 3.0;
    obs_cent_zip.col(2) = obs_cent;
    obs_rad_zip(2) = obs_rad;

    // Stage Count
    N = 100;

    // Dimensions
    dim_x = 6;
    dim_u = 3;

    // Status Setting
    X_init = Eigen::MatrixXd::Zero(dim_x, N+1);
    X_init(0,0) = 5.0;
    X_init(1,0) = 5.0;
    X_init(2,0) = 10.0;
    X_init(3,0) = -0.2;
    X_init(4,0) = -0.2;
    X_init(5,0) = -0.3;

    U_init = Eigen::MatrixXd::Zero(dim_u, N);
    U_init.row(0) = 0.01 * Eigen::VectorXd::Random(N);
    U_init.row(1) = 0.01 * Eigen::VectorXd::Random(N);
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
        return 1e-4 * u.squaredNorm();
    };

    // Terminal Cost Function
    p = [this](const VectorXdual2nd& x) -> dual2nd {
        return 0.0;
    };

    // Nonnegative Orthant Constraint Mapping
    dim_g = 5;
    g = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd g_n(5);
        g_n(0) = umax - u.norm();
        g_n(1) = -x(5) - 0.0;
        g_n(2) = (x.head(3) - obs_cent_zip.col(0)).norm() - obs_rad_zip(0);
        g_n(3) = (x.head(3) - obs_cent_zip.col(1)).norm() - obs_rad_zip(1);
        g_n(4) = (x.head(3) - obs_cent_zip.col(2)).norm() - obs_rad_zip(2);
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

    // Connic Constraint Mapping
    dim_h = 3;
    h = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        const double glideslope_angmax = tan(45.0 * (M_PI/180.0));
        VectorXdual2nd h_n(3);
        h_n(0) = glideslope_angmax * x(2);
        h_n(1) = x(0);
        h_n(2) = x(1);
        return -h_n;
    };
    hs.push_back(h);
    dim_hs.push_back(dim_h);

    // Connic Constraint Mapping
    dim_hT = 3;
    hT = [this](const VectorXdual2nd& x) -> VectorXdual2nd {
        const double glideslope_angmax = tan(45.0 * (M_PI/180.0));
        VectorXdual2nd h_n(3);
        h_n(0) = glideslope_angmax * x(2);
        h_n(1) = x(0);
        h_n(2) = x(1);
        return -h_n;
    };
    hTs.push_back(hT);
    dim_hTs.push_back(dim_hT);

    // Terminal State Equality Constraint (Full State)
    dim_ecT = 6;
    ecT = [this](const VectorXdual2nd& x) -> VectorXdual2nd {
        VectorXdual2nd ecT_n(6);
        ecT_n(0) = x(0);
        ecT_n(1) = x(1);
        ecT_n(2) = x(2) - 1.0;
        ecT_n(3) = x(3);
        ecT_n(4) = x(4);
        ecT_n(5) = x(5);
        return ecT_n;
    };
}

Drone3D::~Drone3D() {
}