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
    Eigen::Vector4d q_desired;

    // Quaternion Setup
    int n = 13;
    int q_idx = 9;
    int q_dim = 4;

    Eigen::Matrix<double, 4, 3> H;

    Vector4dual2nd Phi(Vector3dual2nd w) {
        Vector4dual2nd phi;
        phi << 1, w;
        phi.normalize();
        // auto theta = w.norm();
        // phi(0) = cos(theta);
        // phi.segment(1,3) = sin(theta) * (w/theta);
        return phi;
    }

    Matrix4dual2nd Lq(Vector4dual2nd q) {
        Matrix4dual2nd lq;
        lq << q(0), -q(1), -q(2), -q(3),
        q(1),  q(0), -q(3),  q(2),
        q(2),  q(3),  q(0), -q(1),
        q(3), -q(2),  q(1),  q(0);
        return lq;
    };

    inline Eigen::MatrixXd GG(const VectorXdual2nd& x, const VectorXdual2nd& u) {
        return Lq(f(x,u).segment(q_idx, q_dim)).cast<double>() * H;
    }

    inline Eigen::MatrixXd G(const VectorXdual2nd& x) {
        return Lq(x.segment(q_idx, q_dim)).cast<double>() * H;
    }

    inline Eigen::MatrixXd EE(const VectorXdual2nd& x, const VectorXdual2nd& u) {
        Eigen::MatrixXd ee = Eigen::MatrixXd::Zero(n, dim_rn);
        ee.topLeftCorner(9, 9) = Eigen::MatrixXd::Identity(9, 9);
        ee.bottomRightCorner(q_dim, 3) = GG(x,u);
        return ee;
    }

    inline Eigen::MatrixXd E(const VectorXdual2nd& x) {
        Eigen::MatrixXd ee = Eigen::MatrixXd::Zero(n, dim_rn);
        ee.topLeftCorner(9, 9) = Eigen::MatrixXd::Identity(9, 9);
        ee.bottomRightCorner(q_dim, 3) = G(x);
        return ee;
    }

    inline Eigen::MatrixXd Id(const VectorXdual2nd& x, const double &fq_q) {
        Eigen::MatrixXd id = Eigen::MatrixXd::Zero(dim_rn, dim_rn);
        id.bottomRightCorner(3, 3) = fq_q * Eigen::MatrixXd::Identity(3,3);
        return id;
    }

    virtual Eigen::MatrixXd fx(VectorXdual2nd& x, VectorXdual2nd& u) override {
        return EE(x,u).transpose() * jacobian(f, wrt(x), at(x,u)) * E(x);
    }
    virtual Eigen::MatrixXd fu(VectorXdual2nd& x, VectorXdual2nd& u) override {
        return EE(x,u).transpose() * jacobian(f, wrt(u), at(x,u));
    }
    virtual Eigen::VectorXd px(VectorXdual2nd& x) override {
        return E(x).transpose() * gradient(p, wrt(x), at(x));
    }
    virtual Eigen::MatrixXd pxx(VectorXdual2nd& x) {
        Eigen::VectorXd px = gradient(p, wrt(x), at(x));
        double pqq = (px.segment(q_idx, q_dim).transpose() * x.segment(q_idx, q_dim)).cast<double>()(0);
        return E(x).transpose() * hessian(p, wrt(x), at(x)) * E(x) - Id(x, pqq);
    }
    virtual Eigen::VectorXd qx(VectorXdual2nd& x, VectorXdual2nd& u) override {
        return E(x).transpose() * gradient(q, wrt(x), at(x,u));
    }
    virtual Eigen::MatrixXd qdd(VectorXdual2nd& x, VectorXdual2nd& u) override{
        Eigen::VectorXd qx = gradient(q, wrt(x), at(x, u));
        double qqq = (qx.segment(q_idx, q_dim).transpose() * x.segment(q_idx, q_dim)).cast<double>()(0);
        Eigen::MatrixXd qdd = hessian(q, wrt(x, u), at(x, u));
        Eigen::MatrixXd quat_qdd = Eigen::MatrixXd::Zero(dim_rn + dim_u, dim_rn + dim_u);
        quat_qdd.topLeftCorner(dim_rn, dim_rn) = E(x).transpose() * qdd.topLeftCorner(dim_x, dim_x) * E(x) - Id(x, qqq);
        quat_qdd.block(0, dim_rn, dim_rn, dim_u) = E(x).transpose() * qdd.block(0, dim_x, dim_x, dim_u);
        quat_qdd.bottomRightCorner(dim_u, dim_u) = qdd.bottomRightCorner(dim_u, dim_u);
        return quat_qdd;
    }
    virtual Eigen::MatrixXd cx(VectorXdual2nd& x, VectorXdual2nd& u) override{
        return jacobian(c, wrt(x), at(x, u)) * E(x);
    }
    virtual Eigen::MatrixXd cTx(VectorXdual2nd& x) override{
        return jacobian(cT, wrt(x), at(x)) * E(x);
    }
    virtual Eigen::VectorXd perturb(const Eigen::VectorXd& xn, const Eigen::VectorXd& x) override{
        Eigen::VectorXd dx(dim_rn);
        Eigen::VectorXd q_qn = Lq(x.segment(q_idx, q_dim)).cast<double>() * xn.segment(q_idx, q_dim);
        dx << xn.segment(0,q_idx) - x.segment(0,q_idx),
            q_qn.segment(1,3)/q_qn(0);
        return dx;
    }
};

Rocket3D::Rocket3D() {
    // For Quaternion
    q_desired << 1, 0, 0, 0;
    dim_rn = 12;

    H << 0, 0, 0,
    1, 0, 0,
    0, 1, 0,
    0, 0, 1;

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
        // x_n.segment(9,4) += dt * (0.5 * Omega * q);
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
                + (1000 * x.segment(3,6).squaredNorm())
                // + (1 * 1e-0 * x.segment(6,3).squaredNorm())
                + (1 * 1e-0 * (Lq(q_desired).transpose() * x.segment(q_idx, q_dim)).segment(1,3).squaredNorm());
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