#include <eigen3/Eigen/Dense>

class InvPend {
public:
    InvPend();
    ~InvPend();

    int N;
    int dim_x;
    int dim_u;
    Eigen::MatrixXd X;
    Eigen::MatrixXd U;
    // Discrete Time System
    std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> f;
    // Stage Cost Function
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> q;
    // Terminal Cost Function
    std::function<double(Eigen::VectorXd)> p;
};

InvPend::InvPend() {
    // Stage Count
    N = 500;

    // Dimensions
    dim_x = 2;
    dim_u = 1;

    // Status Setting
    X = Eigen::MatrixXd::Zero(dim_x, N+1);
    X(0,0) = -M_PI;
    X(1,0) = 0.0;
    X(0,N) = 0,0;
    X(1,N) = 0.0;
    U = Eigen::MatrixXd::Zero(dim_u, N);
    U(0,0) = 0.0;

    // Discrete Time System
    f = [this](const Eigen::VectorXd& x0, const Eigen::VectorXd& u) -> Eigen::VectorXd {
        const double h = 0.05;
        Eigen::VectorXd x1(x0.size());
        x1(0) = x0(0) + h * x0(1);
        x1(1) = x0(1) + h * std::sin(x0(0)) + h * u(0);
        return x1;
    };

    // Stage Cost Function
    q = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> double {
        return 0.025 * (x.squaredNorm() + u.squaredNorm());
    };

    // Terminal Cost Function
    p = [this](const Eigen::VectorXd& x) -> double {
        return 5.0 * x.squaredNorm();
    };
}

InvPend::~InvPend() {
}