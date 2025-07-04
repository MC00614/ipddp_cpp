#include "optimal_control_problem.h"

// Dynamics
template <typename Scalar>
class Dynamics : public DiscreteDynamicsBase<Scalar> {
public:
    Dynamics() {
        this->dim_x = 6;
        this->dim_u = 3;
        this->dt = 0.1;
    }
    Vector<Scalar> f(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Vector<Scalar> xnext(6);
        auto pos = x.segment(0,3);
        auto vel = x.segment(3,3);
        Vector<Scalar> acc = u;
        acc(2) -= 9.81;
        auto pos_next = pos + this->dt * vel;
        auto vel_next = vel + this->dt * acc;
        xnext << pos_next, vel_next;
        return xnext;
    }

    Matrix<Scalar> fx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> Fx = Matrix<Scalar>::Zero(6,6);
        Fx.block(0,0,3,3) = Matrix<Scalar>::Identity(3,3);
        Fx.block(0,3,3,3) = this->dt * Matrix<Scalar>::Identity(3,3);
        Fx.block(3,3,3,3) = Matrix<Scalar>::Identity(3,3);
        return Fx;
    }

    Matrix<Scalar> fu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> Fu = Matrix<Scalar>::Zero(6,3);
        Fu.block(3,0,3,3) = this->dt * Matrix<Scalar>::Identity(3,3);
        return Fu;
    }
};

// Cost
template <typename Scalar>
class StageCost : public StageCostBase<Scalar> {
public:
    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return 1e-6 * u.squaredNorm();
    }

    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return Vector<Scalar>::Zero(x.size());
    }

    Vector<Scalar> qu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return 2e-6 * u;
    }

    Matrix<Scalar> qxx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(x.size(), x.size());
    }

    Matrix<Scalar> quu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return 2e-6 * Matrix<Scalar>::Identity(u.size(), u.size());
    }

    Matrix<Scalar> qxu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(x.size(), u.size());
    }
};

template <typename Scalar>
class TerminalCost : public TerminalCostBase<Scalar> {
public:
    Scalar p(const Vector<Scalar>& x) const override {
        return 0.0;
    }

    Vector<Scalar> px(const Vector<Scalar>& x) const override {
        return Vector<Scalar>::Zero(x.size());
    }

    Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
        return Matrix<Scalar>::Zero(x.size(), x.size());
    }
};

// Costraint
template <typename Scalar>
class Obstacle : public StageConstraintBase<Scalar> {
public:
    Obstacle() {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 3;

        obs_cent_dim = 3;
        obs_cnt = 3;
        obs_cent_zip = Eigen::MatrixXd(obs_cent_dim, obs_cnt);
        obs_rad_zip = Eigen::VectorXd(obs_cnt);

        // hard-coded example obstacles
        obs_cent_zip.col(0) = Eigen::Vector3d(5.0,4.0,6.0);
        obs_rad_zip(0) = 3.0;

        obs_cent_zip.col(1) = Eigen::Vector3d(3.0,2.0,5.0);
        obs_rad_zip(1) = 2.0;

        obs_cent_zip.col(2) = Eigen::Vector3d(1.0,4.0,4.0);
        obs_rad_zip(2) = 3.0;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Vector<Scalar> c_n(3);
        c_n(0) = (x.head(3) - obs_cent_zip.col(0)).norm() - obs_rad_zip(0);
        c_n(1) = (x.head(3) - obs_cent_zip.col(1)).norm() - obs_rad_zip(1);
        c_n(2) = (x.head(3) - obs_cent_zip.col(2)).norm() - obs_rad_zip(2);
        return c_n;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(3, x.size());
        for (int i=0; i<3; i++) {
            J.block(i,0,1,3) = (x.head(3) - obs_cent_zip.col(i)).transpose() / ( (x.head(3) - obs_cent_zip.col(i)).norm() );
        }
        return J;
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(3, u.size());
    }

private:
    int obs_cnt;
    int obs_cent_dim;
    Eigen::MatrixXd obs_cent_zip;
    Eigen::VectorXd obs_rad_zip;
};

template <typename Scalar>
class MaxInput : public StageConstraintBase<Scalar> {
public:
    MaxInput() {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 1;

        double gravity = 9.81;
        umax = gravity * 1.5;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Vector<Scalar> c_n(1);
        c_n(0) = umax - u.norm();
        return c_n;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(1, x.size());
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, u.size());
        J.row(0) = -u.transpose() / u.norm();
        return J;
    }

private:
    double umax;
};

template <typename Scalar>
class InputConeConstraint : public StageConstraintBase<Scalar> {
public:
    InputConeConstraint() {
        this->constraint_type = ConstraintType::SOC;
        this->dim_c = 3;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        const double input_angmax = std::tan(20.0 * M_PI / 180.0);
        Vector<Scalar> c_n(3);
        c_n(0) = input_angmax * u(2);
        c_n(1) = u(0);
        c_n(2) = u(1);
        return -c_n;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(3, x.size());
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(3, u.size());
        const double input_angmax = std::tan(20.0 * M_PI / 180.0);
        J(0,2) = input_angmax;
        J(1,0) = 1.0;
        J(2,1) = 1.0;
        return -J;
    }
};

template <typename Scalar>
class GlideSlopeConeConstraint : public StageConstraintBase<Scalar> {
public:
    GlideSlopeConeConstraint() {
        this->constraint_type = ConstraintType::SOC;
        this->dim_c = 3;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        const double glideslope_angmax = std::tan(45.0 * M_PI / 180.0);
        Vector<Scalar> c_n(3);
        c_n(0) = glideslope_angmax * x(2);
        c_n(1) = x(0);
        c_n(2) = x(1);
        return -c_n;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(3, x.size());
        J(0, 2) = -std::tan(45.0 * M_PI / 180.0);
        J(1, 0) = -1.0;
        J(2, 1) = -1.0;
        return J;
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(3, u.size());
    }
};

template <typename Scalar>
class TerminalEqualityConstraint : public TerminalConstraintBase<Scalar> {
public:
    TerminalEqualityConstraint() {
        this->constraint_type = ConstraintType::EQ;
        this->dim_cT = 6;
    }

    Vector<Scalar> cT(const Vector<Scalar>& x) const override {
        Vector<Scalar> cT_n(6);
        cT_n(0) = x(0);
        cT_n(1) = x(1);
        cT_n(2) = x(2) - 1.0;
        cT_n(3) = x(3);
        cT_n(4) = x(4);
        cT_n(5) = x(5);
        return cT_n;
    }

    Matrix<Scalar> cTx(const Vector<Scalar>& x) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(6, x.size());
        J(0,0) = 1.0;
        J(1,1) = 1.0;
        J(2,2) = 1.0;
        J(3,3) = 1.0;
        J(4,4) = 1.0;
        J(5,5) = 1.0;
        return J;
    }
};

int main() {
    using Scalar = double;

    int N = 100;
    Vector<Scalar> x0(6);
    x0(0) = 5.0;
    x0(1) = 5.0;
    x0(2) = 10.0;
    x0(3) = -0.2;
    x0(4) = -0.2;
    x0(5) = -0.3;
    x0.setZero();
    OptimalControlProblem<Scalar> problem(N, x0);

    for (int i = 0; i < N; ++i) {
        problem.setStageDynamics(i, std::make_shared<Dynamics<Scalar>>());
    }

    for (int i = 0; i < N; ++i) {
        problem.setStageCost(i, std::make_shared<StageCost<Scalar>>());
    }

    problem.setTerminalCost(std::make_shared<TerminalCost<Scalar>>());

    for (int i = 0; i < N; ++i) {
        problem.addStageConstraint(i, std::make_shared<Obstacle<Scalar>>());
        problem.addStageConstraint(i, std::make_shared<MaxInput<Scalar>>());
        problem.addStageConstraint(i, std::make_shared<InputConeConstraint<Scalar>>());
        problem.addStageConstraint(i, std::make_shared<GlideSlopeConeConstraint<Scalar>>());
    }

    problem.addTerminalConstraint(std::make_shared<TerminalEqualityConstraint<Scalar>>());

    problem.init();

    

    return 0;
}