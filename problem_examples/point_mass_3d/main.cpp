#include "optimal_control_problem.h"

// Dynamics
template <typename Scalar>
class Dynamics : public DiscreteDynamics<Dynamics<Scalar>, Scalar> {
public:
    Dynamics() {
        this->dim_x = 6;
        this->dim_u = 3;
        this->dt = 0.1;
    }

    template <typename Vector_Dyn>
    Vector_Dyn f(const Vector_Dyn& x, const Vector_Dyn& u) const {
        Vector_Dyn xnext(6);
        auto pos = x.segment(0,3);
        auto vel = x.segment(3,3);

        Vector_Dyn acc = u;
        acc(2) -= 9.81;

        auto pos_next = pos + this->dt * vel;
        auto vel_next = vel + this->dt * acc;

        xnext << pos_next, vel_next;
        return xnext;
    }

    Matrix<Scalar> fx(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        Matrix<Scalar> Fx = Matrix<Scalar>::Zero(6,6);
        Fx.block(0,0,3,3) = Matrix<Scalar>::Identity(3,3);
        Fx.block(0,3,3,3) = this->dt * Matrix<Scalar>::Identity(3,3);
        Fx.block(3,3,3,3) = Matrix<Scalar>::Identity(3,3);
        return Fx;
    }

    Matrix<Scalar> fu(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        Matrix<Scalar> Fu = Matrix<Scalar>::Zero(6,3);
        Fu.block(3,0,3,3) = this->dt * Matrix<Scalar>::Identity(3,3);
        return Fu;
    }
};

// Cost
template <typename Scalar>
class StageCost : public StageCostFunction<StageCost<Scalar>, Scalar> {
public:
    StageCost();

    template <typename Vector_Cost, typename Vector_Dyn>
    Vector_Cost q(const Vector_Dyn& x, const Vector_Dyn& u) const {
        return 1e-6 * u.squaredNorm();
    }
};

template <typename Scalar>
class TerminalCost : public TerminalCostFunction<TerminalCost<Scalar>, Scalar> {
public:
    TerminalCost();

    template <typename Vector_Cost, typename Vector_Dyn>
    Vector_Cost p(const Vector_Dyn& x) const {
        return 0.0;
    }
};

// Costraint


template <typename Scalar>
class Obstacle : public StageConstraintFunction<Obstacle<Scalar>, Scalar> {
public:
    Eigen::Vector3d obs_cent;
    Eigen::MatrixXd obs_cent_zip;
    double obs_rad;
    Eigen::VectorXd obs_rad_zip;

    Obstacle() {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 3;

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
    }

    template <typename Vector_Cst, typename Vector_Dyn>
    Vector_Cst c(const Vector_Dyn& x, const Vector_Dyn& u) const {
        Vector_Cst c_n(3);
        c_n(0) = (x.head(3) - obs_cent_zip.col(0)).norm() - obs_rad_zip(0);
        c_n(1) = (x.head(3) - obs_cent_zip.col(1)).norm() - obs_rad_zip(1);
        c_n(2) = (x.head(3) - obs_cent_zip.col(2)).norm() - obs_rad_zip(2);
        return c_n;
    }
};

template <typename Scalar>
class MaxInput : public StageConstraintFunction<MaxInput<Scalar>, Scalar> {
public:
    double umax;

    MaxInput() {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 1;

        double gravity = 9.81;
        umax = gravity * 1.5;
    }
    template <typename Vector_Cst, typename Vector_Dyn>
    Vector_Cst c(const Vector_Dyn& x, const Vector_Dyn& u) const {
        Vector_Cst c_n(1);
        c_n(0) = umax - u.norm();
        return c_n;
    }
};

template <typename Scalar>
class InputConeConstraint : public StageConstraintFunction<InputConeConstraint<Scalar>, Scalar> {
public:
    InputConeConstraint() {
        this->constraint_type = ConstraintType::SOC;
        this->dim_c = 3;
    }

    template <typename Vector_Cst, typename Vector_Dyn>
    Vector_Cst c(const Vector_Dyn& x, const Vector_Dyn& u) const {
        const double input_angmax = tan(20.0 * M_PI / 180.0);
        Vector_Cst c_n(3);
        c_n(0) = input_angmax * u(2);
        c_n(1) = u(0);
        c_n(2) = u(1);
        return -c_n;
    }
};

template <typename Scalar>
class GlideSlopeConeConstraint : public StageConstraintFunction<GlideSlopeConeConstraint<Scalar>, Scalar> {
public:
    GlideSlopeConeConstraint() {
        this->constraint_type = ConstraintType::SOC;
        this->dim_c = 3;
    }

    template <typename Vector_Cst, typename Vector_Dyn>
    Vector_Cst c(const Vector_Dyn& x, const Vector_Dyn& u) const {
        const double glideslope_angmax = tan(45.0 * M_PI / 180.0);
        Vector_Cst c_n(3);
        c_n(0) = glideslope_angmax * x(2);
        c_n(1) = x(0);
        c_n(2) = x(1);
        return -c_n;
    }
};

template <typename Scalar>
class TerminalEqualityConstraint : public TerminalConstraintFunction<TerminalEqualityConstraint<Scalar>, Scalar> {
public:
    TerminalEqualityConstraint() {
        this->constraint_type = ConstraintType::EQ;
        this->dim_cT = 6;
    }

    template <typename Vector_Cst, typename Vector_Dyn>
    Vector_Cst cT(const Vector_Dyn& x) const {
        Vector_Cst cT_n(6);
        cT_n(0) = x(0);
        cT_n(1) = x(1);
        cT_n(2) = x(2) - 1.0;
        cT_n(3) = x(3);
        cT_n(4) = x(4);
        cT_n(5) = x(5);
        return cT_n;
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
        auto dyn = std::make_shared<Dynamics<Scalar>>();
        problem.dynamics_seq[i] = dyn;
    }

    for (int i = 0; i < N; ++i) {
        problem.setStageCost(i, std::make_shared<StageCost<Scalar>>());
    }

    auto costT = std::make_shared<TerminalCost<Scalar>>();
    problem.setTerminalCost(costT);

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