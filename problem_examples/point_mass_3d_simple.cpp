#include "optimal_control_problem.h"
#include "ipddp.h"

#include <cmath>
#include <chrono>

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
        return -c_n;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(3, x.size());
        for (int i=0; i<3; i++) {
            J.block(i,0,1,3) = (x.head(3) - obs_cent_zip.col(i)).transpose() / ( (x.head(3) - obs_cent_zip.col(i)).norm() );
        }
        return -J;
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
private:
    double umax;

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
        return -c_n;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(1, x.size());
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, u.size());
        J.row(0) = -u.transpose() / u.norm();
        return -J;
    }
};

int main() {
    using Scalar = double;
    
    int N = 100;
    OptimalControlProblem<Scalar> problem(N);


    Matrix<Scalar> A = Matrix<Scalar>::Zero(6,6);
    Matrix<Scalar> B = Matrix<Scalar>::Zero(6,3);
    double dt = 0.1;
    A.block(0,0,3,3) = Matrix<Scalar>::Identity(3,3);
    A.block(0,3,3,3) = dt * Matrix<Scalar>::Identity(3,3);
    A.block(3,3,3,3) = Matrix<Scalar>::Identity(3,3);
    B.block(3,0,3,3) = dt * Matrix<Scalar>::Identity(3,3);
    auto dynamics = std::make_shared<LinearDiscreteDynamics<Scalar>>(A, B);
    Vector<Scalar> c = Vector<Scalar>::Zero(6);
    c(5) = -9.81 * dt;
    dynamics->setC(c);
    problem.setStageDynamics(dynamics);


    Matrix<Scalar> Q = Matrix<Scalar>::Zero(6,6);
    Matrix<Scalar> R = 1e-6 * Matrix<Scalar>::Identity(3,3);
    auto stage_cost = std::make_shared<QuadraticStageCost<Scalar>>(Q, R);
    problem.setStageCost(stage_cost);


    Matrix<Scalar> QT = Matrix<Scalar>::Zero(6,6);
    auto terminal_cost = std::make_shared<QuadraticTerminalCost<Scalar>>(QT);
    problem.setTerminalCost(terminal_cost);


    Matrix<double> glideslope_cx = Matrix<double>::Zero(3, 6);
    glideslope_cx(0, 2) = - std::tan(45.0 * M_PI / 180.0);
    glideslope_cx(1, 0) = - 1.0;
    glideslope_cx(2, 1) = - 1.0;
    Matrix<double> glideslope_cu = Matrix<double>::Zero(3, 3);
    auto glideslope_c0 = Vector<double>::Zero(3);
    auto glide_slope_constraint = std::make_shared<LinearStageConstraint<double>>(
        glideslope_cx, glideslope_cu, glideslope_c0, ConstraintType::SOC
    );
    problem.addStageConstraint(glide_slope_constraint);


    Matrix<double> inputcone_cx = Matrix<double>::Zero(3, 6);
    Matrix<double> inputcone_cu = Matrix<double>::Zero(3, 3);
    inputcone_cu(0, 2) = - std::tan(20.0 * M_PI / 180.0);
    inputcone_cu(1, 0) = - 1.0;
    inputcone_cu(2, 1) = - 1.0;
    auto inputcone_c0 = Vector<double>::Zero(3);
    auto input_cone_constraint = std::make_shared<LinearStageConstraint<double>>(
        inputcone_cx, inputcone_cu, inputcone_c0, ConstraintType::SOC
    );
    problem.addStageConstraint(input_cone_constraint);

    // Nonlinear
    problem.addStageConstraint(std::make_shared<Obstacle<Scalar>>());
    problem.addStageConstraint(std::make_shared<MaxInput<Scalar>>());

    Matrix<Scalar> CT = Matrix<Scalar>::Identity(6,6);
    Vector<Scalar> c0T = Vector<Scalar>::Zero(6);
    c0T(2) = -1.0;
    auto terminal_constraint = std::make_shared<LinearTerminalConstraint<Scalar>>(CT, c0T, ConstraintType::EQ);
    problem.addTerminalConstraint(terminal_constraint);

    Eigen::VectorXd x0(6);
    x0 << 5.0, 5.0, 10.0, -0.2, -0.2, -0.3;
    problem.setInitialState(0, x0);

    Eigen::VectorXd u0(3);
    u0 << 0.0, 0.0, 9.81;
    problem.setInitialControl(u0);


    Param param;
    
    clock_t start = clock();
    ALIPDDP<Scalar> solver(problem);
    solver.init(param);
    solver.solve();

    clock_t finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "\nIn Total : " << duration << " Seconds" << std::endl;

    // Parse Result
    std::vector<Eigen::VectorXd> X_result = solver.getResX();
    std::vector<Eigen::VectorXd> U_result = solver.getResU();
    std::vector<double> all_cost = solver.getAllCost();

    // std::cout << "X_result = " << std::endl;
    // for (int k = 0; k < problem.N+1; ++k) {
    //     std::cout << X_result[k].transpose() << std::endl;
    // }

    // std::cout << "U_result = " << std::endl;
    // for (int k = 0; k < problem.N; ++k) {
    //     std::cout << U_result[k].transpose() << std::endl;
    // }

    std::cout<<"X_last = \n"<<X_result[problem.getHorizon()].transpose()<<std::endl;
    std::cout<<"U_last = \n"<<U_result[problem.getHorizon() - 1].transpose()<<std::endl;

    return 0;
}