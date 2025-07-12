#pragma once

#include "optimal_control_problem.h"
#include "problem_parser.h"
#include "param.h"

#include "helper_functions/no_helper.h"
#include "helper_functions/soc_helper.h"
#include "helper_functions/unit_quaternion_helper.h"

#include <Eigen/Dense>

#include <functional>
#include <cmath>
#include <memory>
#include <iostream>
#include <iomanip>

template <typename Scalar>
class ALIPDDP {
public:
    explicit ALIPDDP(OptimalControlProblem<Scalar>& ocp_ref);
    ~ALIPDDP();

    void init(Param param);
    void solve();

    std::vector<Eigen::VectorXd> getResX();
    std::vector<Eigen::VectorXd> getResU();
    std::vector<double> getAllCost();

private:
    ProblemParser<Scalar> ocp;

    const int N;
    std::vector<Eigen::VectorXd> X; // State
    std::vector<Eigen::VectorXd> U; // Input
    
    std::vector<Eigen::VectorXd> Z; // Equality Lagrangian Multiplier
    std::vector<Eigen::VectorXd> R; // Equality Slack
    std::vector<Eigen::VectorXd> Y; // Inequality Lagrangian Multiplier
    std::vector<Eigen::VectorXd> S; // Inequality Slack
    std::vector<Eigen::VectorXd> C; // Inequality Constraint
    std::vector<Eigen::VectorXd> EC; // Equality Constraint

    Eigen::VectorXd ZT; // Equality Lagrangian Multiplier (Terminal)
    Eigen::VectorXd RT; // Equality Slack (Terminal)
    Eigen::VectorXd YT; // Inequality Lagrangian Multiplier (Terminal)
    Eigen::VectorXd ST; // Inequality Slack (Terminal)
    Eigen::VectorXd CT; // Inequality Constraint (Terminal)
    Eigen::VectorXd ECT; // Equality Constraint (Terminal)

    std::vector<bool> is_c_active;
    std::vector<bool> is_ec_active;
    bool is_c_active_all;
    bool is_ec_active_all;
    bool is_cT_active;
    bool is_ecT_active;

    Eigen::VectorXd scale_ec;
    Eigen::VectorXd scale_c;
    Eigen::VectorXd scale_ecT;
    Eigen::VectorXd scale_cT;

    std::vector<Eigen::VectorXd> lambda;
    Eigen::VectorXd lambdaT;
    std::vector<int> dim_rn;
    int dim_rnT;
    Eigen::VectorXd perturb(const int& k, const Eigen::VectorXd& xn, const Eigen::VectorXd& x);
    
    double cost;
    Param param;
    void initialRoll();
    void resetCost();
    void resetError();
    double logcost;
    double error;
    
    std::vector<double> step_list; // Step Size List
    int step; // Step Size Index
    int forward_failed;
    bool is_traj_moved;

    int update_counter;
    int iter;
    int inner_iter;
    void resetRegulation();
    int regulate;
    bool backward_failed;

    std::vector<Eigen::MatrixXd> fx_all;
    std::vector<Eigen::MatrixXd> fu_all;
    // std::vector<std::vector<Eigen::MatrixXd>> fxx_all;
    // std::vector<std::vector<Eigen::MatrixXd>> fxu_all;
    // std::vector<std::vector<Eigen::MatrixXd>> fuu_all;
    Eigen::VectorXd px_all;
    Eigen::MatrixXd pxx_all;
    std::vector<Eigen::VectorXd> qx_all;
    std::vector<Eigen::VectorXd> qu_all;
    std::vector<Eigen::MatrixXd> qxx_all;
    std::vector<Eigen::MatrixXd> qxu_all;
    std::vector<Eigen::MatrixXd> quu_all;
    std::vector<Eigen::MatrixXd> cx_all;
    std::vector<Eigen::MatrixXd> cu_all;
    std::vector<Eigen::MatrixXd> ecx_all;
    std::vector<Eigen::MatrixXd> ecu_all;
    Eigen::MatrixXd cTx_all;
    Eigen::MatrixXd ecTx_all;

    std::vector<Eigen::VectorXd> du; // Input Feedforward Gain 
    std::vector<Eigen::VectorXd> dr; // Equality Slack Feedforward Gain
    std::vector<Eigen::VectorXd> dz; // Equality Lagrangian Multiplier Feedforward Gain
    std::vector<Eigen::VectorXd> ds; // Inequality Slack Feedforward Gain
    std::vector<Eigen::VectorXd> dy; // Inequality Lagrangian Multiplier Feedforward Gain
    
    std::vector<Eigen::MatrixXd> Ku; // Input Feedback Gain
    std::vector<Eigen::MatrixXd> Kr; // Equality Slack Feedback Gain
    std::vector<Eigen::MatrixXd> Kz; // Equality Lagrangian Multiplier Feedback Gain
    std::vector<Eigen::MatrixXd> Ks; // Inequality Slack Feedback Gain
    std::vector<Eigen::MatrixXd> Ky; // Inequality Lagrangian Multiplier Feedback Gain

    Eigen::VectorXd drT; // Equality Slack Feedforward Gain (Terminal)
    Eigen::VectorXd dzT; // Equality Lagrangian Multiplier Feedforward Gain (Terminal)
    Eigen::VectorXd dsT; // Inequality Slack Feedforward Gain (Terminal)
    Eigen::VectorXd dyT; // Inequality Lagrangian Multiplier Feedforward Gain (Terminal)

    Eigen::MatrixXd KrT; // Equality Slack Feedback Gain (Terminal)
    Eigen::MatrixXd KzT; // Equality Lagrangian Multiplier Feedback Gain (Terminal)
    Eigen::MatrixXd KsT; // Inequality Slack Feedback Gain (Terminal)
    Eigen::MatrixXd KyT; // Inequality Lagrangian Multiplier Feedback Gain (Terminal)

    std::vector<Eigen::VectorXd> e;
    Eigen::VectorXd eT;

    bool is_alm_updated;
    bool is_alm_updatedT;
    bool is_ipm_updated;
    bool is_ipm_updatedT;

    Eigen::VectorXd Vx_ddp_T;
    Eigen::VectorXd Vx_ipm_T;
    Eigen::VectorXd Vx_alm_T;
    Eigen::MatrixXd Vxx_ddp_T;
    Eigen::MatrixXd Vxx_ipm_T;
    Eigen::MatrixXd Vxx_alm_T;
    std::vector<Eigen::VectorXd> Vx;
    std::vector<Eigen::MatrixXd> Vxx;

    std::vector<Eigen::VectorXd> Rp_c, Rd_c, R_c;
    std::vector<Eigen::VectorXd> Rp_ec, Rd_ec, R_ec;
    Eigen::VectorXd Rp_cT, Rd_cT;
    Eigen::VectorXd Rp_ecT, Rd_ecT;

    std::vector<Eigen::VectorXd> Sinv_r_all;
    std::vector<Eigen::MatrixXd> Sinv_Y_Qyx_all, Sinv_Y_Qyu_all;
    std::vector<Eigen::VectorXd> Qx_ipm_all, Qu_ipm_all, hat_Qu_ipm_all;
    std::vector<Eigen::MatrixXd> hat_Qux_ipm_all, hat_Quu_ipm_all;
    std::vector<Eigen::MatrixXd> rho_Qzx_all, rho_Qzu_all;
    std::vector<Eigen::VectorXd> Qx_alm_all, Qu_alm_all, hat_Qu_alm_all;
    std::vector<Eigen::MatrixXd> hat_Qux_alm_all, hat_Quu_alm_all;


    // Eigen::VectorXd Vx;
    // Eigen::MatrixXd Vxx;

    double opterror;
    double opterror_rpT_ec;
    double opterror_rdT_ec;
    double opterror_rpT_c;
    double opterror_rdT_c;
    double opterror_rp_ec;
    double opterror_rd_ec;
    double opterror_rp_c;
    double opterror_rd_c;
    Eigen::VectorXd dV; // Expected Value Change

    std::vector<Eigen::VectorXd> X_new;
    std::vector<Eigen::VectorXd> U_new;
    std::vector<Eigen::VectorXd> S_new;
    std::vector<Eigen::VectorXd> Y_new;
    std::vector<Eigen::VectorXd> C_new;
    std::vector<Eigen::VectorXd> R_new;
    std::vector<Eigen::VectorXd> Z_new;
    std::vector<Eigen::VectorXd> EC_new;

    Eigen::VectorXd ST_new;
    Eigen::VectorXd YT_new;
    Eigen::VectorXd CT_new;
    Eigen::VectorXd RT_new;
    Eigen::VectorXd ZT_new;
    Eigen::VectorXd ECT_new;

    std::vector<Eigen::VectorXd> Rp_c_new;
    std::vector<Eigen::VectorXd> Rp_ec_new;
    Eigen::VectorXd Rp_cT_new;
    Eigen::VectorXd Rp_ecT_new;

    std::vector<double> all_cost;

    // Algorithm
    void calcAllDiff();
    void calcResiduals();
    void calcCres();
    void calcCTres();
    void calcECres();
    void calcECTres();
    void backwardPass();
    void checkRegulate();
    void forwardPass();
    double calcTotalCost(const std::vector<Eigen::VectorXd>& X, const std::vector<Eigen::VectorXd>& U);
    void logPrint();
};

extern template class ALIPDDP<double>;
