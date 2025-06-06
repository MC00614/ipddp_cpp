#pragma once
#include <Eigen/Dense>

struct Param {
    int max_iter = 500; // maximum iteration
    int max_inner_iter = 100; // maximum iteration for inner loop
    double tolerance = 1e-4; // tolerance for termination
    double mu = 1.0; // initial barrier parameter
    double muT = mu; // initial barrier parameter
    double mu_mul = 0.2; // multiplier for mu update
    double mu_exp = 1.2; // exponent for mu update
    double mu_min = 1e-6; // minimum mu (normally set as tolerance / 10)
    Eigen::VectorXd lambda; // initial lambda
    Eigen::VectorXd lambdaT; // initial lambda for terminal
    double rho = 1.0; // initial rho
    double rho_mul = 10.0; // multiplier for rho update
    double rho_max = 1e+6; // maximum rho
    int max_step_iter = 10; // maximum iteration for forward pass
    double reg1_exp = 10.0; // exponent for regularization update
    double reg1_min = 1e-6; // minimum regularization
    double reg2_exp = 10.0; // exponent for regularization update
    double reg2_min = 1.0; // minimum regularization
    int max_regularization = 10; // maximum regularization for backward pass

    double corr_p_min = 1e-6; // minimum correction for primal
    double corr_d_min = 1e-9; // minimum correction for dual
    double corr_p_mul = 10.0; // multiplier for primal correction
    double corr_d_mul = 10.0; // multiplier for dual correction
    int max_inertia_correction = 0; // maximum inertia correction for backward pass (0 = LLT: no correction)

    bool forward_early_termination = false; // early termination for forward pass
    int forward_filter = 0; // filter selection for forward pass (0 = standard, 1 = combined)
    int forward_cost_threshold = 0.1; // cost threshold for forward pass

    // Deprecated Parameters
    // // Automatic Initialization of Slack and Lagrange
    // bool auto_init = false; // Master Button for all constraints
    // bool auto_init_ec = false; // for equality constraints)
    // bool auto_init_ecT = false; // for terminal equality constraints)
    // bool auto_init_noc = false; // for inequality constraints
    // bool auto_init_cc = false; // for conic constraints
    // bool auto_init_nocT = false; // for terminal inequality constraints
    // bool auto_init_ccT = false; // for terminal conic constraints

    // // Automatic Scaling of Constraints
    // bool auto_scale = false; // Master Button for all constraints
    // bool auto_scale_ec = false; // for equality constraints
    // bool auto_scale_ecT = false; // for terminal equality constraints
    // bool auto_scale_noc = false; // for inequality constraints
    // bool auto_scale_cc = false; // for conic constraints
    // bool auto_scale_nocT = false; // for terminal inequality constraints
    // bool auto_scale_ccT = false; // for terminal conic constraints
};
