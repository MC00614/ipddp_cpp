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

    double rho = 1.0; // initial rho
    double rhoT = rho; // initial rho
    double rho_mul = 10.0; // multiplier for rho update
    double rho_max = 1e+6; // maximum rho

    double reg1_exp = 10.0; // exponent for regularization update
    double reg1_min = 1e-6; // minimum regularization
    double reg2_exp = 10.0; // exponent for regularization update
    double reg2_min = 1.0; // minimum regularization
    int max_regularization = 10; // maximum regularization for backward pass
    
    // double corr_p_min = 1e-6; // minimum correction for primal
    // double corr_d_min = 1e-9; // minimum correction for dual
    // double corr_p_mul = 10.0; // multiplier for primal correction
    // double corr_d_mul = 10.0; // multiplier for dual correction
    // int max_inertia_correction = 0; // maximum inertia correction for backward pass (0 = LLT: no correction)
    
    int max_step_iter = 10; // maximum iteration for forward pass
    bool forward_early_termination = false; // early termination for forward pass
    int forward_filter = 1; // filter selection for forward pass (0 = standard, 1 = combined)
    int forward_cost_threshold = 0.1; // cost threshold for forward pass

    // Only support same dynamics across step
    bool is_quaternion_in_state = false;
    int quaternion_idx;
};
