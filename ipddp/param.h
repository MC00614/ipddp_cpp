struct Param {
    int max_iter = 500; // maximum iteration
    int max_inner_iter = 100; // maximum iteration for inner loop
    double tolerance = 1e-3; // tolerance for termination
    double mu = 1.0; // initial mu
    double mu_mul = 0.2; // multiplier for mu update
    double mu_exp = 1.2; // exponent for mu update
    double mu_min = 1e-4; // normally set as tolerance / 10
    Eigen::VectorXd lambda; // initial lambda
    Eigen::VectorXd lambdaT; // initial lambda for terminal
    double rho = 1.0; // initial rho
    double rho_mul = 10.0; // multiplier for rho update
    double rho_max = 1e7; // maximum rho
    int max_step_iter = 10; // maximum iteration for forward pass
    double reg_exp = 1.6; // exponent for regularization update
    int max_regularization = 24; // maximum regularization for backward pass
};
