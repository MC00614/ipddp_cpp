struct Param {
    int max_iter = 500;
    int max_inner_iter = 100;
    double tolerance = 1e-3;
    double mu = 1.0;
    double mu_mul = 0.2;
    double mu_exp = 1.2;
    double mu_min = 1e-4;
    Eigen::VectorXd lambda;
    Eigen::VectorXd lambdaT;
    double rho = 1.0;
    double rho_mul = 10.0;
    double rho_max = 1e7;
    int max_step_iter = 10;
    int max_regularization = 24;
};
