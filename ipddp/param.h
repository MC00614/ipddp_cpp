struct Param {
    int max_iter;
    double tolerance;
    double mu = 1.0;
    Eigen::VectorXd lambda;
    Eigen::VectorXd lambdaT;
    double rho = 1.0;
    int max_step_iter = 10;
    int max_regularization = 24;
};
