void IPDDP::forwardPass() {
    Eigen::MatrixXd X_new(model->dim_x, model->N+1);
    Eigen::MatrixXd U_new(model->dim_u, model->N);
    Eigen::MatrixXd Y_new(model->dim_c, model->N);
    Eigen::MatrixXd S_new(model->dim_c, model->N);
    Eigen::MatrixXd C_new(model->dim_c, model->N);

    double tau = std::max(0.99, 1.0 - param.mu);
    
    double cost_new = 0.0;
    double logcost_new = 0.0;
    double barrier_g_new = 0.0;
    double barrier_h_new = 0.0;
    double error_new = 0.0;

    // Maximum step (with slack)
    int max_slack_step;
    bool max_slack_step_failed = false;

    forward_failed = false;

    for (int step = 0; step < MAX_STEP; ++step) {
        double step_size = step_list[step];
        X_new.col(0) = X.col(0);
        // 1. Can be optimized with SIMD (cols function)
        // 2. Define dx can be efficient approach
        for (int t = 0; t < model->N; ++t) {
            int t_dim_x = t * model->dim_rn;
            Eigen::VectorXd dx = model->perturb(X_new.col(t), X.col(t));
            Y_new.col(t) = Y.col(t) + (step_size * ky.col(t)) + (Ky.middleCols(t_dim_x, model->dim_rn) * dx);
            U_new.col(t) = U.col(t) + (step_size * ku.col(t)) + (Ku.middleCols(t_dim_x, model->dim_rn) * dx);
            X_new.col(t+1) = model->f(X_new.col(t), U_new.col(t)).cast<double>();
        }
        max_slack_step_failed = false;
        for (int t = 0; t < model->N; ++t) {
            if (model->dim_g) {
                if ((Y_new.col(t).topRows(model->dim_g).array() < (1 - tau) * Y.col(t).topRows(model->dim_g).array()).any())
                {max_slack_step_failed = true; break;}
            }
            for (int i = 0; i < model->dim_hs.size(); ++i) {
                if ((Y_new.col(t).row(dim_hs_top[i]).array() - Y_new.col(t).middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1).norm()
                < (1 - tau)*(1 - tau) * (Y.col(t).row(dim_hs_top[i]).array() - Y.col(t).middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1).norm())).any())
                {max_slack_step_failed = true; break;}
            }
            if (max_slack_step_failed) {break;}
        }
        // Success Step
        if (!max_slack_step_failed) {
            max_slack_step = step;
            break;
        }
    }
    if (max_slack_step_failed) {
        std::cout<<"Forward Failed (in max slack step)"<<std::endl;
        forward_failed = true;
        return;
    }

    // Outer line search (with slack)
    for (int slack_step = max_slack_step; slack_step < MAX_STEP; ++slack_step) {
        forward_failed = true;
        double slack_step_size = step_list[slack_step];
        X_new.col(0) = X.col(0);
        for (int t = 0; t < model->N; ++t) {
            int t_dim_x = t * model->dim_rn;
            Eigen::VectorXd dx = model->perturb(X_new.col(t), X.col(t));
            Y_new.col(t) = Y.col(t) + (slack_step_size * ky.col(t)) + (Ky.middleCols(t_dim_x, model->dim_rn) * dx);
            U_new.col(t) = U.col(t) + (slack_step_size * ku.col(t)) + (Ku.middleCols(t_dim_x, model->dim_rn) * dx);
            X_new.col(t+1) = model->f(X_new.col(t), U_new.col(t)).cast<double>();
        }
        cost_new = calculateTotalCost(X_new, U_new);
        barrier_g_new = 0.0;
        barrier_h_new = 0.0;
        if (model->dim_g) {barrier_g_new = Y_new.topRows(model->dim_g).array().log().sum();}
        for (int i = 0; i < model->dim_hs.size(); ++i) {barrier_h_new += log(Y_new.row(dim_hs_top[i]).array().pow(2.0).sum() - Y_new.middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1).array().pow(2.0).sum())/2;}
        logcost_new = cost_new - param.mu * (barrier_g_new + barrier_h_new);
        for (int t = 0; t < model->N; ++t) {
            C_new.col(t) = model->c(X_new.col(t), U_new.col(t)).cast<double>();
        }
        error_new = std::max(param.tolerance, (C_new + Y_new).lpNorm<1>());
        if (logcost >= logcost_new && error >= error_new) {;}
        else {continue;}

        // Inner line search (with dual)
        bool inner_failed = false;
        // for (int dual_step = 0; dual_step < MAX_STEP; ++dual_step) {
        for (int dual_step = slack_step; dual_step <= slack_step; ++dual_step) {
            double dual_step_size = step_list[dual_step];
            for (int t = 0; t < model->N; ++t) {
                int t_dim_x = t * model->dim_rn;
                Eigen::VectorXd dx = model->perturb(X_new.col(t), X.col(t));
                S_new.col(t) = S.col(t) + (dual_step_size * ks.col(t)) + (Ks.middleCols(t_dim_x, model->dim_rn) * dx);
            }
            inner_failed = false;
            for (int t = 0; t < model->N; ++t) {
                if (model->dim_g) {
                    if ((S_new.col(t).topRows(model->dim_g).array() < (1 - tau) * S.col(t).topRows(model->dim_g).array()).any())
                    {inner_failed = true; break;}
                }
                for (int i = 0; i < model->dim_hs.size(); ++i) {
                    if ((S_new.col(t).row(dim_hs_top[i]).array() - S_new.col(t).middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1).norm()
                    < (1 - tau)*(1 - tau) * (S.col(t).row(dim_hs_top[i]).array() - S.col(t).middleRows(dim_hs_top[i]+1, model->dim_hs[i]-1).norm())).any())
                    {inner_failed = true; break;}
                }
                if (inner_failed) {break;}
            }
            if (!inner_failed) {
                std::cout<<"dual_step = "<<dual_step<<std::endl;
                std::cout<<"slack_step = "<<slack_step<<std::endl;
                break;
            }
        }
        if (inner_failed) {continue;}
        else {
            forward_failed = false;
            step = slack_step;
            break;
        }
    }

    if (!forward_failed) {
        cost = cost_new;
        logcost = logcost_new;
        error = error_new;
        X = X_new;
        U = U_new;
        Y = Y_new;
        S = S_new;
        C = C_new;
        // std::cout<<"Y = "<<Y.transpose()<<std::endl;
    }
    else {std::cout<<"Forward Failed"<<std::endl;}
}