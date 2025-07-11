
template <typename Scalar>
ALIPDDP<Scalar>::ALIPDDP(OptimalControlProblem<Scalar>& ocp_ref) : ocp(std::make_shared<OptimalControlProblem<Scalar>>(ocp_ref)), N(ocp_ref.getHorizon()) {
    ocp.init();

    is_c_active.resize(N);
    is_ec_active.resize(N);
    is_c_active_all = false;
    is_ec_active_all = false;
    for (int k = 0; k < N; ++k) {
        is_c_active[k] = (ocp.getDimC(k) > 0);
        is_ec_active[k] = (ocp.getDimEC(k) > 0);
        is_c_active_all |= is_c_active[k];
        is_ec_active_all |= is_ec_active[k];
    }
    is_cT_active = (ocp.getDimCT() > 0);
    is_ecT_active = (ocp.getDimECT() > 0);

    X.resize(N + 1);
    U.resize(N);
    Z.resize(N);
    R.resize(N);
    Y.resize(N);
    S.resize(N);
    C.resize(N);
    EC.resize(N);
    e.resize(N);

    for (int k = 0; k < N; ++k) {
        if (ocp.isXinit(k)) X[k] = ocp.getXinit(k);
        else X[k].setZero(ocp.getDimX(k));
        
        if (ocp.isUinit(k)) U[k] = ocp.getUinit(k);
        else U[k].setZero(ocp.getDimU(k));
    
        R[k].setZero(ocp.getDimEC(k));
        Z[k].setZero(ocp.getDimEC(k));
        S[k].setZero(ocp.getDimC(k));
        Y[k].setZero(ocp.getDimC(k));
        e[k].setZero(ocp.getDimC(k));
        
        if (ocp.getDimG(k) > 0) {
            S[k].head(ocp.getDimG(k)).setOnes();
            Y[k].head(ocp.getDimG(k)).setOnes();
            e[k].head(ocp.getDimG(k)).setOnes();
        }
        for (auto dim_h_top : ocp.getDimHsTop(k)) {
            S[k](dim_h_top) = 1.0;
            Y[k](dim_h_top) = 1.0;
            e[k](dim_h_top) = 1.0;
        }
    }
    if (ocp.isXinit(N)) X[N] = ocp.getXinit(N);


    RT.setZero(ocp.getDimECT());
    ZT.setZero(ocp.getDimECT());
    ST.setZero(ocp.getDimCT());
    YT.setZero(ocp.getDimCT());
    eT.setZero(ocp.getDimCT());
    if (ocp.getDimGT() > 0) {
        ST.head(ocp.getDimGT()).setOnes();
        YT.head(ocp.getDimGT()).setOnes();
        eT.head(ocp.getDimGT()).setOnes();
    }
    for (auto dim_hT_top : ocp.getDimHTsTop()) {
        ST(dim_hT_top) = 1.0;
        YT(dim_hT_top) = 1.0;
        eT(dim_hT_top) = 1.0;
    }

    fx_all.resize(N);
    fu_all.resize(N);
    // fxx_all.resize(N);
    // fxu_all.resize(N);
    // fuu_all.resize(N);
    qx_all.resize(N);
    qu_all.resize(N);
    qxx_all.resize(N);
    qxu_all.resize(N);
    quu_all.resize(N);
    cx_all.resize(N);
    cu_all.resize(N);
    ecx_all.resize(N);
    ecu_all.resize(N);
    
    du.resize(N);
    dr.resize(N);
    dz.resize(N);
    ds.resize(N);
    dy.resize(N);
    Ku.resize(N);
    Kr.resize(N);
    Kz.resize(N);
    Ks.resize(N);
    Ky.resize(N);

    X_new.resize(N + 1);
    U_new.resize(N);
    S_new.resize(N);
    Y_new.resize(N);
    C_new.resize(N);
    R_new.resize(N);
    Z_new.resize(N);
    EC_new.resize(N);
}

template <typename Scalar>
ALIPDDP<Scalar>::~ALIPDDP() {
}

template <typename Scalar>
void ALIPDDP<Scalar>::init(Param param) {
    this->param = param;

    // Only support same dynamics
    dim_rn.resize(N);
    if (this->param.is_quaternion_in_state) {
        if (!this->param.quaternion_idx) {
            throw std::runtime_error("ALIPDDP: quaternion_idx must be initialized. (see is_quaternion_in_state)");
        }
        else {
            for(int k = 0; k < N; ++k) {
                dim_rn[k] = ocp.getDimX(k) - 1;
            }
            dim_rnT = ocp.getDimXT() - 1;
        }
    }
    else {
        for(int k = 0; k < N; ++k) {
            dim_rn[k] = ocp.getDimX(k);
        }
        dim_rnT = ocp.getDimXT();
    }

    for (int k = 0; k < N; ++k) {
        du[k] = Eigen::VectorXd::Zero(ocp.getDimU(k));
        Ku[k] = Eigen::MatrixXd::Zero(ocp.getDimU(k), dim_rn[k]);
        if (ocp.getDimC(k)) {
            ds[k] = Eigen::VectorXd::Zero(ocp.getDimC(k));
            dy[k] = Eigen::VectorXd::Zero(ocp.getDimC(k));
            Ks[k] = Eigen::MatrixXd::Zero(ocp.getDimC(k), dim_rn[k]);
            Ky[k] = Eigen::MatrixXd::Zero(ocp.getDimC(k), dim_rn[k]);
        }
        if (ocp.getDimEC(k)) {
            dr[k] = Eigen::VectorXd::Zero(ocp.getDimEC(k));
            dz[k] = Eigen::VectorXd::Zero(ocp.getDimEC(k));
            Kr[k] = Eigen::MatrixXd::Zero(ocp.getDimEC(k), dim_rn[k]);
            Kz[k] = Eigen::MatrixXd::Zero(ocp.getDimEC(k), dim_rn[k]);
        }
    }

    initialRoll();
    
    // check dim_c
    // if (this->param.mu == 0) {this->param.mu = cost / N / dim_c;} // Auto Select

    lambda.resize(N);
    for(int k = 0; k < N; ++k) {
        if (ocp.getDimEC(k)) {lambda[k].setZero(ocp.getDimEC(k));}
    }
    lambdaT.setZero(ocp.getDimECT());

    resetFilter();
    resetRegulation();

    for (int i = 0; i <= this->param.max_step_iter; ++i) {
        step_list.push_back(std::pow(2.0, static_cast<double>(-i)));
    }
}

template <typename Scalar>
void ALIPDDP<Scalar>::initialRoll() {
    for (int k = 0; k < N; ++k) {
        const Eigen::VectorXd& xk = X[k];
        const Eigen::VectorXd& uk = U[k];
        if (ocp.getDimC(k)) {C[k] = ocp.c(k, xk, uk);}
        if (ocp.getDimEC(k)) {EC[k] = ocp.ec(k, xk, uk);}
        X[k + 1] = ocp.f(k, xk, uk);
    }
    if (ocp.getDimCT()) {CT = ocp.cT(X[N]);}
    if (ocp.getDimECT()) {ECT = ocp.ecT(X[N]);}
    
    cost = calcTotalCost(X, U);
}

template <typename Scalar>
void ALIPDDP<Scalar>::resetFilter() {
    double barriercost = 0.0;
    double barriercostT = 0.0;
    double alcost = 0.0;
    double alcostT = 0.0;
    
    // CHECK: Possible to use GEMM if we use same dynamics & constraints across all timestep
    // Implement 2 way method?

    for (int k = 0; k < N; ++k) {
        const int dim_g = ocp.getDimG(k);
        const int dim_h = ocp.getDimH(k);
        if (dim_g) {
            no_helper::addBarrierCost(barriercost, S[k].head(dim_g));
        }

        if (dim_h) {
            soc_helper::addBarrierCost(barriercost, S[k], ocp.getDimHs(k), ocp.getDimHsTop(k));
        }
    }
    const int dim_gT = ocp.getDimGT();
    const int dim_hT = ocp.getDimHT();
    if (dim_gT) {
        no_helper::addBarrierCost(barriercostT, ST.head(dim_gT));
    }
    if (dim_hT) {
        soc_helper::addBarrierCost(barriercostT, ST, ocp.getDimHTs(), ocp.getDimHTsTop());
    }

    for (int k = 0; k < N; ++k) {
        if (ocp.getDimEC(k)) {
            alcost += (lambda[k].transpose() * R[k]).sum() + (0.5 * param.rho * R[k].squaredNorm());
        }
    }
    if (ocp.getDimECT()) {
        alcostT += (lambdaT.transpose() * RT) + (0.5 * param.rhoT * RT.squaredNorm());
    }
    logcost = cost - (param.mu * barriercost + param.muT * barriercostT) + (alcost + alcostT);

    error = 0.0;
    for (int k = 0; k < N; ++k) {
        if (ocp.getDimEC(k)) {error += (EC[k] + R[k]).array().abs().sum();}
        if (ocp.getDimC(k)) {error += (C[k] + S[k]).array().abs().sum();}
    }
    if (ocp.getDimECT()) {error += (ECT + RT).array().abs().sum();}
    if (ocp.getDimCT()) {error += (CT + ST).array().abs().sum();}
    error = std::max(param.tolerance, error);
    
    step = 0;
    forward_failed = false;
}

template <typename Scalar>
void ALIPDDP<Scalar>::resetRegulation() {
    this->regulate = 0;
    this->backward_failed = false;
}

template <typename Scalar>
double ALIPDDP<Scalar>::calcTotalCost(const std::vector<Eigen::VectorXd>& X, const std::vector<Eigen::VectorXd>& U) {
    double cost = 0.0;
    for (int k = 0; k < N; ++k) {
        cost += ocp.q(k, X[k], U[k]);
    }
    cost += ocp.p(X[N]);
    return cost;
}

template <typename Scalar>
void ALIPDDP<Scalar>::solve() {
    update_counter = 0;
    iter = 0;
    is_diff_calculated = false;
    
    std::cout << std::setw(4) << "o_it"
    << std::setw(4) << "i_it"
    << std::setw(3) << "bp"
    << std::setw(3) << "fp"
    << std::setw(7) << "mu"
    << std::setw(16) << "rho"
    << std::setw(25) << "LogCost"
    << std::setw(22) << "OptError"
    << std::setw(18) << "Error"
    << std::setw(4) << "Reg"
    << std::setw(7) << "Step"
    << std::setw(5) << "Upt" << std::endl;

    // Outer Loop (Augmented Lagrangian & Interior Point Method)
    while (true) {
        inner_iter = 0;
        // Inner Loop (Differential Dynamic Programming)
        while (inner_iter < this->param.max_inner_iter) {
            if (param.max_iter < ++iter) {break;}

            if (!is_diff_calculated) {
                this->calcAllDiff();
                is_diff_calculated = true;
            }

            this->backwardPass();

            if (backward_failed) {
                this->logPrint();
                if (regulate==param.max_regularization) {break;}
                else {continue;}
            }
            
            this->forwardPass();
            if (!forward_failed) {
                is_diff_calculated = false;
                inner_iter++;
                update_counter++;
            }
            
            this->logPrint();
            
            all_cost.push_back(cost);
    
            // CHECK
            if (std::max(opterror, param.mu) <= param.tolerance) {
                // if (opterror <= param.tolerance) {
                std::cout << "Optimal Solution" << std::endl;
                return;
            }
    
            if (forward_failed && regulate==param.max_regularization) {
                break;
            }
    
            if ((opterror <= std::max(10.0 * param.mu, param.tolerance))) {
                break;
            }
            
            {
                // Seperate time frame?
                // If then, consider active marker to be vector (like switch condition)
                bool updated = false;
                if (is_c_active_all && opterror_rp_c < param.tolerance && opterror_rd_c < param.tolerance) {
                    if (param.mu > param.mu_min) {updated = true;}
                    param.mu = std::max(param.mu_min, std::min(param.mu_mul * param.mu, std::pow(param.mu, param.mu_exp)));
                }
                if (is_cT_active && opterror_rpT_c < param.tolerance && opterror_rdT_c < param.tolerance) {
                    if (param.muT > param.mu_min) {updated = true;}
                    param.muT = std::max(param.mu_min, std::min(param.mu_mul * param.muT, std::pow(param.muT, param.mu_exp)));
                }
                if (is_ec_active_all && opterror_rp_ec < param.tolerance && opterror_rd_ec < param.tolerance) {
                    if (param.rho < param.rho_max) {updated = true;}
                    param.rho = std::min(param.rho_max, param.rho_mul * param.rho);
                    for (int k = 0; k < N; ++k) {
                        lambda[k] = lambda[k] + param.rho * R[k];
                    }
                }
                if (is_ecT_active && opterror_rpT_ec < param.tolerance && opterror_rdT_ec < param.tolerance) {
                    if (param.rhoT < param.rho_max) {updated = true;}
                    param.rhoT = std::min(param.rho_max, param.rho_mul * param.rhoT);
                    lambdaT = lambdaT + param.rhoT * RT;
                }
                if (updated) {
                    resetFilter();
                    inner_iter = 0;
                }
                // std::cout << "opterror_rp_c = " << opterror_rp_c << std::endl;
                // std::cout << "opterror_rd_c = " << opterror_rd_c << std::endl;
                // std::cout << "opterror_rpT_c = " << opterror_rpT_c << std::endl;
                // std::cout << "opterror_rdT_c = " << opterror_rdT_c << std::endl;
                // std::cout << "opterror_rpT_ec = " << opterror_rpT_ec << std::endl;
                // std::cout << "opterror_rdT_ec = " << opterror_rdT_ec << std::endl;
            }
        }
        if (param.max_iter < iter) {
            std::cout << "Max Iteration" << std::endl;
            break;
        }

        bool mu_stop = (param.mu <= param.mu_min);
        bool muT_stop = (param.muT <= param.mu_min);
        bool rho_stop = (param.rho >= param.rho_max);
        bool rhoT_stop = (param.rhoT >= param.rho_max);

        bool c_done   = (!is_c_active_all || mu_stop);
        bool ec_done   = (!is_ec_active_all || rho_stop);
        bool cT_done  = (!is_cT_active || muT_stop);
        bool ecT_done = (!is_ecT_active || rhoT_stop);
      
        if (c_done && cT_done && ec_done && ecT_done) {
            std::cout << "Outer Max/Min" << std::endl;
            break;
        }

        // Update Outer Loop Parameters
        if (is_c_active_all) {param.mu = std::max(param.mu_min, std::min(param.mu_mul * param.mu, std::pow(param.mu, param.mu_exp)));}
        if (is_cT_active) {param.muT = std::max(param.mu_min, std::min(param.mu_mul * param.muT, std::pow(param.muT, param.mu_exp)));}
        if (is_ec_active_all) {
            param.rho = std::min(param.rho_max, param.rho_mul * param.rho);
            for (int k = 0; k < N; ++k) {
                lambda[k] = lambda[k] + param.rho * R[k];
            }
        }
        if (is_ecT_active) {
            param.rhoT = std::min(param.rho_max, param.rho_mul * param.rhoT);
            lambdaT = lambdaT + param.rhoT * RT;
        }
        resetFilter();
        resetRegulation();
    }
}

template <typename Scalar>
void ALIPDDP<Scalar>::calcAllDiff() {
    for (int k = 0; k < N; ++k) {
        const Eigen::VectorXd& x = X[k];
        const Eigen::VectorXd& u = U[k];

        // No Differentiation for u
        qu_all[k].noalias() = ocp.qu(k, x, u);
        quu_all[k].noalias() = ocp.quu(k, x, u);

        if (is_c_active[k]) {
            cu_all[k].noalias() = ocp.cu(k, x, u);
        }
        if (is_ec_active[k]) {
            ecu_all[k].noalias() = ocp.ecu(k, x, u);
        }

        if (!param.is_quaternion_in_state) {
            fx_all[k].noalias() = ocp.fx(k, x, u);
            fu_all[k].noalias() = ocp.fu(k, x, u);
            qx_all[k].noalias() = ocp.qx(k, x, u);
            qxx_all[k].noalias() = ocp.qxx(k, x, u);
            qxu_all[k].noalias() = ocp.qxu(k, x, u);
            if (is_c_active[k]) {
                cx_all[k].noalias() = ocp.cx(k, x, u);
            }
            if (is_ec_active[k]) {
                ecx_all[k].noalias() = ocp.ecx(k, x, u);
            }
        }
        else {
            Eigen::MatrixXd E;
            quaternion_helper::calcE(E, x, param.quaternion_idx, dim_rn[k]);

            Eigen::MatrixXd EE;
            const Eigen::VectorXd xn = X[k+1];
            quaternion_helper::calcEE(EE, xn, param.quaternion_idx, dim_rn[k]);

            Eigen::MatrixXd Id;
            Eigen::VectorXd qx = ocp.qx(k, x, u);
            double qx_q = qx.segment(param.quaternion_idx, 4).transpose() * x.segment(param.quaternion_idx, 4);
            quaternion_helper::Id(Id, qx_q, param.quaternion_idx, dim_rn[k]);

            fx_all[k].noalias() = EE.transpose() * ocp.fx(k, x, u) * E;
            fu_all[k].noalias() = EE.transpose() * ocp.fu(k, x, u);

            qx_all[k].noalias() = E.transpose() * qx;
            qxx_all[k].noalias() = E.transpose() * ocp.qxx(k, x, u) * E - Id;
            qxu_all[k].noalias() = E.transpose() * ocp.qxu(k, x, u);

            if (is_c_active[k]) {
                cx_all[k].noalias() = ocp.cx(k, x, u) * E;
            }
            if (is_ec_active[k]) {
                ecx_all[k].noalias() = ocp.ecx(k, x, u) * E;
            }
        }
    }
    const Eigen::VectorXd& xT = X[N];

    if (!param.is_quaternion_in_state) {
        px_all.noalias() = ocp.px(xT);
        pxx_all.noalias() = ocp.pxx(xT);
        if (ocp.getDimCT()) {
            cTx_all.noalias() = ocp.cTx(xT);
        }
        if (ocp.getDimECT()) {
            ecTx_all.noalias() = ocp.ecTx(xT);
        }
    }
    else {
        Eigen::MatrixXd E;
        quaternion_helper::calcE(E, xT, param.quaternion_idx, dim_rnT);
        Eigen::MatrixXd Id;
        Eigen::VectorXd px = ocp.px(xT);
        double px_q = px.segment(param.quaternion_idx, 4).transpose() * xT.segment(param.quaternion_idx, 4);
        quaternion_helper::Id(Id, px_q, param.quaternion_idx, dim_rnT);
        px_all.noalias() = E.transpose() * px;
        pxx_all.noalias() = E.transpose() * ocp.pxx(xT) * E - Id;
        if (ocp.getDimCT()) {
            cTx_all.noalias() = ocp.cTx(xT) * E;
        }
        if (ocp.getDimECT()) {
            ecTx_all.noalias() = ocp.ecTx(xT) * E;
        }
    }
}

template <typename Scalar>
void ALIPDDP<Scalar>::backwardPass() {
    Eigen::VectorXd Vx;
    Eigen::MatrixXd Vxx;

    Eigen::VectorXd Qx, Qu;
    Eigen::MatrixXd Qxx, Qxu, Quu;
    Eigen::MatrixXd hat_Quu;

    Eigen::VectorXd rp_c, rd_c, r_c;
    Eigen::VectorXd Sinv_r;
    Eigen::MatrixXd Sinv_Y_Qyx, Sinv_Y_Qyu;

    Eigen::VectorXd rp_ec, rd_ec, r_ec;
    Eigen::MatrixXd rho_Qzx, rho_Qzu;
    
    Eigen::LLT<Eigen::MatrixXd> Quu_llt;

    opterror = 0.0;
    opterror_rpT_ec = 0.0;
    opterror_rdT_ec = 0.0;
    opterror_rpT_c = 0.0;
    opterror_rdT_c = 0.0;
    opterror_rp_c = 0.0;
    opterror_rd_c = 0.0;
    opterror_rp_ec = 0.0;
    opterror_rd_ec = 0.0;

    dV = Eigen::VectorXd::Zero(2);

    checkRegulate();

    double reg1_mu = param.reg1_min * (std::pow(param.reg1_exp, regulate));
    double reg2_mu = param.reg2_min * (std::pow(param.reg2_exp, regulate));

    Vx = px_all;
    Vxx = pxx_all;

    // Inequality Terminal Constraint
    if (is_cT_active) {
        const int dim_cT = ocp.getDimCT();
        const int dim_gT = ocp.getDimGT();
    
        const auto& dim_hTs     = ocp.getDimHTs();
        const auto& dim_hTs_top = ocp.getDimHTsTop();
        const int dim_hTs_max   = ocp.getDimHTsMax();

        Eigen::Ref<const Eigen::MatrixXd> QyxT = cTx_all;

        Eigen::VectorXd rp_cT = CT + ST;
        Eigen::VectorXd rd_cT(dim_cT);
        rd_cT.head(dim_gT) = ST.head(dim_gT).cwiseProduct(YT.head(dim_gT));
        for (int i = 0; i < dim_hTs.size(); ++i) {
            const int d   = dim_hTs[i];
            const int idx = dim_hTs_top[i];
            soc_helper::LtimesVec(rd_cT.segment(idx, d), YT.segment(idx, d), ST.segment(idx, d));
        }
        rd_cT -= param.muT * eT;

        Eigen::VectorXd rT(dim_cT);
        rT.head(dim_gT) = YT.head(dim_gT).cwiseProduct(rp_cT.head(dim_gT));
        for (int i = 0; i < dim_hTs.size(); ++i) {
            const int d   = dim_hTs[i];
            const int idx = dim_hTs_top[i];
            soc_helper::LtimesVec(rT.segment(idx, d), YT.segment(idx, d), rp_cT.segment(idx, d));
        }
        rT -= rd_cT;

        Eigen::VectorXd Sinv_rT(dim_cT);
        Sinv_rT.head(dim_gT) = rT.head(dim_gT).cwiseQuotient(ST.head(dim_gT));
        for (int i = 0; i < dim_hTs.size(); ++i) {
            const int d   = dim_hTs[i];
            const int idx = dim_hTs_top[i];
            soc_helper::LinvTimesVec(Sinv_rT.segment(idx, d), ST.segment(idx, d), rT.segment(idx, d));
        }

        Eigen::MatrixXd Sinv_Y_QyxT(dim_cT, dim_rnT);
        Sinv_Y_QyxT.topRows(dim_gT) = QyxT.topRows(dim_gT).array().colwise() * (YT.head(dim_gT).array() / ST.head(dim_gT).array());
    
        Eigen::MatrixXd Sinv_Y_hT_max(dim_hTs_max, dim_hTs_max);
        for (int i = 0; i < dim_hTs.size(); ++i) {
            const int d   = dim_hTs[i];
            const int idx = dim_hTs_top[i];
            Eigen::Ref<Eigen::MatrixXd> Sinv_Y_hT = Sinv_Y_hT_max.topLeftCorner(d, d);
            soc_helper::LinvTimesArrow(Sinv_Y_hT, ST.segment(idx, d), YT.segment(idx, d));
            Sinv_Y_QyxT.middleRows(idx, d) = Sinv_Y_hT * QyxT.middleRows(idx, d);
        }

        dsT = - rp_cT;
        KsT = - QyxT;
        
        dyT = Sinv_rT;
        KyT = Sinv_Y_QyxT;

        Vx += KyT.transpose() * CT + QyxT.transpose() * dyT;
        Vxx += QyxT.transpose() * KyT + KyT.transpose() * QyxT;

        // with slack
        // Eigen::VectorXd QsT = YTinv * rd_cT;
        // Eigen::VectorXd QyT = rp_cT;
        // Eigen::MatrixXd I_cT = Eigen::VectorXd::Ones(dim_cT).asDiagonal();
        // dV(0) += QsT.transpose() * dsT;
        // dV(0) += QyT.transpose() * dyT;
        // dV(1) += dyT.transpose() * I_cT * dsT; 

        // Vx += KyT.transpose() * QyT + QyxT.transpose() * dyT;
        // Vx += KsT.transpose() * QsT + KyT.transpose() * I_cT * dsT + KsT.transpose() * I_cT * dyT;

        // Vxx += QyxT.transpose() * KyT + KyT.transpose() * QyxT;
        // Vxx += KyT.transpose() * I_cT * KsT + KsT.transpose() * I_cT * KyT;

        opterror = std::max({rp_cT.lpNorm<Eigen::Infinity>(), rd_cT.lpNorm<Eigen::Infinity>(), opterror});
        opterror_rpT_c = std::max({rp_cT.lpNorm<Eigen::Infinity>(), opterror_rpT_c});
        opterror_rdT_c = std::max({rd_cT.lpNorm<Eigen::Infinity>(), opterror_rdT_c});
    }

    // Equality Terminal Constraint
    if (is_ecT_active) {
        // const int dim_ecT = ocp.getDimECT();

        Eigen::Ref<const Eigen::MatrixXd> QzxT = ecTx_all;

        Eigen::VectorXd rp_ecT = ECT + RT;
        Eigen::VectorXd rd_ecT = ZT + lambdaT + (param.rhoT * RT);
        
        drT = - rp_ecT;
        KrT = - QzxT;
        dzT = param.rhoT * rp_ecT - rd_ecT;
        KzT = param.rhoT * QzxT;

        // CHECK: New Value Decrement
        // dV(0) += dzT.transpose() * ECT;

        Vx += KzT.transpose() * ECT + QzxT.transpose() * dzT;
        Vxx += QzxT.transpose() * KzT + KzT.transpose() * QzxT;

        // with slack
        // Eigen::VectorXd QrT = rd_ecT;
        // Eigen::VectorXd QzT = rp_ecT;
        // Eigen::MatrixXd I_ecT = Eigen::VectorXd::Ones(dim_ecT).asDiagonal();
        // dV(0) += QrT.transpose() * drT;
        // dV(0) += QzT.transpose() * dzT;
        // dV(1) += dzT.transpose() * I_ecT * drT; // Qrr = 0
        
        // Vx += KzT.transpose() * QzT + QzxT.transpose() * dzT;
        // Vx += KrT.transpose() * QrT + KzT.transpose() * I_ecT * drT + KrT.transpose() * I_ecT * dzT;
        
        // Vxx += QzxT.transpose() * KzT + KzT.transpose() * QzxT;
        // Vxx += KzT.transpose() * I_ecT * KrT + KrT.transpose() * I_ecT * KzT;
        
        opterror = std::max({rp_ecT.lpNorm<Eigen::Infinity>(), rd_ecT.lpNorm<Eigen::Infinity>(), opterror});
        opterror_rpT_ec = std::max({rp_ecT.lpNorm<Eigen::Infinity>(), opterror_rpT_ec});
        opterror_rdT_ec = std::max({rd_ecT.lpNorm<Eigen::Infinity>(), opterror_rdT_ec});
    }

    backward_failed = false;

    for (int k = N - 1; k >= 0; --k) {
        Eigen::Ref<const Eigen::MatrixXd> fx = fx_all[k];
        Eigen::Ref<const Eigen::MatrixXd> fu = fu_all[k];

        Eigen::Ref<const Eigen::VectorXd> qx = qx_all[k];
        Eigen::Ref<const Eigen::VectorXd> qu = qu_all[k];

        Eigen::Ref<const Eigen::MatrixXd> qxx = qxx_all[k];
        Eigen::Ref<const Eigen::MatrixXd> qxu = qxu_all[k];
        Eigen::Ref<const Eigen::MatrixXd> quu = quu_all[k];

        Qx = qx + (fx.transpose() * Vx);
        Qu = qu + (fu.transpose() * Vx);
        
        // DDP (TODO: Vector-Hessian Product)
        // Qxx = qxx + (fx.transpose() * Vxx * fx) + tensdot(Vx, fxx);
        // Qxu = qxu + (fx.transpose() * Vxx * fu) + tensdot(Vx, fxu);
        // Quu = quu + (fu.transpose() * Vxx * fu) + tensdot(Vx, fuu);
        
        // iLQR
        // Qxx = qxx + (fx.transpose() * Vxx * fx);
        // Qxu = qxu + (fx.transpose() * Vxx * fu);
        // Quu = quu + (fu.transpose() * Vxx * fu);

        // Regularization
        Vxx.diagonal().array() += reg1_mu;
        Qxx = qxx + (fx.transpose() * Vxx * fx);
        Qxu = qxu + (fx.transpose() * Vxx * fu);
        Quu = quu + (fu.transpose() * Vxx * fu);
        Quu.diagonal().array() += reg2_mu;

        Eigen::Ref<Eigen::VectorXd> du_ = du[k];
        Eigen::Ref<Eigen::MatrixXd> Ku_ = Ku[k];

        du_ = - Qu; // hat_Qu
        Ku_ = - Qxu.transpose(); // hat_Qxu
        hat_Quu = Quu;

        if (is_c_active[k]) {
            Eigen::Ref<Eigen::VectorXd> s = S[k];
            Eigen::Ref<Eigen::VectorXd> y = Y[k];
            Eigen::Ref<Eigen::VectorXd> c_v = C[k];
            
            Eigen::Ref<Eigen::MatrixXd> Qyx = cx_all[k];
            Eigen::Ref<Eigen::MatrixXd> Qyu = cu_all[k];
            
            const int dim_c = ocp.getDimC(k);
            const int dim_g = ocp.getDimG(k);

            const auto& dim_hs     = ocp.getDimHs(k);
            const auto& dim_hs_top = ocp.getDimHsTop(k);
            const int dim_hs_max   = ocp.getDimHsMax(k);

            Qx += Qyx.transpose() * y;
            Qu += Qyu.transpose() * y;
            
            rp_c = c_v + s;

            rd_c.resize(dim_c);
            rd_c.head(dim_g) = s.head(dim_g).cwiseProduct(y.head(dim_g));
            for (int i = 0; i < dim_hs.size(); ++i) {
                const int idx = dim_hs_top[i];
                const int n   = dim_hs[i];
                soc_helper::LtimesVec(rd_c.segment(idx, n), y.segment(idx, n), s.segment(idx, n));
            }
            rd_c -= param.mu * e[k];

            r_c.resize(dim_c);
            r_c.head(dim_g) = y.head(dim_g).cwiseProduct(rp_c.head(dim_g));
            for (int i = 0; i < dim_hs.size(); ++i) {
                const int d   = dim_hs[i];
                const int idx = dim_hs_top[i];
                soc_helper::LtimesVec(r_c.segment(idx, d), y.segment(idx, d), rp_c.segment(idx, d));
            }
            r_c -= rd_c;

            Sinv_r.resize(dim_c);
            Sinv_r.head(dim_g) = r_c.head(dim_g).cwiseQuotient(s.head(dim_g));
            for (int i = 0; i < dim_hs.size(); ++i) {
                const int d   = dim_hs[i];
                const int idx = dim_hs_top[i];
                soc_helper::LinvTimesVec(Sinv_r.segment(idx, d), s.segment(idx, d), r_c.segment(idx, d));
            }
        
            // more complex, but more fast
            Sinv_Y_Qyx.resize(dim_c, dim_rn[k]);
            Sinv_Y_Qyu.resize(dim_c, ocp.getDimU(k));
            Eigen::VectorXd Sinv_Y_g = y.head(dim_g).cwiseQuotient(s.head(dim_g));
            Sinv_Y_Qyx.topRows(dim_g) = Qyx.topRows(dim_g).array().colwise() * Sinv_Y_g.array();
            Sinv_Y_Qyu.topRows(dim_g) = Qyu.topRows(dim_g).array().colwise() * Sinv_Y_g.array();
            Eigen::MatrixXd SYinv_h_max(dim_hs_max, dim_hs_max);
            for (int i = 0; i < dim_hs.size(); ++i) {
                const int d   = dim_hs[i];
                const int idx = dim_hs_top[i];
                Eigen::Ref<Eigen::MatrixXd> Sinv_Y_h = SYinv_h_max.topLeftCorner(d, d);
                soc_helper::LinvTimesArrow(Sinv_Y_h, s.segment(idx, d), y.segment(idx, d));
                Sinv_Y_Qyx.middleRows(idx, d) = Sinv_Y_h * Qyx.middleRows(idx, d);
                Sinv_Y_Qyu.middleRows(idx, d) = Sinv_Y_h * Qyu.middleRows(idx, d);
            }

            /* less complex, but more slow
            Eigen::VectorXd Sinv_Y_g = y.head(dim_g).cwiseQuotient(s.head(dim_g));
            Sinv_Y_Qyx.topRows(dim_g) = Qyx.topRows(dim_g).array().colwise() * Sinv_Y_g.array();
            Sinv_Y_Qyu.topRows(dim_g) = Qyu.topRows(dim_g).array().colwise() * Sinv_Y_g.array();
            Eigen::VectorXd Y_Qyx_h_max(ocp.getDimHsMax(k));
            Eigen::VectorXd Y_Qyu_h_max(ocp.getDimHsMax(k));
            for (int i = 0; i < dim_hs.size(); ++i) {
                const int d   = dim_hs[i];
                const int idx = dim_hs_top[i];
                Eigen::Ref<Eigen::VectorXd> Y_Qyx_h = Y_Qyx_h_max.topRows(d);
                for (int j = 0; j < dim_rn[k]; ++j) {
                    soc_helper::LtimesVec(Y_Qyx_h, y.segment(idx, d), Qyx.block(idx, j, d, 1));
                    soc_helper::LinvTimesVec(Sinv_Y_Qyx.block(idx, j, d, 1), s.segment(idx, d), Y_Qyx_h);
                }
                Eigen::Ref<Eigen::VectorXd> Y_Qyu_h = Y_Qyu_h_max.topRows(d);
                for (int j = 0; j < ocp.getDimU(k); ++j) {
                    soc_helper::LtimesVec(Y_Qyu_h, y.segment(idx, d), Qyu.block(idx, j, d, 1));
                    soc_helper::LinvTimesVec(Sinv_Y_Qyu.block(idx, j, d, 1), s.segment(idx, d), Y_Qyu_h);
                }
            }
            */
            
            // Inplace Calculation
            du_ -= (Qyu.transpose() * Sinv_r); // hat_Qu
            Ku_ -= (Qyx.transpose() * Sinv_Y_Qyu).transpose(); // hat_Qxu
            hat_Quu += (Qyu.transpose() * Sinv_Y_Qyu);
        }
        
        if (is_ec_active[k]) {
            Eigen::Ref<Eigen::VectorXd> r = R[k];
            Eigen::Ref<Eigen::VectorXd> z = Z[k];
            Eigen::Ref<Eigen::VectorXd> ec_v = EC[k];

            Eigen::Ref<Eigen::MatrixXd> Qzx = ecx_all[k];
            Eigen::Ref<Eigen::MatrixXd> Qzu = ecu_all[k];

            const int dim_ec = ocp.getDimEC(k);

            Qx += Qzx.transpose() * z;
            Qu += Qzu.transpose() * z;

            rp_ec = ec_v + r;
            rd_ec = z + lambda[k] + (param.rho * r);
            r_ec = param.rho * rp_ec - rd_ec;

            rho_Qzx = param.rho * Qzx;
            rho_Qzu = param.rho * Qzu;

            du_ -= (Qzu.transpose() * r_ec); // hat_Qu
            Ku_ -= (Qzx.transpose() * rho_Qzx).transpose(); // hat_Qxu
            hat_Quu += (Qzu.transpose() * rho_Qzu);
        }
        
        Quu_llt.compute(hat_Quu.selfadjointView<Eigen::Upper>());
        if (Quu_llt.info() == Eigen::NumericalIssue) {
            backward_failed = true;
            break;
        }
        
        // Inplace Calculation
        Quu_llt.solveInPlace(du_);
        Quu_llt.solveInPlace(Ku_);
        
        // du_ = - Quu_llt.solve(hat_Qu);
        // Ku_ = - Quu_llt.solve(hat_Qxu.transpose());
        // du[k] = du_;
        // Ku[k] = Ku_;

        // dV(0) += du_.transpose() * Qu;
        // dV(1) += 0.5 * du_.transpose() * Quu * du_;
        
        Vx = Qx + (Ku_.transpose() * Qu) + (Ku_.transpose() * Quu * du_) + (Qxu * du_);
        Vxx = Qxx + (Ku_.transpose() * Qxu.transpose()) + (Qxu * Ku_) + (Ku_.transpose() * Quu * Ku_);
        

        opterror = std::max({Qu.lpNorm<Eigen::Infinity>(), opterror});

        // Inequality Constraint
        if (is_c_active[k]) {
            Eigen::Ref<Eigen::VectorXd> s = S[k];
            Eigen::Ref<Eigen::VectorXd> y = Y[k];
            Eigen::Ref<Eigen::VectorXd> c_v = C[k];
        
            Eigen::Ref<Eigen::MatrixXd> Qyx = cx_all[k];
            Eigen::Ref<Eigen::MatrixXd> Qyu = cu_all[k];
        
            // const int dim_g = ocp.getDimG(k);
            // const int dim_c = ocp.getDimC(k);
        
            Eigen::Ref<Eigen::VectorXd> ds_ = ds[k];
            Eigen::Ref<Eigen::MatrixXd> Ks_ = Ks[k];
            Eigen::Ref<Eigen::VectorXd> dy_ = dy[k];
            Eigen::Ref<Eigen::MatrixXd> Ky_ = Ky[k];

            ds_ = - (rp_c + Qyu * du_);
            Ks_ = - (Qyx + Qyu * Ku_);    

            // more complex, but more fast
            dy_ = Sinv_r + (Sinv_Y_Qyu * du_);
            Ky_ = Sinv_Y_Qyx + (Sinv_Y_Qyu * Ku_);

            /* less complex, but more slow
            Eigen::VectorXd rd_plus_Y_ds(dim_c);
            rd_plus_Y_ds.head(dim_g) = y.head(dim_g).cwiseProduct(ds_.head(dim_g));
            const auto& dim_hs     = ocp.getDimHs(k);
            const auto& dim_hs_top  = ocp.getDimHsTop(k);

            for (int i = 0; i < dim_hs.size(); ++i) {
                const int d   = dim_hs[i];
                const int idx = dim_hs_top[i];
                soc_helper::LtimesVec(rd_plus_Y_ds.segment(idx, d), y.segment(idx, d), ds_.segment(idx, d));
            }
            rd_plus_Y_ds += rd_c;

            dy_.head(dim_g) = rd_plus_Y_ds.head(dim_g).cwiseQuotient(s.head(dim_g));
            for (int i = 0; i < dim_hs.size(); ++i) {
                const int d   = dim_hs[i];
                const int idx = dim_hs_top[i];
                soc_helper::LinvTimesVec(dy_.segment(idx, d), s.segment(idx, d), rd_plus_Y_ds.segment(idx, d));
            }
            dy_ = -dy_;

            Ky_.topRows(dim_g) = Ks_.topRows(dim_g).array().colwise() * (y.head(dim_g).array() / s.head(dim_g).array());
            Eigen::VectorXd Y_Ks_h(ocp.getDimHsMax(k));
            for (int i = 0; i < dim_hs.size(); ++i) {
                const int d   = dim_hs[i];
                const int idx = dim_hs_top[i];
                for (int j = 0; j < ocp.getDimX(k); ++j) {
                    soc_helper::LtimesVec(Y_Ks_h, y.segment(idx, d), Ks_.block(idx, j, d, 1));
                    soc_helper::LinvTimesVec(Ky_.block(idx, j, d, 1), s.segment(idx, d), Y_Ks_h);
                }
            }
            Ky_ = -Ky_;
            */

            // CHECK: New Value Decrement
            // dV(0) += dy_.transpose() * c_v;
            // dV(1) += du_.transpose() * Qyu.transpose() * dy_;

            Vx += (Ky_.transpose() * c_v) + (Qyx.transpose() * dy_) + (Ku_.transpose() * Qyu.transpose() * dy_) + (Ky_.transpose() * Qyu * du_);
            Vxx += (Qyx.transpose() * Ky_) + (Ky_.transpose() * Qyx) + (Ku_.transpose() * Qyu.transpose() * Ky_) + (Ky_.transpose() * Qyu * Ku_);

            // with slack
            // Eigen::VectorXd Qo = Sinv * rd_c;
            // Eigen::VectorXd Qy = rp_c;
            // Eigen::MatrixXd I_c = Eigen::VectorXd::Ones(dim_c).asDiagonal();
            // dV(0) += Qo.transpose() * ds_;
            // dV(0) += Qy.transpose() * dy_;
            // dV(1) += dy_.transpose() * Qyu * du_;
            // dV(1) += dy_.transpose() * I_c * ds_; // Qyy = 0

            // Vx += Ky_.transpose() * Qy + Ku_.transpose() * Qyu.transpose() * dy_ + Ky_.transpose() * Qyu * du_ + Qyx.transpose() * dy_;
            // Vx += Ks_.transpose() * Qo + Ky_.transpose() * I_c * ds_ + Ks_.transpose() * I_c * dy_;

            // Vxx += Qyx.transpose() * Ky_ + Ky_.transpose() * Qyx + Ky_.transpose() * Qyu * Ku_ + Ku_.transpose() * Qyu.transpose() * Ky_;
            // Vxx += Ky_.transpose() * I_c * Ks_ + Ks_.transpose() * I_c * Ky_;

            // dy[k] = dy_;
            // Ky[k] = Ky_;
            // ds[k] = ds_;
            // Ks[k] = Ks_;

            opterror = std::max({rp_c.lpNorm<Eigen::Infinity>(), rd_c.lpNorm<Eigen::Infinity>(), opterror});
            opterror_rp_c = std::max({rp_c.lpNorm<Eigen::Infinity>(), opterror_rp_c});
            opterror_rd_c = std::max({rd_c.lpNorm<Eigen::Infinity>(), opterror_rd_c});
        }

        if (is_ec_active[k]) {
            Eigen::Ref<Eigen::VectorXd> r = R[k];
            Eigen::Ref<Eigen::VectorXd> z = Z[k];
            Eigen::Ref<Eigen::VectorXd> ec_v = EC[k];

            Eigen::Ref<Eigen::MatrixXd> Qzx = ecx_all[k];
            Eigen::Ref<Eigen::MatrixXd> Qzu = ecu_all[k];

            Eigen::Ref<Eigen::VectorXd> dr_ = dr[k];
            Eigen::Ref<Eigen::MatrixXd> Kr_ = Kr[k];
            Eigen::Ref<Eigen::VectorXd> dz_ = dz[k];
            Eigen::Ref<Eigen::MatrixXd> Kz_ = Kz[k];

            dr_ = - (rp_ec + Qzu * du_);
            Kr_ = - (Qzx + Qzu * Ku_);

            dz_ = r_ec + rho_Qzu * dr_;
            Kz_ = rho_Qzx + (rho_Qzu * Ku_);

            Vx += (Kz_.transpose() * ec_v) + (Qzx.transpose() * dz_) + (Ku_.transpose() * Qzu.transpose() * dz_) + (Kz_.transpose() * Qzu * du_);
            Vxx += (Qzx.transpose() * Kz_) + (Kz_.transpose() * Qzx) + (Ku_.transpose() * Qzu.transpose() * Kz_) + (Kz_.transpose() * Qzu * Ku_);

            opterror = std::max({rp_ec.lpNorm<Eigen::Infinity>(), rd_ec.lpNorm<Eigen::Infinity>(), opterror});
            opterror_rp_ec = std::max({rp_ec.lpNorm<Eigen::Infinity>(), opterror_rp_ec});
            opterror_rd_ec = std::max({rd_ec.lpNorm<Eigen::Infinity>(), opterror_rd_ec});
        }
    }
    // std::cout << "opterror_rpT_ec = " << opterror_rpT_ec << std::endl;
    // std::cout << "opterror_rdT_ec = " << opterror_rdT_ec << std::endl;
    // std::cout << "opterror_rpT_c = " << opterror_rpT_c << std::endl;
    // std::cout << "opterror_rdT_c = " << opterror_rdT_c << std::endl;
    // std::cout << "opterror_rp_c = " << opterror_rp_c << std::endl;
    // std::cout << "opterror_rd_c = " << opterror_rd_c << std::endl;
}

template <typename Scalar>
void ALIPDDP<Scalar>::checkRegulate() {
    if (forward_failed || backward_failed) {++regulate;}
    // else if (step <= 3) {regulate = regulate;}
    // else {--regulate;}

    if (regulate < 0) {regulate = 0;}
    else if (param.max_regularization < regulate) {regulate = param.max_regularization;}
}

template <typename Scalar>
Eigen::VectorXd ALIPDDP<Scalar>::perturb(const int& k, const Eigen::VectorXd& xn, const Eigen::VectorXd& x) {
    if (this->param.is_quaternion_in_state) {
        int dim_;
        if (k == N) {dim_ = dim_rnT;}
        else {dim_ = dim_rn[k];}
        Eigen::VectorXd dx(dim_);

        const Eigen::Vector4d& q = x.segment(param.quaternion_idx, 4);
        const Eigen::Vector4d& qn = xn.segment(param.quaternion_idx, 4);
    
        Eigen::Vector4d q_rel = quaternion_helper::Lq(q).transpose() * qn;

        dx.head(param.quaternion_idx) = xn.head(param.quaternion_idx) - x.head(param.quaternion_idx);
        dx.segment(param.quaternion_idx, 3) = q_rel.segment(1,3) / q_rel(0);
        const int tail_len = dim_ - (param.quaternion_idx + 3);
        dx.tail(tail_len) = xn.tail(tail_len) - x.tail(tail_len);
        
        return dx;
    }
    else {
        return xn - x;
    }
}

template <typename Scalar>
void ALIPDDP<Scalar>::forwardPass() {
    Eigen::VectorXd dx;

    Eigen::VectorXd dxT;

    double tau = std::max(0.99, 1.0 - param.mu);
    // double tau = 0.9;
    double one_tau = 1.0 - tau;
    
    double cost_new = 0.0;
    double logcost_new = 0.0;
    double barriercost_new = 0.0;
    double barriercostT_new = 0.0;
    double alcost_new = 0.0;
    double alcostT_new = 0.0;
    double error_new = 0.0;

    double dV_act;
    double dV_exp;

    for (step = 0; step < this->param.max_step_iter; ++step) {

        forward_failed = 0;
        const double step_size = step_list[step];

        // dV_exp = -(step_size * dV(0) + step_size * step_size * dV(1));
        // CHECK: Using Expected Value Decrement -> For Early Termination
        // if (param.forward_early_termination) {
        //     if (error <= param.tolerance && dV_exp > 0) {
        //         forward_failed = 3; continue;
        //     }
        // }

        X_new[0] = X[0];
        for (int k = 0; k < N; ++k) {
            dx = perturb(k, X_new[k], X[k]);
            U_new[k].noalias() = U[k] + (step_size * du[k]) + Ku[k] * dx;
            X_new[k+1].noalias() = ocp.f(k, X_new[k], U_new[k]);
            if (is_c_active[k]) {
                const int dim_g = ocp.getDimG(k);
                const int dim_h = ocp.getDimH(k);
                S_new[k].noalias() = S[k] + (step_size * ds[k]) + Ks[k] * dx;
                Y_new[k].noalias() = Y[k] + (step_size * dy[k]) + Ky[k] * dx;
                if (dim_g) {
                    if (no_helper::isFractionToBoundary(S_new[k].head(dim_g), S[k].head(dim_g), one_tau)
                        || no_helper::isFractionToBoundary(Y_new[k].head(dim_g), Y[k].head(dim_g), one_tau)) {
                        forward_failed = 11; break;
                    }
                }
                if (dim_h) {
                    if (soc_helper::isFractionToBoundary(S_new[k], S[k], one_tau, ocp.getDimHs(k), ocp.getDimHsTop(k))
                        || soc_helper::isFractionToBoundary(Y_new[k], Y[k], one_tau, ocp.getDimHs(k), ocp.getDimHsTop(k))) {
                        forward_failed = 13; break;
                    }
                }
                if (forward_failed) {break;}
                
                C_new[k].noalias() = ocp.c(k, X_new[k], U_new[k]);
            }
            if (is_ec_active[k]) {
                R_new[k].noalias() = R[k] + (step_size * dr[k]) + Kr[k] * dx;
                Z_new[k].noalias() = Z[k] + (step_size * dz[k]) + Kz[k] * dx;

                EC_new[k].noalias() = ocp.ec(k, X_new[k], U_new[k]);
            }        
        }
        if (forward_failed) {continue;}
        
        dxT = perturb(N, X_new[N], X[N]);
        if (is_cT_active) {
            ST_new.noalias() = ST + (step_size * dsT) + KsT * dxT;
            YT_new.noalias() = YT + (step_size * dyT) + KyT * dxT;
            const int dim_gT = ocp.getDimGT();
            const int dim_hT = ocp.getDimHT();
            if (dim_gT) {
                if (no_helper::isFractionToBoundary(ST_new.head(dim_gT), ST.head(dim_gT), one_tau)
                    || no_helper::isFractionToBoundary(YT_new.head(dim_gT), YT.head(dim_gT), one_tau)) {
                    // forward_failed = 21; continue;
                    forward_failed = 21;
                }
            }
            if (dim_hT) {
                if (soc_helper::isFractionToBoundary(ST_new, ST, one_tau, ocp.getDimHTs(), ocp.getDimHTsTop())
                    || soc_helper::isFractionToBoundary(YT_new, YT, one_tau, ocp.getDimHTs(), ocp.getDimHTsTop())) {
                    // forward_failed = 23; break;
                    forward_failed = 23;
                }
            }
            if (forward_failed) {continue;}

            CT_new.noalias() = ocp.cT(X_new[N]);
        }

        if (is_ecT_active) {
            RT_new.noalias() = RT + (step_size * drT) + KrT * dxT;
            ZT_new.noalias() = ZT + (step_size * dzT) + KzT * dxT;

            ECT_new.noalias() = ocp.ecT(X_new[N]);
        }
        
        error_new = 0.0;
        for (int k = 0; k < N; ++k) {
            if (is_c_active[k]) {
                error_new += (C_new[k] + S_new[k]).array().abs().sum();
            }
            if (is_ec_active[k]) {
                error_new += (EC_new[k] + R_new[k]).array().abs().sum();
            }
        }
        if (is_ecT_active) {
            error_new += (ECT_new + RT_new).array().abs().sum();
        }
        if (is_cT_active) {
            error_new += (CT_new + ST_new).array().abs().sum();
        }
        error_new = std::max(param.tolerance, error_new);

        // Cost
        barriercost_new = 0.0;
        barriercostT_new = 0.0;
        alcost_new = 0.0;
        alcostT_new = 0.0;
        cost_new = calcTotalCost(X_new, U_new);

        for (int k = 0; k < N; ++k) {
            const int dim_g = ocp.getDimG(k);
            const int dim_h = ocp.getDimH(k);
            if (dim_g) {
                no_helper::addBarrierCost(barriercost_new, S_new[k].head(dim_g));
            }
            if (dim_h) {
                soc_helper::addBarrierCost(barriercost_new, S_new[k], ocp.getDimHs(k), ocp.getDimHsTop(k));
            }
        }
        const int dim_gT = ocp.getDimGT();
        const int dim_hT = ocp.getDimHT();
        if (dim_gT) {
            no_helper::addBarrierCost(barriercostT_new, ST_new.head(dim_gT));
        }
        if (dim_hT) {
            soc_helper::addBarrierCost(barriercostT_new, ST_new, ocp.getDimHTs(), ocp.getDimHTsTop());
        }

        for (int k = 0; k < N; ++k) {
            if (is_ec_active[k]) {
                alcost_new += lambda[k].transpose() * R_new[k] + 0.5 * param.rho * R_new[k].squaredNorm();
            }
        }
        if (is_ecT_active) {
            alcostT_new += lambdaT.transpose() * RT_new + 0.5 * param.rhoT * RT_new.squaredNorm();
        }
        logcost_new = cost_new - (param.mu * barriercost_new + param.muT * barriercostT_new) + (alcost_new + alcostT_new);
        if (std::isnan(logcost_new)) {forward_failed = 5; continue;}
        dV_act = logcost - logcost_new;
                
        // Error Decrement
        if (param.forward_filter == 0) {
            if (error < error_new) {forward_failed = 1; continue;}
            if (dV_act < 0.0) {forward_failed = 2; continue;}
        }
        if (param.forward_filter == 1) {
            if (error <= param.tolerance) {
                if (error < error_new) {forward_failed = 1; continue;}
            }
            if (error <= error_new) {forward_failed = 1;}
            if (forward_failed == 1) {
                if (dV_act < -(param.forward_cost_threshold * error)) {forward_failed = 2; continue;}
                else {forward_failed = 0;}
            }
        }

        // if (param.forward_early_termination) {
        //     if (error <= param.tolerance){
        //         // if (dV_act < 0.0) {forward_failed = 2; continue;}
        //         // if (dV_exp <= 0.0) {forward_failed = 3; continue;}
    
        //         if (dV_exp >= 0.0) {
        //             if (!(1e-4 * dV_exp < dV_act && dV_act < 10 * dV_exp)) {forward_failed = 4; continue;}
        //         }
        //     }
        // }
        
        if (!forward_failed) {break;}
    }

    if (!forward_failed) {
        cost = cost_new;
        logcost = logcost_new;
        error = error_new;
        std::swap(X, X_new);
        std::swap(U, U_new);
        if (is_c_active_all) {
            std::swap(S, S_new);
            std::swap(Y, Y_new);
            std::swap(C, C_new);
        }
        if (is_ec_active_all) {
            std::swap(R, R_new);
            std::swap(Z, Z_new);
            std::swap(EC, EC_new);
        }
        if (is_cT_active) {
            std::swap(ST, ST_new);
            std::swap(YT, YT_new);
            std::swap(CT, CT_new);
        if (is_ecT_active) {
            std::swap(RT, RT_new);
            std::swap(ZT, ZT_new);
            std::swap(ECT, ECT_new);
        }
    }
}

template <typename Scalar>
std::vector<Eigen::VectorXd> ALIPDDP<Scalar>::getResX() {
    return X;
}

template <typename Scalar>
std::vector<Eigen::VectorXd> ALIPDDP<Scalar>::getResU() {
    return U;
}

template <typename Scalar>
std::vector<double> ALIPDDP<Scalar>::getAllCost() {
    return all_cost;
}

template <typename Scalar>
void ALIPDDP<Scalar>::logPrint() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::setw(4) << iter
              << std::setw(4) << inner_iter
              << std::setw(3) << backward_failed
              << std::setw(3) << forward_failed
              << std::setw(7) << param.mu
              << std::setw(16) << param.rho
              << std::setw(25) << logcost
              << std::setw(22) << opterror
              << std::setw(18) << error
              << std::setw(4) << regulate
              << std::setw(7) << step_list[step]
              << std::setw(5) << update_counter << std::endl;
}