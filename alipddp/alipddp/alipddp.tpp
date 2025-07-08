
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
    
    ku.resize(N);
    kr.resize(N);
    kz.resize(N);
    ks.resize(N);
    ky.resize(N);
    Ku.resize(N);
    Kr.resize(N);
    Kz.resize(N);
    Ks.resize(N);
    Ky.resize(N);
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
        ku[k] = Eigen::VectorXd::Zero(ocp.getDimU(k));
        Ku[k] = Eigen::MatrixXd::Zero(ocp.getDimU(k), dim_rn[k]);
        if (ocp.getDimC(k)) {
            ks[k] = Eigen::VectorXd::Zero(ocp.getDimC(k));
            ky[k] = Eigen::VectorXd::Zero(ocp.getDimC(k));
            Ks[k] = Eigen::MatrixXd::Zero(ocp.getDimC(k), dim_rn[k]);
            Ky[k] = Eigen::MatrixXd::Zero(ocp.getDimC(k), dim_rn[k]);
        }
        if (ocp.getDimEC(k)) {
            kr[k] = Eigen::VectorXd::Zero(ocp.getDimEC(k));
            kz[k] = Eigen::VectorXd::Zero(ocp.getDimEC(k));
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
        alcostT += (lambdaT.transpose() * RT) + (0.5 * param.rho * RT.squaredNorm());
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
                if (is_ecT_active && opterror_rpT_ec < param.tolerance && opterror_rdT_ec < param.tolerance) {
                    if (param.rho < param.rho_max) {updated = true;}
                    param.rho = std::min(param.rho_max, param.rho_mul * param.rho);
                    lambdaT = lambdaT + param.rho * RT;
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
            return;
        }

        bool mu_stop = (param.mu <= param.mu_min);
        bool muT_stop = (param.muT <= param.mu_min);
        bool rho_stop = (param.rho >= param.rho_max);

        bool c_done   = (!is_c_active_all || mu_stop);
        bool ec_done   = (!is_ec_active_all || rho_stop);
        bool cT_done  = (!is_cT_active || muT_stop);
        bool ecT_done = (!is_ecT_active || rho_stop);
      
        if (c_done && cT_done && ec_done && ecT_done) {
            std::cout << "Outer Max/Min" << std::endl;
            return;
        }

        // Update Outer Loop Parameters
        if (is_c_active_all) {param.mu = std::max(param.mu_min, std::min(param.mu_mul * param.mu, std::pow(param.mu, param.mu_exp)));}
        if (is_ec_active_all || is_ecT_active) {param.rho = std::min(param.rho_max, param.rho_mul * param.rho);}
        if (is_cT_active) {param.muT = std::max(param.mu_min, std::min(param.mu_mul * param.muT, std::pow(param.muT, param.mu_exp)));}
        if (is_ecT_active) {lambdaT = lambdaT + param.rho * RT;}
        resetFilter();
        resetRegulation();
    }
}

template <typename Scalar>
void ALIPDDP<Scalar>::calcAllDiff() {
    // CHECK: Multithreading (TODO: with CUDA)
    // CHECK 1: Making branch in for loop is fine for parallelization?
    // CHECK 2: Move to Model to make solver only consider Eigen (not autodiff::dual)

    // #pragma omp parallel for
    for (int k = 0; k < N; ++k) {
        const Eigen::VectorXd& x = X[k];
        const Eigen::VectorXd& u = U[k];

        fx_all[k] = ocp.fx(k, x, u);
        fu_all[k] = ocp.fu(k, x, u);
        qx_all[k] = ocp.qx(k, x, u);
        qu_all[k] = ocp.qu(k, x, u);
        qxx_all[k] = ocp.qxx(k, x, u);
        quu_all[k] = ocp.quu(k, x, u);
        qxu_all[k] = ocp.qxu(k, x, u);
        if (ocp.getDimC(k)) {
            cx_all[k] = ocp.cx(k, x, u);
            cu_all[k] = ocp.cu(k, x, u);
        }
        if (ocp.getDimEC(k)) {
            ecx_all[k] = ocp.ecx(k, x, u);
            ecu_all[k] = ocp.ecu(k, x, u);
        }
    }
    const Eigen::VectorXd& xT = X[N];

    px_all = ocp.px(xT);
    pxx_all = ocp.pxx(xT);

    if (ocp.getDimCT()) {
        cTx_all = ocp.cTx(xT);
    }
    if (ocp.getDimECT()) {
        ecTx_all = ocp.ecTx(xT);
    }
}

template <typename Scalar>
void ALIPDDP<Scalar>::backwardPass() {
    Eigen::VectorXd Vx;
    Eigen::MatrixXd Vxx;

    Eigen::VectorXd Qx, Qu;
    Eigen::MatrixXd Qxx, Qxu, Quu;
    Eigen::MatrixXd hat_Quu;

    Eigen::VectorXd rp, rd;
    
    Eigen::VectorXd Sinv_r;
    Eigen::MatrixXd Sinv_Y_Qyx, Sinv_Y_Qyu;
    
    Eigen::LLT<Eigen::MatrixXd> Quu_llt;

    opterror = 0.0;
    opterror_rpT_ec = 0.0;
    opterror_rdT_ec = 0.0;
    opterror_rpT_c = 0.0;
    opterror_rdT_c = 0.0;
    opterror_rp_c = 0.0;
    opterror_rd_c = 0.0;

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

        Eigen::VectorXd rpT = CT + ST;
        Eigen::VectorXd rdT(dim_cT);
        rdT.head(dim_gT) = ST.head(dim_gT).cwiseProduct(YT.head(dim_gT));
        for (int i = 0; i < dim_hTs.size(); ++i) {
            const int d   = dim_hTs[i];
            const int idx = dim_hTs_top[i];
            soc_helper::LtimesVec(rdT.segment(idx, d), YT.segment(idx, d), ST.segment(idx, d));
        }
        rdT -= param.muT * eT;

        Eigen::VectorXd rT(dim_cT);
        rT.head(dim_gT) = YT.head(dim_gT).cwiseProduct(rpT.head(dim_gT));
        for (int i = 0; i < dim_hTs.size(); ++i) {
            const int d   = dim_hTs[i];
            const int idx = dim_hTs_top[i];
            soc_helper::LtimesVec(rT.segment(idx, d), YT.segment(idx, d), rpT.segment(idx, d));
        }
        rT -= rdT;

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

        ksT = - rpT;
        KsT = - QyxT;
        
        kyT = Sinv_rT;
        KyT = Sinv_Y_QyxT;

        Vx += KyT.transpose() * CT + QyxT.transpose() * kyT;
        Vxx += QyxT.transpose() * KyT + KyT.transpose() * QyxT;

        // with slack
        // Eigen::VectorXd QsT = YTinv * rdT;
        // Eigen::VectorXd QyT = rpT;
        // Eigen::MatrixXd I_cT = Eigen::VectorXd::Ones(dim_cT).asDiagonal();
        // dV(0) += QsT.transpose() * ksT;
        // dV(0) += QyT.transpose() * kyT;
        // dV(1) += kyT.transpose() * I_cT * ksT; 

        // Vx += KyT.transpose() * QyT + QyxT.transpose() * kyT;
        // Vx += KsT.transpose() * QsT + KyT.transpose() * I_cT * ksT + KsT.transpose() * I_cT * kyT;

        // Vxx += QyxT.transpose() * KyT + KyT.transpose() * QyxT;
        // Vxx += KyT.transpose() * I_cT * KsT + KsT.transpose() * I_cT * KyT;

        opterror = std::max({rpT.lpNorm<Eigen::Infinity>(), rdT.lpNorm<Eigen::Infinity>(), opterror});
        opterror_rpT_c = std::max({rpT.lpNorm<Eigen::Infinity>(), opterror_rpT_c});
        opterror_rdT_c = std::max({rdT.lpNorm<Eigen::Infinity>(), opterror_rdT_c});
    }

    // Equality Terminal Constraint
    if (is_ecT_active) {
        // const int dim_ecT = ocp.getDimECT();

        Eigen::Ref<const Eigen::MatrixXd> QzxT = ecTx_all;

        Eigen::VectorXd rpT = ECT + RT;
        Eigen::VectorXd rdT = ZT + lambdaT + (param.rho * RT);
        
        krT = - rpT;
        KrT = - QzxT;
        kzT = param.rho * rpT - rdT;
        KzT = param.rho * QzxT;

        // CHECK: New Value Decrement
        // dV(0) += kzT.transpose() * ECT;

        Vx += KzT.transpose() * ECT + QzxT.transpose() * kzT;
        Vxx += QzxT.transpose() * KzT + KzT.transpose() * QzxT;

        // with slack
        // Eigen::VectorXd QrT = rdT;
        // Eigen::VectorXd QzT = rpT;
        // Eigen::MatrixXd I_ecT = Eigen::VectorXd::Ones(dim_ecT).asDiagonal();
        // dV(0) += QrT.transpose() * krT;
        // dV(0) += QzT.transpose() * kzT;
        // dV(1) += kzT.transpose() * I_ecT * krT; // Qrr = 0
        
        // Vx += KzT.transpose() * QzT + QzxT.transpose() * kzT;
        // Vx += KrT.transpose() * QrT + KzT.transpose() * I_ecT * krT + KrT.transpose() * I_ecT * kzT;
        
        // Vxx += QzxT.transpose() * KzT + KzT.transpose() * QzxT;
        // Vxx += KzT.transpose() * I_ecT * KrT + KrT.transpose() * I_ecT * KzT;
        
        opterror = std::max({rpT.lpNorm<Eigen::Infinity>(), rdT.lpNorm<Eigen::Infinity>(), opterror});
        opterror_rpT_ec = std::max({rpT.lpNorm<Eigen::Infinity>(), opterror_rpT_ec});
        opterror_rdT_ec = std::max({rdT.lpNorm<Eigen::Infinity>(), opterror_rdT_ec});
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

        Eigen::Ref<Eigen::VectorXd> ku_ = ku[k];
        Eigen::Ref<Eigen::MatrixXd> Ku_ = Ku[k];

        ku_ = - Qu; // hat_Qu
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
            
            rp = c_v + s;

            rd.resize(dim_c);
            rd.head(dim_g) = s.head(dim_g).cwiseProduct(y.head(dim_g));
            for (int i = 0; i < dim_hs.size(); ++i) {
                const int idx = dim_hs_top[i];
                const int n   = dim_hs[i];
                soc_helper::LtimesVec(rd.segment(idx, n), y.segment(idx, n), s.segment(idx, n));
            }
            rd -= param.mu * e[k];

            Eigen::VectorXd r(dim_c);
            r.head(dim_g) = y.head(dim_g).cwiseProduct(rp.head(dim_g));
            for (int i = 0; i < dim_hs.size(); ++i) {
                const int d   = dim_hs[i];
                const int idx = dim_hs_top[i];
                soc_helper::LtimesVec(r.segment(idx, d), y.segment(idx, d), rp.segment(idx, d));
            }
            r -= rd;

            Sinv_r.resize(dim_c);
            Sinv_r.head(dim_g) = r.head(dim_g).cwiseQuotient(s.head(dim_g));
            for (int i = 0; i < dim_hs.size(); ++i) {
                const int d   = dim_hs[i];
                const int idx = dim_hs_top[i];
                soc_helper::LinvTimesVec(Sinv_r.segment(idx, d), s.segment(idx, d), r.segment(idx, d));
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
            ku_ -= (Qyu.transpose() * Sinv_r); // hat_Qu
            Ku_ -= (Qyx.transpose() * Sinv_Y_Qyu).transpose(); // hat_Qxu
            hat_Quu += (Qyu.transpose() * Sinv_Y_Qyu);
        }
        
        // TODO
        // Equality Constraint
        // if (is_ec_active[k]) {
        // }
        
        Quu_llt.compute(hat_Quu.selfadjointView<Eigen::Upper>());
        if (Quu_llt.info() == Eigen::NumericalIssue) {
            backward_failed = true;
            break;
        }
        
        // ku_ = - Quu_llt.solve(hat_Qu);
        // Ku_ = - Quu_llt.solve(hat_Qxu.transpose());

        // Inplace Calculation
        Quu_llt.solveInPlace(ku_);
        Quu_llt.solveInPlace(Ku_);

        // dV(0) += ku_.transpose() * Qu;
        // dV(1) += 0.5 * ku_.transpose() * Quu * ku_;
        
        Vx = Qx + (Ku_.transpose() * Qu) + (Ku_.transpose() * Quu * ku_) + (Qxu * ku_);
        Vxx = Qxx + (Ku_.transpose() * Qxu.transpose()) + (Qxu * Ku_) + (Ku_.transpose() * Quu * Ku_);
        
        // ku[k] = ku_;
        // Ku[k] = Ku_;

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
        
            Eigen::Ref<Eigen::VectorXd> ks_ = ks[k];
            Eigen::Ref<Eigen::MatrixXd> Ks_ = Ks[k];
            Eigen::Ref<Eigen::VectorXd> ky_ = ky[k];
            Eigen::Ref<Eigen::MatrixXd> Ky_ = Ky[k];

            ks_ = - (rp + Qyu * ku_);
            Ks_ = - (Qyx + Qyu * Ku_);    

            // more complex, but more fast
            ky_ = Sinv_r + (Sinv_Y_Qyu * ku_);
            Ky_ = Sinv_Y_Qyx + (Sinv_Y_Qyu * Ku_);

            /* less complex, but more slow
            Eigen::VectorXd rd_plus_Y_ds(dim_c);
            rd_plus_Y_ds.head(dim_g) = y.head(dim_g).cwiseProduct(ks_.head(dim_g));
            const auto& dim_hs     = ocp.getDimHs(k);
            const auto& dim_hs_top  = ocp.getDimHsTop(k);

            for (int i = 0; i < dim_hs.size(); ++i) {
                const int d   = dim_hs[i];
                const int idx = dim_hs_top[i];
                soc_helper::LtimesVec(rd_plus_Y_ds.segment(idx, d), y.segment(idx, d), ks_.segment(idx, d));
            }
            rd_plus_Y_ds += rd;

            ky_.head(dim_g) = rd_plus_Y_ds.head(dim_g).cwiseQuotient(s.head(dim_g));
            for (int i = 0; i < dim_hs.size(); ++i) {
                const int d   = dim_hs[i];
                const int idx = dim_hs_top[i];
                soc_helper::LinvTimesVec(ky_.segment(idx, d), s.segment(idx, d), rd_plus_Y_ds.segment(idx, d));
            }
            ky_ = -ky_;

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
            // dV(0) += ky_.transpose() * c_v;
            // dV(1) += ku_.transpose() * Qyu.transpose() * ky_;

            Vx += (Ky_.transpose() * c_v) + (Qyx.transpose() * ky_) + (Ku_.transpose() * Qyu.transpose() * ky_) + (Ky_.transpose() * Qyu * ku_);
            Vxx += (Qyx.transpose() * Ky_) + (Ky_.transpose() * Qyx) + (Ku_.transpose() * Qyu.transpose() * Ky_) + (Ky_.transpose() * Qyu * Ku_);

            // with slack
            // Eigen::VectorXd Qo = Sinv * rd;
            // Eigen::VectorXd Qy = rp;
            // Eigen::MatrixXd I_c = Eigen::VectorXd::Ones(dim_c).asDiagonal();
            // dV(0) += Qo.transpose() * ks_;
            // dV(0) += Qy.transpose() * ky_;
            // dV(1) += ky_.transpose() * Qyu * ku_;
            // dV(1) += ky_.transpose() * I_c * ks_; // Qyy = 0

            // Vx += Ky_.transpose() * Qy + Ku_.transpose() * Qyu.transpose() * ky_ + Ky_.transpose() * Qyu * ku_ + Qyx.transpose() * ky_;
            // Vx += Ks_.transpose() * Qo + Ky_.transpose() * I_c * ks_ + Ks_.transpose() * I_c * ky_;

            // Vxx += Qyx.transpose() * Ky_ + Ky_.transpose() * Qyx + Ky_.transpose() * Qyu * Ku_ + Ku_.transpose() * Qyu.transpose() * Ky_;
            // Vxx += Ky_.transpose() * I_c * Ks_ + Ks_.transpose() * I_c * Ky_;

            // ky[k] = ky_;
            // Ky[k] = Ky_;
            // ks[k] = ks_;
            // Ks[k] = Ks_;

            opterror = std::max({rp.lpNorm<Eigen::Infinity>(), rd.lpNorm<Eigen::Infinity>(), opterror});
            opterror_rp_c = std::max({rp.lpNorm<Eigen::Infinity>(), opterror_rp_c});
            opterror_rd_c = std::max({rd.lpNorm<Eigen::Infinity>(), opterror_rd_c});
        }

        // TODO
        // Equality Constraint
        // if (is_ec_active[k]) {

        // }
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
        Eigen::VectorXd dx;
        // Eigen::VectorXd dx(dim_rn[k]);
        if (k == N) {dx.resize(dim_rnT);}
        else {dx.resize(dim_rn[k]);}

        const Eigen::Vector4d& q = x.segment(param.quaternion_idx, param.quaternion_dim);
        const Eigen::Vector4d& qn = xn.segment(param.quaternion_idx, param.quaternion_dim);
    
        Eigen::Vector4d q_rel = quaternion_helper::Lq(q).transpose() * qn;

        dx.head(param.quaternion_idx) = xn.head(param.quaternion_idx) - x.head(param.quaternion_idx);
        dx.segment(param.quaternion_idx, 3) = q_rel.segment(1,3) / q_rel(0);
        const int tail_len = dim_rn[k] - (param.quaternion_idx + param.quaternion_dim);
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
    std::vector<Eigen::VectorXd> X_new(N+1);
    std::vector<Eigen::VectorXd> U_new(N);
    std::vector<Eigen::VectorXd> S_new(N);
    std::vector<Eigen::VectorXd> Y_new(N);
    std::vector<Eigen::VectorXd> C_new(N);
    std::vector<Eigen::VectorXd> R_new(N);
    std::vector<Eigen::VectorXd> Z_new(N);
    std::vector<Eigen::VectorXd> EC_new(N);

    Eigen::VectorXd dxT;
    Eigen::VectorXd ST_new;
    Eigen::VectorXd YT_new;
    Eigen::VectorXd CT_new;
    Eigen::VectorXd RT_new;
    Eigen::VectorXd ZT_new;
    Eigen::VectorXd ECT_new;

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
        if (param.forward_early_termination) {
            if (error <= param.tolerance && dV_exp > 0) {
                forward_failed = 3; continue;
            }
        }

        X_new[0] = X[0];
        for (int k = 0; k < N; ++k) {
            dx = perturb(k, X_new[k], X[k]);
            U_new[k] = U[k] + (step_size * ku[k]) + Ku[k] * dx;
            X_new[k+1] = ocp.f(k, X_new[k], U_new[k]);
            if (is_c_active[k]) {
                const int dim_g = ocp.getDimG(k);
                const int dim_h = ocp.getDimH(k);
                S_new[k] = S[k] + (step_size * ks[k]) + Ks[k] * dx;
                Y_new[k] = Y[k] + (step_size * ky[k]) + Ky[k] * dx;
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
                
                C_new[k] = ocp.c(k, X_new[k], U_new[k]);
            }
            if (is_ec_active[k]) {
                R_new[k] = R[k] + (step_size * kr[k]) + Kr[k] * dx;
                Z_new[k] = Z[k] + (step_size * kz[k]) + Kz[k] * dx;

                EC_new[k] = ocp.ec(k, X_new[k], U_new[k]);
            }        
        }
        if (forward_failed) {continue;}
        
        dxT = perturb(N, X_new[N], X[N]);
        if (is_cT_active) {
            ST_new = ST + (step_size * ksT) + KsT * dxT;
            YT_new = YT + (step_size * kyT) + KyT * dxT;
            const int dim_gT = ocp.getDimGT();
            const int dim_hT = ocp.getDimHT();
            if (dim_gT) {
                if (no_helper::isFractionToBoundary(ST_new.head(dim_gT), ST.head(dim_gT), one_tau)
                    || no_helper::isFractionToBoundary(YT_new.head(dim_gT), YT.head(dim_gT), one_tau)) {
                    forward_failed = 21; continue;
                }
            }
            if (dim_hT) {
                if (soc_helper::isFractionToBoundary(ST_new, ST, one_tau, ocp.getDimHTs(), ocp.getDimHTsTop())
                    || soc_helper::isFractionToBoundary(YT_new, YT, one_tau, ocp.getDimHTs(), ocp.getDimHTsTop())) {
                    forward_failed = 23; break;
                }
            }
            if (forward_failed) {continue;}

            CT_new = ocp.cT(X_new[N]);
        }

        if (is_ecT_active) {
            RT_new = RT + (step_size * krT) + KrT * dxT;
            ZT_new = ZT + (step_size * kzT) + KzT * dxT;

            ECT_new = ocp.ecT(X_new[N]);
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
            alcostT_new += lambdaT.transpose() * RT_new + 0.5 * param.rho * RT_new.squaredNorm();
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

        if (param.forward_early_termination) {
            if (error <= param.tolerance){
                // if (dV_act < 0.0) {forward_failed = 2; continue;}
                // if (dV_exp <= 0.0) {forward_failed = 3; continue;}
    
                if (dV_exp >= 0.0) {
                    if (!(1e-4 * dV_exp < dV_act && dV_act < 10 * dV_exp)) {forward_failed = 4; continue;}
                }
            }
        }
        
        if (!forward_failed) {break;}
    }

    if (!forward_failed) {
        cost = cost_new;
        logcost = logcost_new;
        error = error_new;
        X = std::move(X_new);
        U = std::move(U_new);
        if (is_c_active_all) {
            S = std::move(S_new);
            Y = std::move(Y_new);
            C = std::move(C_new);
        }
        if (is_ec_active_all) {
            R = std::move(R_new);
            Z = std::move(Z_new);
            EC = std::move(EC_new);
        }
        if (is_cT_active) {
            ST = std::move(ST_new);
            YT = std::move(YT_new);
            CT = std::move(CT_new);
        }
        if (is_ecT_active) {
            RT = std::move(RT_new);
            ZT = std::move(ZT_new);
            ECT = std::move(ECT_new);
        }
    }
    // else {std::cout<<"Forward Failed"<<std::endl;}
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