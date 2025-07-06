#pragma once

#include "problem_maker/optimal_control_problem.h"

#include <numeric>

template <typename Scalar>
class ProblemParser {
private:
    std::shared_ptr<OptimalControlProblem<Scalar>> ocp;

    std::vector<int> dim_x;
    std::vector<int> dim_u;
    std::vector<int> dim_g;
    std::vector<int> dim_h;
    std::vector<int> dim_c;
    std::vector<int> dim_ec;
    int dim_xT;
    int dim_gT;
    int dim_hT;
    int dim_cT;
    int dim_ecT;

    std::vector<std::vector<int>> dim_gs; // NO Constraint Dimension Stack
    std::vector<std::vector<int>> dim_gs_top; // NO Constraint Head Stack
    std::vector<std::vector<int>> dim_hs; // SOC Constraint Dimension Stack
    std::vector<std::vector<int>> dim_hs_top; // SOC Constraint Head Stack
    std::vector<int> dim_hs_max; // Maximum Dimension of SOC Constraint
    std::vector<int> dim_gTs; // NO Constraint Dimension Stack (Terminal)
    std::vector<int> dim_hTs; // SOC Constraint Dimension Stack (Terminal)
    std::vector<int> dim_gTs_top; // NO Constraint Head Stack (Terminal)
    std::vector<int> dim_hTs_top; // SOC Constraint Head Stack (Terminal)
    int dim_hTs_max; // Maximum Dimension of SOC Constraint (Terminal)

    std::vector<std::vector<int>> dim_ecs; // EQ Constraint Dimension Stack
    std::vector<std::vector<int>> dim_ecs_top; // EQ Constraint Head Stack
    std::vector<int> dim_ecTs; // EQ Constraint Dimension Stack (Terminal)
    std::vector<int> dim_ecTs_top; // EQ Constraint Head Stack (Terminal)

public:
    ProblemParser(std::shared_ptr<OptimalControlProblem<Scalar>> ocp) : ocp(ocp) {
    }

    void init() {
        // x_initialized.resize(N+1, false);
        // u_initialized.resize(N, false);

        dim_x.resize(ocp->N);
        dim_u.resize(ocp->N);

        for (int k = 0; k < N; ++k) {
            auto dynamics = ocp->getDynamics(k);
            if (!dynamics) throw std::runtime_error("ProblemParser: dynamics_seq[" + std::to_string(k) + "] is not set.");
            dim_x[k] = dynamics->getDimX();
            dim_u[k] = dynamics->getDimU();
        }
        dim_xT = ocp->getDynamics(N-1)->getDimX();

        initStageConic();
        initTerminalConic();
        initStageEquality();
        initTerminalEquality();
    }

    void initStageConic() {
        const int N = ocp->getHorizon();

        dim_g.resize(N);
        dim_gs.resize(N);
        dim_gs_top.resize(N);
        for (int k = 0; k < N; ++k) {
            int dim_g_top = 0;
            dim_gs[k].clear();
            dim_gs_top[k].clear();
            for (const auto& constraint : ocp->getStageConstraints(k, ConstraintType::NO)) {
                const int dim_ = constraint->getDimC();
                dim_gs[k].push_back(dim_);
                dim_gs_top[k].push_back(dim_g_top);
                dim_g_top += dim_;
            }
            dim_g[k] = std::accumulate(dim_gs[k].begin(), dim_gs[k].end(), 0);
        }

        dim_h.resize(N);
        dim_c.resize(N);
        dim_hs.resize(N);
        dim_hs_top.resize(N);
        dim_hs_max.resize(N);
        for (int k = 0; k < N; ++k) {
            int max_h = 0, dim_h_top = dim_g[k];
            dim_hs[k].clear();
            dim_hs_top[k].clear();
            for (const auto& constraint : ocp->getStageConstraints(k, ConstraintType::SOC)) {
                const int dim_ = constraint->getDimC();
                dim_hs[k].push_back(dim_);
                dim_hs_top[k].push_back(dim_h_top);
                dim_h_top += dim_;
                if (dim_ > max_h) max_h = dim_;
            }
            dim_hs_max[k] = max_h;
            dim_h[k] = std::accumulate(dim_hs[k].begin(), dim_hs[k].end(), 0);
            dim_c[k] = dim_g[k] + dim_h[k];
        }
    }

    void initTerminalConic() {
        dim_gTs.clear();
        dim_gTs_top.clear();
        int dim_gT_top = 0;
        for (const auto& constraint : ocp->getTerminalConstraints(ConstraintType::NO)) {
            const int dim_ = constraint->getDimCT();
            dim_gTs.push_back(dim_);
            dim_gTs_top.push_back(dim_gT_top);
            dim_gT_top += dim_;
        }
        dim_gT = std::accumulate(dim_gTs.begin(), dim_gTs.end(), 0);

        dim_hTs.clear();
        dim_hTs_top.clear();
        dim_hTs_max = 0;
        int dim_hT_top = dim_gT;
        for (const auto& constraint : ocp->getTerminalConstraints(ConstraintType::SOC)) {
            const int dim_ = constraint->getDimCT();
            dim_hTs.push_back(dim_);
            dim_hTs_top.push_back(dim_hT_top);
            dim_hT_top += dim_;
            if (dim_ > dim_hTs_max) dim_hTs_max = dim_;
        }
        dim_hT = std::accumulate(dim_hTs.begin(), dim_hTs.end(), 0);
        dim_cT = dim_gT + dim_hT;
    }

    void initStageEquality() {
        const int N = ocp->getHorizon();
        dim_ec.resize(N, 0);
        dim_ecs.resize(N);
        dim_ecs_top.resize(N);
        for (int k = 0; k < N; ++k) {
            int dim_ec_top = 0;
            dim_ecs[k].clear();
            dim_ecs_top[k].clear();
            for (const auto& constraint : ocp->getStageConstraints(k, ConstraintType::EQ)) {
                const int dim_ = constraint->getDimC();
                dim_ecs[k].push_back(dim_);
                dim_ecs_top[k].push_back(dim_ec_top);
                dim_ec_top += dim_;
            }
            dim_ec[k] = std::accumulate(dim_ecs[k].begin(), dim_ecs[k].end(), 0);
        }
    }

    void initTerminalEquality() {
        dim_ecTs.clear();
        dim_ecTs_top.clear();
        int dim_ecT_top = 0;
        for (const auto& constraint : ocp->getTerminalConstraints(ConstraintType::EQ)) {
            const int dim_ = constraint->getDimC();
            dim_ecTs.push_back(dim_);
            dim_ecTs_top.push_back(dim_ecT_top);
            dim_ecT_top += dim_;
        }
        dim_ecT = std::accumulate(dim_ecTs.begin(), dim_ecTs.end(), 0);
    }

    int getHorizon() const {return ocp->getHorizon();}
    bool isXinit(int k) const {return ocp->getInitialState(k).size()>0;}
    bool isUinit(int k) const {return ocp->getInitialControl(k).size()>0;}
    const Vector<Scalar>& getXinit(int k) const {return ocp->getInitialState(k);}
    const Vector<Scalar>& getUinit(int k) const {return ocp->getInitialControl(k);}
    int getDimX(const int& k) const {return dim_x[k];}
    int getDimXT(const int& k) const {return dim_xT;}
    int getDimU(const int& k) const {return dim_u[k];}
    int getDimG(const int& k) const {return dim_g[k];}
    int getDimH(const int& k) const {return dim_h[k];}
    int getDimC(const int& k) const {return dim_c[k];}
    int getDimEC(const int& k) const {return dim_ec[k];}

    const std::vector<int>& getDimGs(int k) const {return dim_gs[k];}
    const std::vector<int>& getDimGsTop(int k) const {return dim_gs_top[k];}
    const std::vector<int>& getDimHs(int k) const {return dim_hs[k];}
    const std::vector<int>& getDimHsTop(int k) const {return dim_hs_top[k];}
    int getDimHsMax(int k) const {return dim_hs_max[k];}
    const std::vector<int>& getDimGTs() const {return dim_gTs;}
    const std::vector<int>& getDimGTsTop() const {return dim_gTs_top;}
    const std::vector<int>& getDimHTs() const {return dim_hTs;}
    const std::vector<int>& getDimHTsTop() const {return dim_hTs_top;}
    int getDimHTsMax() const {return dim_hTs_max;}
    const std::vector<int>& getDimECs(int k) const {return dim_ecs[k];}
    const std::vector<int>& getDimECsTop(int k) const {return dim_ecs_top[k];}
    const std::vector<int>& getDimECTs() const {return dim_ecTs;}
    const std::vector<int>& getDimECTsTop() const {return dim_ecTs_top;}

    Vector<Scalar> f(int k, const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        return ocp->getDynamics(k)->f(x, u);
    }
    Matrix<Scalar> fx(int k, const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        return ocp->getDynamics(k)->fx(x, u);
    }
    Matrix<Scalar> fu(int k, const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        return ocp->getDynamics(k)->fu(x, u);
    }
    Scalar q(int k, const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        return ocp->getStageCost(k)->q(x, u);
    }
    Vector<Scalar> qx(int k, const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        return ocp->getStageCost(k)->qx(x, u);
    }
    Vector<Scalar> qu(int k, const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        return ocp->getStageCost(k)->qu(x, u);
    }
    Matrix<Scalar> qxx(int k, const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        return ocp->getStageCost(k)->qxx(x, u);
    }
    Matrix<Scalar> qxu(int k, const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        return ocp->getStageCost(k)->qxu(x, u);
    }
    Matrix<Scalar> quu(int k, const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        return ocp->getStageCost(k)->quu(x, u);
    }
    Scalar p(const Vector<Scalar>& x) const {
        return ocp->getTerminalCost()->p(x);
    }
    Vector<Scalar> px(const Vector<Scalar>& x) const {
        return ocp->getTerminalCost()->px(x);
    }
    Vector<Scalar> pxx(const Vector<Scalar>& x) const {
        return ocp->getTerminalCost()->pxx(x);
    }

    Vector<Scalar> c(int k, const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        Vector<Scalar> c_n(dim_c[k]);
        for (int i = 0; i < dim_gs[k].size(); ++i) {
            c_n.segment(dim_gs_top[k][i], dim_gs[k][i]) =
                ocp->getStageConstraints(k, ConstraintType::NO)[i]->c(x, u);
        }
        for (int i = 0; i < dim_hs[k].size(); ++i) {
            c_n.segment(dim_hs_top[k][i], dim_hs[k][i]) =
                ocp->getStageConstraints(k, ConstraintType::SOC)[i]->c(x, u);
        }
        return c_n;
    }
    
    Matrix<Scalar> cx(int k, const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        Matrix<Scalar> J(dim_c[k], dim_x[k]);
        for (int i = 0; i < dim_gs[k].size(); ++i) {
            J.middleRows(dim_gs_top[k][i], dim_gs[k][i]) =
                ocp->getStageConstraints(k, ConstraintType::NO)[i]->cx(x, u);
        }
        for (int i = 0; i < dim_hs[k].size(); ++i) {
            J.middleRows(dim_hs_top[k][i], dim_hs[k][i]) =
                ocp->getStageConstraints(k, ConstraintType::SOC)[i]->cx(x, u);
        }
        return J;
    }
    
    Matrix<Scalar> cu(int k, const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        Matrix<Scalar> J(dim_c[k], dim_u[k]);
        for (int i = 0; i < dim_gs[k].size(); ++i) {
            J.middleRows(dim_gs_top[k][i], dim_gs[k][i]) =
                ocp->getStageConstraints(k, ConstraintType::NO)[i]->cu(x, u);
        }
        for (int i = 0; i < dim_hs[k].size(); ++i) {
            J.middleRows(dim_hs_top[k][i], dim_hs[k][i]) =
                ocp->getStageConstraints(k, ConstraintType::SOC)[i]->cu(x, u);
        }
        return J;
    }
    
    Vector<Scalar> cT(const Vector<Scalar>& x) const {
        Vector<Scalar> c_n(dim_cT);
        for (int i = 0; i < dim_gTs.size(); ++i) {
            c_n.segment(dim_gTs_top[i], dim_gTs[i]) =
                ocp->getTerminalConstraints(ConstraintType::NO)[i]->cT(x);
        }
        for (int i = 0; i < dim_hTs.size(); ++i) {
            c_n.segment(dim_hTs_top[i], dim_hTs[i]) =
                ocp->getTerminalConstraints(ConstraintType::SOC)[i]->cT(x);
        }
        return c_n;
    }
    
    Matrix<Scalar> cTx(const Vector<Scalar>& x) const {
        Matrix<Scalar> J(dim_cT, dim_xT);
        for (int i = 0; i < dim_gTs.size(); ++i) {
            J.middleRows(dim_gTs_top[i], dim_gTs[i]) =
                ocp->getTerminalConstraints(ConstraintType::NO)[i]->cTx(x);
        }
        for (int i = 0; i < dim_hTs.size(); ++i) {
            J.middleRows(dim_hTs_top[i], dim_hTs[i]) =
                ocp->getTerminalConstraints(ConstraintType::SOC)[i]->cTx(x);
        }
        return J;
    }

    Vector<Scalar> ec(int k, const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        Vector<Scalar> ec_n(dim_ec[k]);
        for (int i = 0; i < dim_ecs[k].size(); ++i) {
            ec_n.segment(dim_ecs_top[k][i], dim_ecs[k][i]) =
                ocp->getStageConstraints(k, ConstraintType::EQ)[i]->c(x, u);
        }
        return ec_n;
    }
    
    Matrix<Scalar> ecx(int k, const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        Matrix<Scalar> J(dim_ec[k], dim_x[k]);
        for (int i = 0; i < dim_ecs[k].size(); ++i) {
            J.middleRows(dim_ecs_top[k][i], dim_ecs[k][i]) =
                ocp->getStageConstraints(k, ConstraintType::EQ)[i]->cx(x, u);
        }
        return J;
    }
    
    Matrix<Scalar> ecu(int k, const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        Matrix<Scalar> J(dim_ec[k], dim_u[k]);
        for (int i = 0; i < dim_ecs[k].size(); ++i) {
            J.middleRows(dim_ecs_top[k][i], dim_ecs[k][i]) =
                ocp->getStageConstraints(k, ConstraintType::EQ)[i]->cu(x, u);
        }
        return J;
    }
    
    Vector<Scalar> ecT(const Vector<Scalar>& x) const {
        Vector<Scalar> ec_n(dim_ecT);
        for (int i = 0; i < dim_ecTs.size(); ++i) {
            ec_n.segment(dim_ecTs_top[i], dim_ecTs[i]) =
                ocp->getTerminalConstraints(ConstraintType::EQ)[i]->cT(x);
        }
        return ec_n;
    }
    
    Matrix<Scalar> ecTx(const Vector<Scalar>& x) const {
        Matrix<Scalar> J(dim_ecT, dim_xT);
        for (int i = 0; i < dim_ecTs.size(); ++i) {
            J.middleRows(dim_ecTs_top[i], dim_ecTs[i]) =
                ocp->getTerminalConstraints(ConstraintType::EQ)[i]->cTx(x);
        }
        return J;
    }
};
