#pragma once

#include "types.h"

#include "dynamics/discrete_dynamics_base.h"
#include "cost/stage_cost_base.h"
#include "cost/terminal_cost_base.h"
#include "constraint/stage_constraint_base.h"
#include "constraint/terminal_constraint_base.h"

#include <memory>
#include <vector>
#include <array>

// Debug
#include <iostream>

template <typename Scalar>
class OptimalControlProblem {
public:
    OptimalControlProblem(int horizon_steps) : N(horizon_steps) {
        X0.resize(N+1);
        U0.resize(N);
        dynamics_seq.resize(N);
        cost_seq.resize(N);
        constraint_seq.resize(N);
        for(int i=0; i<3; ++i){    
            for(int k=0; k<N; ++k) {
                constraint_seq[k][i] = std::vector<std::shared_ptr<StageConstraintBase<Scalar>>>();
            }
            terminal_constraint[i] = std::vector<std::shared_ptr<TerminalConstraintBase<Scalar>>>();
        }

        is_x_initialized.resize(N+1, false);
        is_u_initialized.resize(N, false);
    }

    void setInitialState(int k, const Vector<Scalar>& x_init) {
        X0.at(k) = x_init;
        is_x_initialized.at(k) = true;
    }

    void setInitialControl(int k, const Vector<Scalar>& u_init) {
        U0.at(k) = u_init;
        is_u_initialized.at(k) = true;
    }

    void setStageDynamics(int k, std::shared_ptr<DiscreteDynamicsBase<Scalar>> dyn) {
        dynamics_seq.at(k) = dyn;
    }

    void setStageCost(int k, std::shared_ptr<StageCostBase<Scalar>> cost) {
        cost_seq.at(k) = cost;
    }
    
    void setTerminalCost(std::shared_ptr<TerminalCostBase<Scalar>> cost) {
        terminal_cost = cost;
    }

    void addStageConstraint(int k, std::shared_ptr<StageConstraintBase<Scalar>> constraint) {
        constraint_seq[k][static_cast<int>(constraint->getConstraintType())].push_back(constraint);
    }

    void addTerminalConstraint(std::shared_ptr<TerminalConstraintBase<Scalar>> constraint) {
        terminal_constraint[static_cast<int>(constraint->getConstraintType())].push_back(constraint);
    }

    void init() {
        dim_x.resize(N);
        dim_u.resize(N);
        for(int k = 0; k < N; ++k) {
            if (dynamics_seq[k]) {
                dim_x[k] = dynamics_seq[k]->getDimX();
                dim_u[k] = dynamics_seq[k]->getDimU();
            }
            else {
                throw std::runtime_error("OptimalControlProblem: dynamics_seq[" + std::to_string(k) + "] is not set.");
            }
        }
        dim_xT = dynamics_seq[N-1]->getDimX();

        initStageConic();
        initTerminalConic();
        initStageEquality();
        initTerminalEquality();
    }

// private:
    std::vector<std::shared_ptr<DiscreteDynamicsBase<Scalar>>> dynamics_seq;
    std::vector<std::shared_ptr<StageCostBase<Scalar>>> cost_seq;
    std::shared_ptr<TerminalCostBase<Scalar>> terminal_cost;
    std::vector<std::array<std::vector<std::shared_ptr<StageConstraintBase<Scalar>>>, 3>> constraint_seq;
    std::array<std::vector<std::shared_ptr<TerminalConstraintBase<Scalar>>>, 3> terminal_constraint;

    std::vector<Vector<Scalar>> X0;
    std::vector<Vector<Scalar>> U0;
    std::vector<bool> is_x_initialized;
    std::vector<bool> is_u_initialized;
    double T{};
    int N{};

    std::vector<int> dim_x;
    std::vector<int> dim_u;
    int dim_xT;
    std::vector<std::array<int, 2>> dim_c;
    std::array<int, 2> dim_cT;
    std::vector<int> dim_ec;
    int dim_ecT;

    // Add another Parser Class?
    // Weird to containt these in here, as this class is implemeted only for storage

    std::vector<std::function<Vector<Scalar>(const Vector<Scalar>&, const Vector<Scalar>&)>> c;
    std::vector<std::function<Matrix<Scalar>(const Vector<Scalar>&, const Vector<Scalar>&)>> cx;
    std::vector<std::function<Matrix<Scalar>(const Vector<Scalar>&, const Vector<Scalar>&)>> cu;
    std::function<Vector<Scalar>(const Vector<Scalar>&)> cT;
    std::function<Matrix<Scalar>(const Vector<Scalar>&)> cTx;
    std::vector<std::vector<int>> dim_hs; // Connic Constraint Dimension Stack
    std::vector<std::vector<int>> dim_hs_top; // Connic Constraint Head Stack
    std::vector<int> dim_hs_max; // Maximum Dimension of Connic Constraint
    std::vector<int> dim_hTs; // Connic Constraint Dimension Stack (Terminal)
    std::vector<int> dim_hTs_top; // Connic Constraint Head Stack (Terminal)
    int dim_hTs_max; // Maximum Dimension of Connic Constraint (Terminal)

    std::vector<std::function<Vector<Scalar>(const Vector<Scalar>&, const Vector<Scalar>&)>> ec;
    std::vector<std::function<Matrix<Scalar>(const Vector<Scalar>&, const Vector<Scalar>&)>> ecx;
    std::vector<std::function<Matrix<Scalar>(const Vector<Scalar>&, const Vector<Scalar>&)>> ecu;
    std::function<Vector<Scalar>(const Vector<Scalar>&)> ecT;
    std::function<Matrix<Scalar>(const Vector<Scalar>&)> ecTx;

    void initStageConic() {
        dim_c.resize(N, {0, 0});
        dim_hs.resize(N);
        dim_hs_top.resize(N);
        dim_hs_max.resize(N);
        c.resize(N);
        cx.resize(N);
        cu.resize(N);
        for (int k = 0; k < N; ++k) {
            int dim_hs_max_ = 0;
            for (int i = 0; i < 2; ++i) {
                for (const auto& constraint : constraint_seq[k][i]) {
                    dim_c[k][i] += constraint->getDimC();
                }
            }
    
            int dim_h_top = dim_c[k][static_cast<int>(ConstraintType::NO)];
            dim_hs_top[k].clear();
            dim_hs[k].clear();
            for (const auto& constraint : constraint_seq[k][static_cast<int>(ConstraintType::SOC)]) {
                const int dim_h = constraint->getDimC();
                dim_hs[k].push_back(dim_h);
                dim_hs_top[k].push_back(dim_h_top);
                dim_h_top += dim_h;
                if (dim_h > dim_hs_max_) {
                    dim_hs_max_ = dim_h;
                }
            }
            dim_hs_max[k] = dim_hs_max_;

            // CHECK: Actually lambda function is not needed. For loop is sufficient. 
            c[k] = [this,k](const Vector<Scalar>& x, const Vector<Scalar>& u) -> Vector<Scalar> {
                int total_dim = dim_c[k][static_cast<int>(ConstraintType::NO)] + dim_c[k][static_cast<int>(ConstraintType::SOC)];
                Vector<Scalar> c_total(total_dim);
                int offset = 0;
                for (const auto& constraint : constraint_seq[k][static_cast<int>(ConstraintType::NO)]) {
                    const int dim_g = constraint->getDimC();
                    c_total.segment(offset, dim_g) = constraint->c(x, u);
                    offset += dim_g;
                }
                for (const auto& constraint : constraint_seq[k][1]) {
                    const int dim_h = constraint->getDimC();
                    c_total.segment(offset, dim_h) = constraint->c(x, u);
                    offset += dim_h;
                }
                return c_total;
            };

            cx[k] = [this,k](const Vector<Scalar>& x, const Vector<Scalar>& u) -> Matrix<Scalar> {
                int total_dim = dim_c[k][static_cast<int>(ConstraintType::NO)] + dim_c[k][static_cast<int>(ConstraintType::SOC)];
                Matrix<Scalar> J(total_dim, dim_x[k]);
                int offset = 0;
                for(const auto& constraint : constraint_seq[k][0]) {
                    const int dim_g = constraint->getDimC();
                    J.middleRows(offset, dim_g) = constraint->cx(x,u);
                    offset += dim_g;
                }
                for(const auto& constraint : constraint_seq[k][1]) {
                    const int dim_h = constraint->getDimC();
                    J.middleRows(offset, dim_h) = constraint->cx(x,u);
                    offset += dim_h;
                }
                return J;
            };
    
            cu[k] = [this,k](const Vector<Scalar>& x, const Vector<Scalar>& u) -> Matrix<Scalar> {
                int total_dim = dim_c[k][static_cast<int>(ConstraintType::NO)] + dim_c[k][static_cast<int>(ConstraintType::SOC)];
                Matrix<Scalar> J(total_dim, dim_u[k]);
                int offset = 0;
                for(const auto& constraint : constraint_seq[k][0]) {
                    const int dim_g = constraint->getDimC();
                    J.middleRows(offset, dim_g) = constraint->cu(x,u);
                    offset += constraint->getDimC();
                }
                for(const auto& constraint : constraint_seq[k][1]) {
                    const int dim_h = constraint->getDimC();
                    J.middleRows(offset, dim_h) = constraint->cu(x,u);
                    offset += dim_h;
                }
                return J;
            };
    
        }
    }

    void initTerminalConic() {
        dim_cT = {0, 0};
        dim_hTs.clear();
        dim_hTs_top.clear();
        dim_hTs_max = 0;

        for (int i = 0; i < 2; ++i) {
            for (const auto& constraint : terminal_constraint[i]) {
                dim_cT[i] += constraint->getDimCT();
            }
        }

        int dim_hT_top = dim_cT[static_cast<int>(ConstraintType::NO)];
        for (const auto& constraint : terminal_constraint[static_cast<int>(ConstraintType::SOC)]) {
            const int dim_hT = constraint->getDimCT();
            dim_hTs.push_back(dim_hT);
            dim_hTs_top.push_back(dim_hT_top);
            dim_hT_top += dim_hT;
            if (dim_hT > dim_hTs_max) {
                dim_hTs_max = dim_hT;
            }
        }

        cT = [this](const Vector<Scalar>& x) -> Vector<Scalar> {
            int total_dim = dim_cT[static_cast<int>(ConstraintType::NO)] + dim_cT[static_cast<int>(ConstraintType::SOC)];
            Vector<Scalar> cT_total(total_dim);
            int offset = 0;
            for (const auto& constraint : terminal_constraint[0]) {
                const int dim_gT = constraint->getDimCT();
                cT_total.segment(offset, dim_gT) = constraint->cT(x);
                offset += dim_gT;
            }
            for (const auto& constraint : terminal_constraint[1]) {
                const int dim_hT = constraint->getDimCT();
                cT_total.segment(offset, dim_hT) = constraint->cT(x);
                offset += dim_hT;
            }
            return cT_total;
        };

        cTx = [this](const Vector<Scalar>& x) -> Matrix<Scalar> {
            int total_dim = dim_cT[static_cast<int>(ConstraintType::NO)] + dim_cT[static_cast<int>(ConstraintType::SOC)];
            Matrix<Scalar> J(total_dim, dim_xT);
            int offset = 0;

            for(const auto& constraint : terminal_constraint[0]) {
                const int dim_gT = constraint->getDimCT();
                J.middleRows(offset, dim_gT) = constraint->cTx(x);
                offset += dim_gT;
            }
            for(const auto& constraint : terminal_constraint[1]) {
                const int dim_hT = constraint->getDimCT();
                J.middleRows(offset, dim_hT) = constraint->cTx(x);
                offset += dim_hT;
            }
            return J;
        };
    }

    void initStageEquality() {
        dim_ec.resize(N, 0);
        ec.resize(N);
        ecx.resize(N);
        ecu.resize(N);
    
        for(int k=0; k<N; ++k) {
            for(const auto& constraint : constraint_seq[k][static_cast<int>(ConstraintType::EQ)]) {
                dim_ec[k] += constraint->getDimC();
            }
    
            ec[k] = [this,k](const Vector<Scalar>& x, const Vector<Scalar>& u) -> Vector<Scalar> {
                Vector<Scalar> ec_total(dim_ec[k]);
                int offset = 0;
                for(const auto& constraint : constraint_seq[k][static_cast<int>(ConstraintType::EQ)]) {
                    ec_total.segment(offset, constraint->getDimC()) = constraint->c(x,u);
                    offset += constraint->getDimC();
                }
                return ec_total;
            };

            ecx[k] = [this,k](const Vector<Scalar>& x, const Vector<Scalar>& u) -> Matrix<Scalar> {
                Matrix<Scalar> J(dim_ec[k], dim_x[k]);
                int offset = 0;
                for(const auto& constraint : constraint_seq[k][static_cast<int>(ConstraintType::EQ)]) {
                    J.middleRows(offset, constraint->getDimC()) = constraint->cx(x,u);
                    offset += constraint->getDimC();
                }
                return J;
            };
            
            ecu[k] = [this,k](const Vector<Scalar>& x, const Vector<Scalar>& u) -> Matrix<Scalar> {
                Matrix<Scalar> J(dim_ec[k], dim_u[k]);
                int offset = 0;
                for(const auto& constraint : constraint_seq[k][static_cast<int>(ConstraintType::EQ)]) {
                    J.middleRows(offset, constraint->getDimC()) = constraint->cu(x,u);
                    offset += constraint->getDimC();
                }
                return J;
            };
        }
    }

    void initTerminalEquality() {
        dim_ecT = 0;
    
        for(const auto& constraint : terminal_constraint[static_cast<int>(ConstraintType::EQ)]) {
            dim_ecT += constraint->getDimCT();
        }
    
        ecT = [this](const Vector<Scalar>& x) -> Vector<Scalar> {
            Vector<Scalar> ecT_total(dim_ecT);
            int offset = 0;
            for(const auto& constraint : terminal_constraint[static_cast<int>(ConstraintType::EQ)]) {
                ecT_total.segment(offset, constraint->getDimCT()) = constraint->cT(x);
                offset += constraint->getDimCT();
            }
            return ecT_total;
        };
        
        ecTx = [this](const Vector<Scalar>& x) -> Matrix<Scalar> {
            Matrix<Scalar> J(dim_ecT, dim_xT);
            int offset = 0;
            for(const auto& constraint : terminal_constraint[static_cast<int>(ConstraintType::EQ)]) {
                J.middleRows(offset, constraint->getDimCT()) = constraint->cTx(x);
                offset += constraint->getDimCT();
            }
            return J;
        };
    }
};
