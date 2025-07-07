#pragma once

#include "types.h"

#include "dynamics/linear_discrete_dynamics.h"
#include "cost/quadratic_stage_cost.h"
#include "cost/quadratic_terminal_cost.h"
#include "constraint/linear_stage_constraint.h"
#include "constraint/linear_terminal_constraint.h"

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
    }

    void setInitialState(int k, const Vector<Scalar>& x_init) {
        X0[k] = x_init;
    }
    void setInitialState(const Vector<Scalar>& x_init) {
        for (int i = 0; i < N; ++i) {
            setInitialState(i, x_init);
        }
    }

    void setInitialControl(int k, const Vector<Scalar>& u_init) {
        U0[k] = u_init;
    }
    void setInitialControl(const Vector<Scalar>& u_init) {
        for (int i = 0; i < N; ++i) {
            setInitialControl(i, u_init);
        }
    }

    void setStageDynamics(int k, std::shared_ptr<DiscreteDynamicsBase<Scalar>> dyn) {
        dynamics_seq[k] = dyn;
    }
    void setStageDynamics(std::shared_ptr<DiscreteDynamicsBase<Scalar>> dyn) {
        for (int i = 0; i < N; ++i) {
            setStageDynamics(i, dyn);
        }
    }

    void setStageCost(int k, std::shared_ptr<StageCostBase<Scalar>> cost) {
        cost_seq[k] = cost;
    }
    void setStageCost(std::shared_ptr<StageCostBase<Scalar>> cost) {
        for (int i = 0; i < N; ++i) {
            setStageCost(i, cost);
        }
    }
    
    void setTerminalCost(std::shared_ptr<TerminalCostBase<Scalar>> cost) {
        terminal_cost = cost;
    }

    void addStageConstraint(int k, std::shared_ptr<StageConstraintBase<Scalar>> constraint) {
        constraint_seq[k][static_cast<int>(constraint->getConstraintType())].push_back(constraint);
    }
    void addStageConstraint(std::shared_ptr<StageConstraintBase<Scalar>> constraint) {
        for (int i = 0; i < N; ++i) {
            addStageConstraint(i, constraint);
        }
    }

    void addTerminalConstraint(std::shared_ptr<TerminalConstraintBase<Scalar>> constraint) {
        terminal_constraint[static_cast<int>(constraint->getConstraintType())].push_back(constraint);
    }

    int getHorizon() const {return N;}
    const Vector<Scalar>& getInitialState(int k) const {return X0[k];}
    const Vector<Scalar>& getInitialControl(int k) const {return U0[k];}
    const auto& getDynamics(int k) const {return dynamics_seq[k];}
    const auto& getStageCost(int k) const {return cost_seq[k];}
    const auto& getTerminalCost() const {return terminal_cost;}
    const auto& getStageConstraints(int k, ConstraintType type) const {return constraint_seq[k][static_cast<int>(type)];}
    const auto& getTerminalConstraints(ConstraintType type) const {return terminal_constraint[static_cast<int>(type)];}

private:
    const int N{};
    std::vector<Vector<Scalar>> X0;
    std::vector<Vector<Scalar>> U0;
    std::vector<std::shared_ptr<DiscreteDynamicsBase<Scalar>>> dynamics_seq;
    std::vector<std::shared_ptr<StageCostBase<Scalar>>> cost_seq;
    std::shared_ptr<TerminalCostBase<Scalar>> terminal_cost;
    std::vector<std::array<std::vector<std::shared_ptr<StageConstraintBase<Scalar>>>, 3>> constraint_seq;
    std::array<std::vector<std::shared_ptr<TerminalConstraintBase<Scalar>>>, 3> terminal_constraint;
};
