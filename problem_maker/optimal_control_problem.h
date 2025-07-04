#pragma once

#include "discrete_dynamics.h"
#include "stage_cost_function.h"
#include "terminal_cost_function.h"
#include "stage_constraint_function.h"
#include "terminal_constraint_function.h"

#include <memory>
#include <vector>
#include <array>

template <typename Scalar>
class OptimalControlProblem {
public:
    OptimalControlProblem(int horizon_steps, const Vector<Scalar>& x0_in) {
        N = horizon_steps;
        x0 = x0_in;
    
        dynamics_seq.resize(N);
        cost_seq.resize(N);
        constraint_seq.resize(N);
    }

    void init() {
        std::cout << "INIT" << std::endl;
        // Stack Constraints
    }

    void setStageDynamics(int k, std::shared_ptr<DiscreteDynamics<Scalar>> dyn) {
        dynamics_seq.at(k) = dyn;
    }

    void setStageCost(int k, std::shared_ptr<StageCostFunction<Scalar>> cost) {
        cost_seq.at(k) = cost;
    }
    
    void setTerminalCost(std::shared_ptr<TerminalCostFunction<Scalar>> cost) {
        terminal_cost = cost;
    }


    void addStageConstraint(int k, std::shared_ptr<StageConstraintFunction<Scalar>> constraint) {
        constraint_seq[k][static_cast<int>(constraint->getConstraintType())] = constraint;
    }

    void addTerminalConstraint(std::shared_ptr<TerminalConstraintFunction<Scalar>> constraint) {
        terminal_constraint[static_cast<int>(constraint->getConstraintType())] = constraint;
    }
private:
    std::vector<std::shared_ptr<DiscreteDynamics<Scalar>>> dynamics_seq;
    std::vector<std::shared_ptr<StageCostFunction<Scalar>>> cost_seq;
    std::vector<std::shared_ptr<TerminalCostFunction<Scalar>>> terminal_cost;
    std::vector<std::array<std::shared_ptr<StageConstraintFunction<Scalar>>, 3>> constraint_seq;
    std::array<std::shared_ptr<TerminalConstraintFunction<Scalar>>, 3> terminal_constraint;

    std::vector<double> x0;
    double T;
    int N;
};