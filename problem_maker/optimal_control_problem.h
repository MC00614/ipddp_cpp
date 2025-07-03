#pragma once

#include "discrete_dynamics.h"
#include "stage_cost_function.h"
#include "terminal_cost_function"
#include "stage_constraint_function"
#include "terminal_constraint_function"


#include <memory>
#include <vector>

template <typename Scalar>
class OptimalControlProblem {
public:
    std::vector<std::shared_ptr<DiscreteDynamics<Scalar>>> dynamics_seq;
    std::vector<std::shared_ptr<StageCostFunction<Scalar>>> cost_seq;
    std::vector<std::shared_ptr<StageConstraintFunction<Scalar>>> constraint_seq;
    std::shared_ptr<TerminalConstraintFunction<Scalar>> terminal_constraint;
    std::vector<std::shared_ptr<TermCostFunction<Scalar>>> terminal_cost;

    std::vector<double> x0;
    double T;
    int N;

    OptimalControlProblem();
    OptimalControlProblem() = default;

};