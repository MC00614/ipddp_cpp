#include <Eigen/Dense>

struct Param {
    int max_iter;
    double tolerance;
    double mu;
    bool infeasible;
    double q;
};
