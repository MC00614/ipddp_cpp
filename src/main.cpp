#include <Eigen/Dense>
#include <iostream>

#include <cmath>

#include "invpend.h"
#include "soc_ipddp.h"
#include "visualize.h"

int main() {
    // Load Dynamic Model
    auto model = InvPend();
    
    // Solver Setting
    SOC_IPDDP soc_ipddp(model);

    soc_ipddp.init(100, 1e-2);

    soc_ipddp.solve();

    // Parse Result
    Eigen::MatrixXd X_result = soc_ipddp.getX();
    Eigen::MatrixXd U_result = soc_ipddp.getU();
    std::vector<double> all_cost = soc_ipddp.getAllCost();

    // Visualize
    visualize(model.X, model.U, X_result, U_result, all_cost);

    return 0;
}
