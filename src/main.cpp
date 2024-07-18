#include <Eigen/Dense>
#include <iostream>

#include <cmath>

#include "invpend.h"
// #include "cart_pole.h"
// #include "car_parking.h"

#include "soc_ipddp.h"
#include "visualize.h"

int main() {
    // Load Dynamic Model
    auto model = InvPend();
    // auto model = CartPole();
    // auto model = CarParking();
    
    // Parameter Setting
    Param param;
    param.tolerance = 1e-7;
    param.max_iter = 100;
    param.mu = 0;
    param.infeasible = false;

    // Solver Setting
    SOC_IPDDP soc_ipddp(model);
    soc_ipddp.init(param);

    soc_ipddp.solve();

    // Parse Result
    Eigen::MatrixXd X_init = soc_ipddp.getInitX();
    Eigen::MatrixXd U_init = soc_ipddp.getInitU();
    Eigen::MatrixXd X_result = soc_ipddp.getResX();
    Eigen::MatrixXd U_result = soc_ipddp.getResU();
    std::vector<double> all_cost = soc_ipddp.getAllCost();

    // Visualize
    visualize(X_init, U_init, X_result, U_result, all_cost);

    return 0;
}
