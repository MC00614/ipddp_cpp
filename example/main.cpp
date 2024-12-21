#include <Eigen/Dense>
#include <iostream>

#include <cmath>

#include "rocket2d.h"
// #include "rocket3d.h"
#include "drone3d.h"

#include "ipddp.h"
#include "visualize.h"

int main() {
    // Load Dynamic Model
    auto model = Drone3D();
    // auto model = Rocket2D();
    // auto model = Rocket3D();
    // auto model = InvPend();
    // auto model = CartPole();
    // auto model = CarParking();
    
    // Parameter Setting
    Param param;
    param.tolerance = 1e-7;
    param.max_iter = 1000;
    param.mu = 0.2;
    param.infeasible = true;

    // Solver Setting
    clock_t start = clock();

    IPDDP ipddp(model);
    ipddp.init(param);
    ipddp.solve();

    clock_t finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "\nIn Total : " << duration << " Seconds" << std::endl;

    // Parse Result
    Eigen::MatrixXd X_init = Eigen::MatrixXd::Zero(model.dim_x, model.N+1);
    Eigen::MatrixXd U_init = Eigen::MatrixXd::Zero(model.dim_u, model.N);
    Eigen::MatrixXd X_result = ipddp.getResX();
    Eigen::MatrixXd U_result = ipddp.getResU();
    std::vector<double> all_cost = ipddp.getAllCost();

    // Visualize
    visualize(X_init, U_init, X_result, U_result, all_cost);

    return 0;
}
