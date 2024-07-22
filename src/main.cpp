#include <Eigen/Dense>
#include <iostream>

#include <cmath>

#include "invpend.h"
// #include "cart_pole.h"
#include "car_parking.h"

#include "ipddp.h"
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
    clock_t start = clock();

    IPDDP ipddp(model);
    ipddp.init(param);
    ipddp.solve();

    clock_t finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "\nIn Total : " << duration << " Seconds" << std::endl;

    // Parse Result
    Eigen::MatrixXd X_init = ipddp.getInitX();
    Eigen::MatrixXd U_init = ipddp.getInitU();
    Eigen::MatrixXd X_result = ipddp.getResX();
    Eigen::MatrixXd U_result = ipddp.getResU();
    std::vector<double> all_cost = ipddp.getAllCost();

    // Visualize
    visualize(X_init, U_init, X_result, U_result, all_cost);

    return 0;
}
