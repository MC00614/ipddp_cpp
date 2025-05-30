#include <Eigen/Dense>
#include <iostream>

#include <cmath>

#include "crazyflie.h"

#include "ipddp.h"
#include "visualize.h"

int main() {
    // Load Dynamic Model
    auto model = std::make_shared<CrazyFlie>();
    
    // Parameter Setting
    Param param;
    param.tolerance = 1e-3;
    param.max_iter = 1000;
    param.max_inner_iter = 50;
    param.max_regularization = 20;

    param.rho_max = 1e8;
    param.mu_min = 1e-8;
    // param.mu_mul = 0.5;
    // param.rho_mul = 10.0;

    // param.auto_init = true;
    // param.auto_init_cc = true;
    // param.auto_init_ec = true;
    
    // Solver Setting
    clock_t start = clock();

    IPDDP ipddp(model);
    ipddp.init(param);
    ipddp.solve();

    clock_t finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "\nIn Total : " << duration << " Seconds" << std::endl;

    // Parse Result
    Eigen::MatrixXd X_init = Eigen::MatrixXd::Zero(model->dim_x, model->N+1);
    Eigen::MatrixXd U_init = Eigen::MatrixXd::Zero(model->dim_u, model->N);
    Eigen::MatrixXd X_result = ipddp.getResX();
    Eigen::MatrixXd U_result = ipddp.getResU();
    std::vector<double> all_cost = ipddp.getAllCost();

    std::cout<<"X_result = \n"<<X_result.transpose()<<std::endl;
    std::cout<<"U_result = \n"<<U_result.transpose()<<std::endl;

    // Visualize
    visualize(X_init, U_init, X_result, U_result, all_cost);

    return 0;
}
