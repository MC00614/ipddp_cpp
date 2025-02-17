#include <Eigen/Dense>
#include <iostream>

#include <cmath>

#include "drone3d.h"

#include "ipddp.h"
#include "visualize.h"

int main() {
    // Load Dynamic Model
    auto model = std::make_shared<Drone3D>();
    
    // Parameter Setting
    Param param;
    param.tolerance = 1e-6;
    param.max_iter = 1000;
    param.mu = 0.1;
    param.max_step_iter = 10;
    
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

    // std::cout<<"X_result = \n"<<X_result.transpose()<<std::endl;
    // std::cout<<"U_result = \n"<<U_result.transpose()<<std::endl;

    // Visualize
    visualize(X_init, U_init, X_result, U_result, all_cost);

    return 0;
}
