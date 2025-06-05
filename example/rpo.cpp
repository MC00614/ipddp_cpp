#include <Eigen/Dense>
#include <iostream>

#include <cmath>
#include <chrono>

#include "rpo.h"

#include "ipddp.h"
#include "visualize.h"

int main() {
    // Load Dynamic Model
    auto model = std::make_shared<RPO>();

    // Parameter Setting
    Param param;
    param.tolerance = 1e-2;
    param.forward_filter = 1;
    param.max_inertia_correction = 10;
    
    // Solver Setting
    auto start = std::chrono::steady_clock::now();

    IPDDP ipddp(model);
    ipddp.init(param);
    ipddp.solve();

    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = finish - start;
    std::cout << "\nIn Total : " << duration.count() << " Seconds" << std::endl;

    // Parse Result
    Eigen::MatrixXd X_init = Eigen::MatrixXd::Zero(model->dim_x, model->N+1);
    Eigen::MatrixXd U_init = Eigen::MatrixXd::Zero(model->dim_u, model->N);
    Eigen::MatrixXd X_result = ipddp.getResX();
    Eigen::MatrixXd U_result = ipddp.getResU();
    std::vector<double> all_cost = ipddp.getAllCost();

    std::cout<<"X_result = \n"<<X_result.transpose()<<std::endl;
    std::cout<<"U_result = \n"<<U_result.transpose()<<std::endl;

    std::cout<<"X_last = \n"<<X_result.col(model->N).transpose()<<std::endl;
    std::cout<<"U_last = \n"<<U_result.col(model->N-1).transpose()<<std::endl;
    
    // Visualize
    visualize(X_init, U_init, X_result, U_result, all_cost);

    return 0;
}
