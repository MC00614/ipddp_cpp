#include <Eigen/Dense>
#include <iostream>

#include <cmath>
#include <chrono>

// #include "rocket2d_eq_st.h"
// #include "rocket2d.h"
#include "rocket3d.h"
// #include "drone3d.h"

#include "ipddp.h"
#include "visualize.h"

int main() {
    // Load Dynamic Model
    // auto model = std::make_shared<Rocket2D>();
    auto model = std::make_shared<Rocket3D>();
    
    // Parameter Setting
    Param param;
    param.tolerance = 1e-3;
    param.max_iter = 500;
    param.max_inner_iter = 20;
    param.mu = 1.0;
    param.rho = 1.0;
    param.max_step_iter = 10;
    param.max_regularization = 10;
    
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

    // std::cout<<"X_result = \n"<<X_result.transpose()<<std::endl;
    // std::cout<<"U_result = \n"<<U_result.transpose()<<std::endl;
    std::cout<<"X_0 = \n"<<X_result.col(0).transpose()<<std::endl;
    std::cout<<"X_T = \n"<<X_result.col(model->N).transpose()<<std::endl;
    std::cout<<"U_T = \n"<<U_result.col(model->N-1).transpose()<<std::endl;

    // Visualize
    visualize(X_init, U_init, X_result, U_result, all_cost);

    return 0;
}
