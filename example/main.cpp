#include <Eigen/Dense>
#include <iostream>

#include <cmath>
#include <chrono>

#include "rocket3d_rss.h"
// #include "rocket3d_move.h"

#include "ipddp.h"
#include "visualize.h"

int main() {
    // Load Dynamic Model
    auto model = std::make_shared<Rocket3D>();
    
    // Parameter Setting
    Param param;
    // param.tolerance = 1e-3;
    // param.max_iter = 500;
    // param.max_inner_iter = 20;
    param.tolerance = 1e-4;
    param.max_iter = 1000;
    param.max_inner_iter = 100;
    param.mu = 1.0;
    param.mu_min = 1e-6;
    param.rho = 1.0;
    param.max_step_iter = 10;
    param.max_regularization = 20;
    // param.auto_init_noc = true;
    // param.auto_init_cc = true;
    // param.auto_init_ccT = true;
    // param.auto_init_ecT = true;

    // Solver Setting
    auto start = std::chrono::steady_clock::now();

    IPDDP ipddp(model);
    ipddp.init(param);
    ipddp.solve();

    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = finish - start;
    
    // Parse Result
    Eigen::MatrixXd X_init = Eigen::MatrixXd::Zero(model->dim_x, model->N+1);
    Eigen::MatrixXd U_init = Eigen::MatrixXd::Zero(model->dim_u, model->N);
    Eigen::MatrixXd X_result = ipddp.getResX();
    Eigen::MatrixXd U_result = ipddp.getResU();
    std::vector<double> all_cost = ipddp.getAllCost();

    // RESCALE
    X_result.topRows(6) *= model->r_scale;
    U_result *= (model->m_scale * model->r_scale);
    
    std::cout<<"X_result = \n"<<X_result.transpose()<<std::endl;
    std::cout<<"U_result = \n"<<U_result.transpose()<<std::endl;
    std::cout<<"X_0 = \n"<<X_result.col(0).transpose()<<std::endl;
    std::cout<<"X_T = \n"<<X_result.col(model->N).transpose()<<std::endl;
    std::cout<<"U_0 = \n"<<U_result.col(0).transpose()<<std::endl;
    std::cout<<"U_T = \n"<<U_result.col(model->N-1).transpose()<<std::endl;
    
    std::cout << "\nIn Total : " << duration.count() << " Seconds" << std::endl;

    // Visualize
    visualize(X_init, U_init, X_result, U_result, all_cost);

    return 0;
}
