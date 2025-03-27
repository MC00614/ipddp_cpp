#pragma once

#include "model_base.h"
#include <unsupported/Eigen/MatrixFunctions>  // For matrix exponential
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class RPO : public ModelBase {
public:
    RPO();
    ~RPO();

    // User Variable
    double dt;                 // Time step (seconds)
    Eigen::Matrix<double,6,6> Ad;  // Discrete-time A matrix
    Eigen::Matrix<double,6,3> Bd;  // Discrete-time B matrix
    Eigen::VectorXd xg;        // Goal state
    double umax;               // Maximum control value per element
    Eigen::MatrixXd U;         // Control matrix

    Eigen::Vector3d obstacle_center;  // Center of the obstacle
    double obstacle_radius;           // Radius of the obstacle
    double spacecraft_radius;         // Radius of the spacecraft
    double min_dist_2;                // Minimum distance squared

    // Method to check final state
    void checkFinalState(const Eigen::MatrixXd& X_result);
    bool checkTerminalConstraint(const Eigen::VectorXd& final_state);
    // Method to export trajectory for Julia visualization
    void exportTrajectory(const Eigen::MatrixXd& X, const Eigen::MatrixXd& U, const std::string& prefix = "trajectory");
};

RPO::RPO() {
    // Time step
    dt = 1.0;

    // Orbital parameters (matching the Julia code)
    double mu = 3.986004418e14;  // Earth's gravitational parameter (m³/s²)
    double a = 6971100.0;        // Semi-major axis of orbit (meters)
    double n = std::sqrt(mu / (a*a*a));  // Mean motion (rad/s)

    obstacle_center = Eigen::Vector3d(0.0, -4.0, 3.05);
    obstacle_radius = 1.0;
    spacecraft_radius = 1.0;
    min_dist_2 = (obstacle_radius + spacecraft_radius) * (obstacle_radius + spacecraft_radius);

    // Continuous-time dynamics matrices
    Eigen::MatrixXd A_ct = Eigen::MatrixXd::Zero(6, 6);
    A_ct <<  0,   0,   0,   1,   0,   0,
             0,   0,   0,   0,   1,   0,
             0,   0,   0,   0,   0,   1,
          3*n*n, 0,   0,   0, 2*n,   0,
             0,   0,   0, -2*n,  0,   0,
             0,   0, -n*n,   0,   0,   0;

    Eigen::MatrixXd B_ct = Eigen::MatrixXd::Zero(6, 3);
    B_ct.block(3, 0, 3, 3) = 0.1 * Eigen::MatrixXd::Identity(3, 3);

    // Compute discrete-time matrices
    Eigen::MatrixXd Aug = Eigen::MatrixXd::Zero(9, 9);
    Aug.topLeftCorner(6, 6) = A_ct;
    Aug.topRightCorner(6, 3) = B_ct;

    Eigen::MatrixXd expAug = (Aug * dt).exp();
    Ad = expAug.topLeftCorner(6, 6);
    Bd = expAug.topRightCorner(6, 3);

    // Goal state (space station)
    xg = Eigen::VectorXd::Zero(6);
    xg << 0.0, -0.68, 3.05, 0.0, 0.0, 0.0;

    // Maximum control value
    umax = 0.4;  // Each control component is limited to +/- 0.4 (matching Julia)

    // Stage Count
    N = 240;

    // Dimensions
    dim_x = 6;
    dim_u = 3;

    // Status Setting (Initial conditions)
    X_init = Eigen::MatrixXd::Zero(dim_x, N+1);
    // X_init.col(0) << 0.0, -6.68, 3.05, 0.0, 0.0, 0.0;  // Initial position and velocity
    // X_init.col(0) << -2.0, -6.0, 4.0, 0.0, 0.0, 0.0;  // Initial position and velocity
    X_init.col(0) << -4.0, -8.0, 0.0, 0.0, 0.0, 0.0;  // Initial position and velocity

    // Simple warm-start
    U_init = Eigen::MatrixXd::Zero(dim_u, N);
    U_init.setZero();  // Start with zero control
    // U_init = Eigen::MatrixXd::Random(dim_u, N) * 0.1;  // Random initial control

    // Discrete Time System
    f = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd x_next(dim_x);
        x_next = Ad * x + Bd * u;
        return x_next;
    };

    // Stage Cost Function - LQR cost to regulate to goal
    q = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> dual2nd {
      // Separate position and velocity errors
      VectorXdual2nd pos_error = x.head(3) - xg.head(3).cast<dual2nd>();
      VectorXdual2nd vel_error = x.tail(3) - xg.tail(3).cast<dual2nd>();

      // Higher weight on position errors (10x), lower on velocity
    //   dual2nd pos_cost = 1.0 * pos_error.squaredNorm();
    //   dual2nd vel_cost = 1.0 * vel_error.squaredNorm();
      dual2nd control_cost = 100.0 * u.squaredNorm();

    //   return 0.5 * (pos_cost + vel_cost + control_cost);
      return 0.5 * (control_cost);
    };

    // Terminal Cost Function - Higher weight on final error
    p = [this](const VectorXdual2nd& x) -> dual2nd {
      // Separate position and velocity errors
      VectorXdual2nd pos_error = x.head(3) - xg.head(3).cast<dual2nd>();
      VectorXdual2nd vel_error = x.tail(3) - xg.tail(3).cast<dual2nd>();

      // Very high weight on final position error (100x), moderate on velocity
      dual2nd pos_cost = 1.0 * pos_error.squaredNorm();
      dual2nd vel_cost = 1.0 * vel_error.squaredNorm();

      return 0.5 * (pos_cost + vel_cost);
    };

    // Control component bounds: -umax ≤ u[i] ≤ umax for each component

    dim_g = 7;
    g = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        VectorXdual2nd g_n(7);

        // Upper bounds: u[i] ≤ umax
        g_n(0) = umax - u(0);  // u[0] ≤ umax
        g_n(1) = umax - u(1);  // u[1] ≤ umax
        g_n(2) = umax - u(2);  // u[2] ≤ umax

        // // // Lower bounds: -umax ≤ u[i] => u[i] ≥ -umax => -u[i] ≤ umax
        g_n(3) = umax + u(0);  // -u[0] ≤ umax
        g_n(4) = umax + u(1);  // -u[1] ≤ umax
        g_n(5) = umax + u(2);  // -u[2] ≤ umax

        g_n.head(6) /= 4;

        g_n(6) = ((x.head(3) - obstacle_center).squaredNorm() - min_dist_2)/8;

        return -g_n;  // Important: return negative to match solver's convention
    };

    // Guidance Cone
    dim_h = 3;
    h = [this](const VectorXdual2nd& x, const VectorXdual2nd& u) -> VectorXdual2nd {
        const double state_angmax = tan(45.0 * (M_PI/180.0));
        VectorXdual2nd h_n(3);
        h_n(0) = state_angmax * (-(x(1) - xg(1)));  // Cone axis along negative y
        h_n(1) = x(0) - xg(0);                      // x lateral component
        h_n(2) = x(2) - xg(2);                      // z lateral component
        return -h_n/3;
    };
    hs.push_back(h);
    dim_hs.push_back(dim_h);

    // Terminal State Constraint
    dim_hT = 3;
    hT = [this](const VectorXdual2nd& x) -> VectorXdual2nd {
        const double state_angmax = tan(45.0 * (M_PI/180.0));
        VectorXdual2nd h_n(3);
        h_n(0) = state_angmax * (-(x(1) - xg(1)));  // Cone axis along negative y
        h_n(1) = x(0) - xg(0);                      // x lateral component
        h_n(2) = x(2) - xg(2);                      // z lateral component
        return -h_n;
    };
    hTs.push_back(hT);
    dim_hTs.push_back(dim_hT);
}

RPO::~RPO() {
}