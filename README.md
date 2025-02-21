# Conic IPDDP

## Overview
C++ implementation of the Conic Interior Point Differential Dynamic Programming (Conic IPDDP), designed for solving connic-constrained optimization problems.

$$
\min_{u_{0:T-1}} \ell_f(x_T) + \sum_{t=0}^{T-1} \ell_t(x_t, u_t)
$$

$$
\mathrm{subject \ to } \quad x_{t+1} = f_t(x_t, u_t), \quad x_0 = x_{\text{init}}
$$

$$
-g_t(x_t, u_t) \in K_1, \quad -h_t(x_t, u_t) \in K_2
$$

$$
-g_T(x_T) \in K_1, \quad -h_T(x_T) \in K_2
$$

$$
\quad t = 0,1,\dots,T-1
$$

$$
\quad K_1 = ℝ^{m_1}_+, \quad K_2 = L^{m_2}_+
$$


## Feature
- Automatic conversion between different optimization frameworks:
    - unconstrained DDP
    - Interior Point DDP
    - Conic IPDDP

- Supports multiple constraints

- Provides automatic differentiation

- Allows user-defined function overrides

## Installation
Follow these steps to install and set up the project:

```bash
# Clone the repository
git clone https://github.com/MC00614/ipddp_cpp.git

# Navigate to the project directory
cd ipddp_cpp

# Install dependencies

# 1. Eigen 3.3.9
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
tar -xvzf eigen-3.3.9.tar.gz
mv eigen-3.3.9 eigen

# 2. autodiff 1.0.0
wget https://github.com/autodiff/autodiff/archive/refs/tags/v1.0.0.tar.gz
tar -xvzf v1.0.0.tar.gz 
mv autodiff-1.0.0 autodiff

# 3. matplotlibcpp
git clone https://github.com/lava/matplotlib-cpp.git
```

## Usage
- Include the solver header and compile it together with your project.
- Define your model by inheriting ModelBase.
### Build & Run
```bash
# Create and navigate to the build directory
mkdir build && cd build

# Run CMake and compile the project
cmake ..
make -j$(nproc)

# Execute the solver
./ipddp
```

## Examples
To build with example cases:
```bash
cmake .. -DBUILD_EXAMPLE=ON
make -j$(nproc)
```
### [2D Rocket](example/model/rocket2d.h)
```bash
# Run 2D Rocket example
./rocket2d
```

### [3D Drone](example/model/drone3d.h)
```bash
# Run 3D drone example
./drone3d
```