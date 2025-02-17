# Connic IPDDP

## Description
C++ implementation of the Connic IPDDP algorithm, designed for solving connic constrained optimization problems.

## Installation
Follow these steps to install and set up the project:

```bash
# Clone the repository
git clone https://github.com/mczb/ipddp_cpp.git

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
Here are some examples of how to use the project:

```bash
# Build example
mkdir build
cd build
cmake .. -DBUILD_EXAMPLE=ON
make -j$(nproc)

# Run rocket2d example
./rocket2d

# Run drone3d example
./drone3d
```

# Result