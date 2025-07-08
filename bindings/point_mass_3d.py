import numpy as np
import time

import alipddp

class Obstacle(alipddp.StageConstraintBase):
    def __init__(self):
        super().__init__()
        self.setConstraintType(alipddp.ConstraintType.NO)
        self.setDimC(3)

        self.obs_cent = np.array([
            [5.0, 4.0, 6.0],
            [3.0, 2.0, 5.0],
            [1.0, 4.0, 4.0]
        ])
        self.obs_rad = np.array([3.0, 2.0, 3.0])

    def c(self, x, u):
        c_n = []
        for i in range(3):
            dist = np.linalg.norm(x[:3] - self.obs_cent[i])
            c_n.append(-(dist - self.obs_rad[i]))
        return np.array(c_n)

    def cx(self, x, u):
        J = np.zeros((3, len(x)))
        for i in range(3):
            delta = x[:3] - self.obs_cent[i]
            norm = np.linalg.norm(delta)
        return - J

    def cu(self, x, u):
        return np.zeros((3, len(u)))
    
class MaxInput(alipddp.StageConstraintBase):
    def __init__(self):
        super().__init__()
        self.umax = 9.81 * 1.5
        self.setConstraintType(alipddp.ConstraintType.NO)
        self.setDimC(1)

    def c(self, x, u):
        return np.array([-(self.umax - np.linalg.norm(u))])

    def cx(self, x, u):
        return np.zeros((1, len(x)))

    def cu(self, x, u):
        return -(-u[np.newaxis, :] / np.linalg.norm(u))
    
def point_mass_3d():
    N = 100
    dt = 0.1
    problem = alipddp.OptimalControlProblem(N)

    A = np.zeros((6, 6))
    A[0:3, 0:3] = np.eye(3)
    A[0:3, 3:6] = dt * np.eye(3)
    A[3:6, 3:6] = np.eye(3)
    
    B = np.zeros((6, 3))
    B[3:6, 0:3] = dt * np.eye(3)

    A = A.astype(np.float64)
    B = B.astype(np.float64)

    dynamics = alipddp.LinearDiscreteDynamics(A, B)
    c = np.zeros(6)
    c[5] = -9.81 * dt
    dynamics.setC(c)
    problem.setStageDynamics(dynamics)

    Q = np.zeros((6, 6))
    R = 1e-6 * np.eye(3)
    stage_cost = alipddp.QuadraticStageCost(Q, R)
    problem.setStageCost(stage_cost)

    QT = np.zeros((6, 6))
    terminal_cost = alipddp.QuadraticTerminalCost(QT)
    problem.setTerminalCost(terminal_cost)

    glideslope_cx = np.zeros((3, 6))
    glideslope_cx[0, 2] = -np.tan(np.deg2rad(45))
    glideslope_cx[1, 0] = -1.0
    glideslope_cx[2, 1] = -1.0
    glideslope_cu = np.zeros((3, 3))
    glide_c0 = np.zeros(3)
    
    glide_constraint = alipddp.LinearStageConstraint(glideslope_cx, glideslope_cu, glide_c0, alipddp.ConstraintType.SOC)
    problem.addStageConstraint(glide_constraint)

    inputcone_cx = np.zeros((3, 6))
    inputcone_cu = np.zeros((3, 3))
    inputcone_cu[0, 2] = -np.tan(np.deg2rad(20))
    inputcone_cu[1, 0] = -1.0
    inputcone_cu[2, 1] = -1.0
    inputcone_c0 = np.zeros(3)
    inputcone_constraint = alipddp.LinearStageConstraint(inputcone_cx, inputcone_cu, inputcone_c0, alipddp.ConstraintType.SOC)
    problem.addStageConstraint(inputcone_constraint)

    maxinput_constraint = MaxInput()
    problem.addStageConstraint(maxinput_constraint)
    obstacle_constraint = Obstacle()
    # problem.addStageConstraint(obstacle_constraint)

    CT = np.eye(6)
    c0T = np.zeros(6)
    c0T[2] = -1.0
    terminal_constraint = alipddp.LinearTerminalConstraint(CT, c0T, alipddp.ConstraintType.EQ)
    problem.addTerminalConstraint(terminal_constraint)

    x0 = np.array([5.0, 5.0, 10.0, -0.2, -0.2, -0.3])
    u0 = np.array([0.0, 0.0, 9.81])
    problem.setInitialState(0, x0)
    problem.setInitialControl(u0)

    param = alipddp.Param()
    solver = alipddp.ALIPDDP(problem)
    solver.init(param)
    solver.solve()

    X = solver.getResX()
    U = solver.getResU()

    np.set_printoptions(precision=4, suppress=True)
    print("X_last = ", X[-1])
    print("U_last = ", U[-1])

if __name__ == "__main__":
    start_time = time.perf_counter()
    point_mass_3d()
    end_time = time.perf_counter()
    print("Time taken for Whole Process: {:.5f} seconds".format(end_time - start_time))
