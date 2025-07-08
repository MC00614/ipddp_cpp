import numpy as np
import alipddp

def test_alipddp_solver():
    N = 100
    dt = 0.1
    problem = alipddp.OptimalControlProblem(N)

    A = np.zeros((6, 6))
    A[0:3, 0:3] = np.eye(3)
    A[0:3, 3:6] = dt * np.eye(3)
    A[3:6, 3:6] = np.eye(3)
    
    B = np.zeros((6, 3))
    B[3:6, 0:3] = dt * np.eye(3)

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
    glide_constraint = alipddp.LinearStageConstraint(glideslope_cx, glideslope_cu, glide_c0, "SOC")
    problem.addStageConstraint(glide_constraint)

    inputcone_cx = np.zeros((3, 6))
    inputcone_cu = np.zeros((3, 3))
    inputcone_cu[0, 2] = -np.tan(np.deg2rad(20))
    inputcone_cu[1, 0] = -1.0
    inputcone_cu[2, 1] = -1.0
    inputcone_c0 = np.zeros(3)
    inputcone_constraint = alipddp.LinearStageConstraint(inputcone_cx, inputcone_cu, inputcone_c0, "SOC")
    problem.addStageConstraint(inputcone_constraint)

    # # Nonlinear constraints (registered from C++)
    # problem.addStageConstraint(alipddp.Obstacle())
    # problem.addStageConstraint(alipddp.MaxInput())

    CT = np.eye(6)
    c0T = np.zeros(6)
    c0T[2] = -1.0
    terminal_constraint = alipddp.LinearTerminalConstraint(CT, c0T, "EQ")
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

    print("X_last = ", X[-1])
    print("U_last = ", U[-1])

    assert X[-1][2] < 1.0, "Rocket did not land correctly"
    print("✅ Test passed.")

if __name__ == "__main__":
    test_alipddp_solver()
