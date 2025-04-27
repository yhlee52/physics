import numpy as np

G = 1.0

# Lagrange solutions
lagrange_masses = [1.0, 1.0, 1.0]
lagrange_positions = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, np.sqrt(3)/2]
]
lagrange_center = np.array([0.5, np.sqrt(3)/6])
lagrange_omega = np.sqrt(G * sum(lagrange_masses) / np.linalg.norm(lagrange_positions[0] - lagrange_center)**3)

# 중심이동
com = np.mean(lagrange_positions, axis=0)
lagrange_positions -= com

# 각운동 속도
r = np.linalg.norm(lagrange_positions[0])
#omega = np.sqrt(G * sum(lagrange_masses) / r**3)
omega = np.sqrt( (3 * np.sqrt(3) * G * lagrange_masses[0]) / (2) )

# 초기 속도 (반시계 방향)
lagrange_velocities = []
for pos in lagrange_positions:
    vx = omega * (-pos[1])
    vy = omega * (pos[0])
    lagrange_velocities.append([vx, vy])

lagrange_condition = {
    "mass": lagrange_masses,
    "position": lagrange_positions, 
    "velocity": lagrange_velocities
}


# Euler solution
euler_masses = [1.0, 1.0, 1.0]

euler_positions = [
    [-1.0, 0.0],
    [0.0, 0.0],
    [1.0, 0.0]
]

euler_v_mag = 0.5  # 대략적으로 잡은 속도 (fine-tune 필요)
euler_velocities = [
    [0.0, euler_v_mag],
    [0.0, 0.0],
    [0.0, -euler_v_mag]
]

euler_condition = {
    "mass": euler_masses,
    "position": euler_positions,
    "velocity": euler_velocities
}

# Figure-8 solution
fig_masses = [1.0, 1.0, 1.0]

fig_positions = [
    [-0.97000436, 0.24308753],
    [ 0.97000436, -0.24308753],
    [0.0, 0.0]
]

fig_velocities = [
    [0.4662036850, 0.4323657300],
    [0.4662036850, 0.4323657300],
    [-0.93240737, -0.86473146]
]

fig_condition = {
    "mass": fig_masses,
    "position": fig_positions,
    "velocity": fig_velocities
}