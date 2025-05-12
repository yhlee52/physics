import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp

# 중력 상수
G = 1.0

def three_body_derivatives(t, state):
    x1, z1, x2, z2, x3, z3 = state[0:6]
    vx1, vz1, vx2, vz2, vx3, vz3 = state[6:12]
    
    r12 = np.array([x2 - x1, z2 - z1])
    r13 = np.array([x3 - x1, z3 - z1])
    r23 = np.array([x3 - x2, z3 - z2])
    
    d12 = np.linalg.norm(r12)
    d13 = np.linalg.norm(r13)
    d23 = np.linalg.norm(r23)
    
    F12 = G * r12 / d12**3
    F13 = G * r13 / d13**3
    F23 = G * r23 / d23**3
    
    ax1 = F12[0] + F13[0]
    az1 = F12[1] + F13[1]
    ax2 = -F12[0] + F23[0]
    az2 = -F12[1] + F23[1]
    ax3 = -F13[0] - F23[0]
    az3 = -F13[1] - F23[1]
    
    derivatives = np.array([
        vx1, vz1, vx2, vz2, vx3, vz3,
        ax1, az1, ax2, az2, ax3, az3
    ])
    return derivatives

def simulate_three_body(initial_positions, initial_velocities, t_eval):
    initial_state = np.concatenate([
        initial_positions.flatten(),
        initial_velocities.flatten()
    ])
    
    t_span = (0, t_eval)
    
    sol = solve_ivp(
        fun=three_body_derivatives,
        t_span=t_span,
        y0=initial_state,
        method='RK45',
        rtol=1e-9,
        atol=1e-9
    )
    
    final_state = sol.y[:, -1]
    
    final_positions = final_state[:6].reshape(3, 2)
    final_velocities = final_state[6:12].reshape(3, 2)
    
    final_state_combined = np.concatenate([final_positions, final_velocities], axis=1)  # (3, 4)
    return final_state_combined

class ThreeBodyDataset(Dataset):
    def __init__(self, num_samples_per_t, x_range, v_range, t_range, dt):
        super().__init__()

        self.t_vals = np.arange(t_range[0], t_range[1] + dt, dt)
        self.data = []
        self.labels = []

        for t in self.t_vals:
            for _ in range(num_samples_per_t):
                positions = np.random.uniform(x_range[0], x_range[1], size=(3, 2))  # (x, z) per body
                velocities = np.random.uniform(v_range[0], v_range[1], size=(3, 2))  # (u, w) per body

                # 입력: 초기 위치, 초기 속도, 시간 t
                x_input = np.concatenate([positions.flatten(), velocities.flatten(), [t]])

                # 출력: t 시점의 위치 + 속도
                y_output = self.simulate_three_body(positions, velocities, t)
                self.data.append(x_input)
                self.labels.append(y_output.flatten())

        self.data = np.stack(self.data)
        self.labels = np.stack(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
