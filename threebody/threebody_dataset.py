import numpy as np
import torch
from torch.utils.data import Dataset

def leapfrog_batch(pos, vel, dt, n_steps, G=1.0):
    """
    pos, vel : (B, 3, 2)  - x,z 좌표만
    dt       : float
    n_steps  : int
    bulk 심플렉틱 leapfrog 적분 (batch 처리)
    """
    for _ in range(n_steps):
        # ----- half-step 위치 업데이트 -----
        pos_half = pos + 0.5 * dt * vel

        # ----- 가속도 계산 -----
        r12 = pos_half[:,1] - pos_half[:,0]
        r13 = pos_half[:,2] - pos_half[:,0]
        r23 = pos_half[:,2] - pos_half[:,1]

        d12 = np.linalg.norm(r12, axis=1, keepdims=True)
        d13 = np.linalg.norm(r13, axis=1, keepdims=True)
        d23 = np.linalg.norm(r23, axis=1, keepdims=True)

        a1 = ( r12/(d12**3) +  r13/(d13**3)) * G
        a2 = (-r12/(d12**3) +  r23/(d23**3)) * G
        a3 = (-r13/(d13**3) -  r23/(d23**3)) * G
        acc = np.stack([a1, a2, a3], axis=1)      # (B,3,2)

        # ----- 속도 full-step -----
        vel = vel + dt * acc

        # ----- 위치 half-step 마무리 -----
        pos = pos_half + 0.5 * dt * vel
    return pos, vel


class ThreeBodyFastDataset(Dataset):
    """
    빠른 벡터화+LeapFrog 기반 삼체 데이터셋
    · num_samples      : 샘플 수
    · t_max, dt        : 통합 종료 시간, time-step
    · pos_range, vel_range : (min,max) 튜플
    """
    def __init__(self,
                 num_samples    = 10000,
                 t_max          = 2.0,
                 dt             = 0.01,
                 pos_range      = (-1.0, 1.0),
                 vel_range      = (-0.5, 0.5),
                 seed           = None):
        super().__init__()
        rng = np.random.default_rng(seed)

        # --- 초기 조건 생성 ---
        self.t_array = rng.uniform(0.0, t_max, size=(num_samples, 1))  # 예측 시점 t
        self.init_pos = rng.uniform(*pos_range, size=(num_samples, 3, 2))
        self.init_vel = rng.uniform(*vel_range, size=(num_samples, 3, 2))

        # --- 시뮬레이션 (벡터화) ---
        n_steps_max = int(t_max / dt)
        # 한 번에 전체를 돌리면 메모리가 클 수 있으므로 1024단위 batch로 나눔
        batch_size = 1024
        final_pos  = []
        final_vel  = []

        for i in range(0, num_samples, batch_size):
            b_end = min(i+batch_size, num_samples)
            pos = self.init_pos[i:b_end].copy()
            vel = self.init_vel[i:b_end].copy()
            # 각 샘플별 step 수 다름 → 가장 큰 n_steps만큼 돌리고, 중간에 필요한 step에서 값 저장
            steps_needed = (self.t_array[i:b_end] / dt).astype(int)
            max_steps_b  = steps_needed.max()
            snapshots_pos = []
            snapshots_vel = []

            for s in range(max_steps_b):
                pos, vel = leapfrog_batch(pos, vel, dt, 1)  # 한 step
                take = (steps_needed.squeeze() == (s+1))
                if take.any():
                    snapshots_pos.append(pos[take])
                    snapshots_vel.append(vel[take])

            # 순서 보존
            final_pos.append(np.concatenate(snapshots_pos, axis=0))
            final_vel.append(np.concatenate(snapshots_vel, axis=0))

        self.final_pos = np.concatenate(final_pos, axis=0)
        self.final_vel = np.concatenate(final_vel, axis=0)

    def __len__(self):
        return len(self.t_array)

    def __getitem__(self, idx):
        x_input = np.concatenate([
            self.init_pos[idx].flatten(),
            self.init_vel[idx].flatten(),
            self.t_array[idx]
        ])
        y_output = np.concatenate([
            self.final_pos[idx].flatten(),
            self.final_vel[idx].flatten()
        ])
        return (
            torch.tensor(x_input, dtype=torch.float32),
            torch.tensor(y_output, dtype=torch.float32)
        )