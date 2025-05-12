from threebody_dataset import ThreeBodyDataset

from torch.utils.data import DataLoader

# 데이터셋 생성
dataset = ThreeBodyDataset(
    num_samples=512,
    x_range=[-1.0, 1.0],
    v_range=[-0.5, 0.5],
    t_range=[0, 10],
    dt=0.01)

# 데이터로더 만들기
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 학습 루프 예시
for batch_x, batch_y in loader:
    print(batch_x.shape)  # (32, 13)
    print(batch_y.shape)  # (32, 12)
    break