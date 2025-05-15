from threebody_dataset import ThreeBodyDataset

from torch.utils.data import DataLoader

# 데이터셋 생성
def prepare_data(num_samples=512, s_range=[-1.0, 1.0], v_range=[-0.5, 0.5], t_range=[0, 10], dt=0.01):
    dataset = ThreeBodyDataset(
        num_samples,
        s_range,
        v_range,
        t_range,
        dt)
    return dataset

# 학습 루프 예시
def test_data(dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch_x, batch_y in loader:
        print(batch_x.shape)  # (32, 13)
        print(batch_y.shape)  # (32, 12)
        break