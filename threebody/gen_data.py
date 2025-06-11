from threebody_dataset import ThreeBodyDataset

from torch.utils.data import DataLoader

# 데이터셋 생성
def prepare_data(num_samples=512, t_max = 10.0, dt = 0.05, s_range=(-1.0, 1.0), v_range=(-0.5, 0.5)):
    dataset = ThreeBodyDataset(
        num_samples = num_samples,
        t_max       = t_max,
        dt          = dt,      # 더 큰 dt ⇒ 더 빠르지만 오차 증가
        pos_range   = s_range,
        vel_range   = v_range,
        seed        = 42
)
    return dataset

# 학습 루프 예시
def test_data(dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch_x, batch_y in loader:
        print(batch_x.shape)  # (32, 13)
        print(batch_y.shape)  # (32, 12)
        break

# 테스트용 메인 함수
if __name__ == "__main__":
    test_data(dataset=prepare_data())