from model import SimplePINN, Trainer
from gen_data import prepare_data

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델, 데이터, 옵티마이저, 스케줄러 준비
model = SimplePINN()
train_dataset = prepare_data(num_samples=64*700)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataset = prepare_data(num_samples=64*100)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

# Trainer 인스턴스 생성
trainer = Trainer(model, train_loader, valid_loader, optimizer, scheduler, device=device)

# 학습 시작
trainer.train(num_epochs=200, alpha=1.0, beta=1.0)