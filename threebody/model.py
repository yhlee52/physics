import torch
import torch.nn as nn

# Simple PINN(FCNN)
class SimplePINN(nn.Module):
    def __init__(self):
        super(SimplePINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 12)
        )
    
    def forward(self, x):
        return self.net(x)

# Trainer  
class Trainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, scheduler, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.history = {
            'train' : {
                'total_loss' : [],
                'data_loss' : [],
                'physics_loss' : [],
                'hamiltonian_loss' : []
            },
            'valid' : {
                'total_loss' : [],
                'data_loss' : [],
                'physics_loss' : [],
                'hamiltonian_loss' : []
            }
        }

    def train(self, num_epochs=100, alpha=1.0, beta=1.0):
        for epoch in range(1, num_epochs + 1):
            # Train
            self.model.train()
            total_loss = 0.0
            total_data_loss = 0.0
            total_physics_loss = 0.0
            total_hamiltonian_loss = 0.0

            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()

                loss, data_loss, physics_loss, hamiltonian_loss = compute_pinn_loss(
                    self.model, batch_x, batch_y, alpha=alpha, beta=beta
                )

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_data_loss += data_loss.item()
                total_physics_loss += physics_loss.item()
                total_hamiltonian_loss += hamiltonian_loss.item()

            self.scheduler.step(total_loss)

            # fill history
            self.history['train']['total_loss'].append(total_loss)
            self.history['train']['data_loss'].append(data_loss)
            self.history['train']['physics_loss'].append(physics_loss)
            self.history['train']['hamiltonian_loss'].append(hamiltonian_loss)

            #if epoch % 10 == 0 or epoch == 1:
            print(f"Train losses"
                    f"[Epoch {epoch:04d}] "
                    f"Total: {total_loss:.6f}, "
                    f"Data: {total_data_loss:.6f}, "
                    f"Physics: {total_physics_loss:.6f}, "
                    f"Hamiltonian: {total_hamiltonian_loss:.6f}")
            
            # Valid
            self.model.eval()
            total_loss = 0.0
            total_data_loss = 0.0
            total_physics_loss = 0.0
            total_hamiltonian_loss = 0.0

            # Train
            with torch.no_grad():
                for batch_x, batch_y in self.valid_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    self.optimizer.zero_grad()

                    loss, data_loss, physics_loss, hamiltonian_loss = compute_pinn_loss(
                        self.model, batch_x, batch_y, alpha=alpha, beta=beta
                    )

                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    total_data_loss += data_loss.item()
                    total_physics_loss += physics_loss.item()
                    total_hamiltonian_loss += hamiltonian_loss.item()

                self.scheduler.step(total_loss)

                # fill history
                self.history['valid']['total_loss'].append(total_loss)
                self.history['valid']['data_loss'].append(data_loss)
                self.history['valid']['physics_loss'].append(physics_loss)
                self.history['valid']['hamiltonian_loss'].append(hamiltonian_loss)

                #if epoch % 10 == 0 or epoch == 1:
                print(f"Validation losses"
                        f"[Epoch {epoch:04d}] "
                        f"Total: {total_loss:.6f}, "
                        f"Data: {total_data_loss:.6f}, "
                        f"Physics: {total_physics_loss:.6f}, "
                        f"Hamiltonian: {total_hamiltonian_loss:.6f}")
        print("Training complete.")

# Loss from Newton's equation
def compute_gravity_acceleration(positions):
    batch_size = positions.size(0)
    pos = positions.view(batch_size, 3, 2)

    r12 = pos[:,1] - pos[:,0]
    r13 = pos[:,2] - pos[:,0]
    r23 = pos[:,2] - pos[:,1]

    d12 = torch.norm(r12, dim=1, keepdim=True)
    d13 = torch.norm(r13, dim=1, keepdim=True)
    d23 = torch.norm(r23, dim=1, keepdim=True)

    F12 = r12 / (d12**3 + 1e-8)
    F13 = r13 / (d13**3 + 1e-8)
    F23 = r23 / (d23**3 + 1e-8)

    a1 = F12 + F13
    a2 = -F12 + F23
    a3 = -F13 - F23

    accelerations = torch.cat([a1, a2, a3], dim=1)
    return accelerations

# Loss from Hamiltonian conservation
def compute_hamiltonian(positions, velocities):
    batch_size = positions.size(0)
    pos = positions.view(batch_size, 3, 2)
    vel = velocities.view(batch_size, 3, 2)

    kinetic_energy = 0.5 * torch.sum(vel**2, dim=[1,2])

    r12 = torch.norm(pos[:,0] - pos[:,1], dim=1)
    r13 = torch.norm(pos[:,0] - pos[:,2], dim=1)
    r23 = torch.norm(pos[:,1] - pos[:,2], dim=1)

    potential_energy = -(1.0 / r12 + 1.0 / r13 + 1.0 / r23)

    hamiltonian = kinetic_energy + potential_energy
    return hamiltonian

# Total loss calculation
def compute_pinn_loss(model, batch_x, batch_y, alpha=1.0, beta=1.0):
    batch_x = batch_x.requires_grad_(True)

    y_pred = model(batch_x)

    data_loss = nn.functional.mse_loss(y_pred, batch_y)

    # Physics residual loss (운동방정식)
    grads = torch.autograd.grad(
        outputs=y_pred,
        inputs=batch_x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    pred_positions = y_pred[:, :6]
    pred_velocities = y_pred[:, 6:]
    pred_accelerations = grads[:, 6:]

    gravity_accelerations = compute_gravity_acceleration(pred_positions)
    physics_loss = nn.functional.mse_loss(pred_accelerations, gravity_accelerations)

    # Hamiltonian loss (에너지 보존)
    H_predicted = compute_hamiltonian(pred_positions, pred_velocities)

    # 초기 상태에서 Hamiltonian 계산
    init_positions = batch_x[:, :6]
    init_velocities = batch_x[:, 6:12]
    H_initial = compute_hamiltonian(init_positions, init_velocities)

    hamiltonian_loss = nn.functional.mse_loss(H_predicted, H_initial)

    # 최종 total loss
    total_loss = data_loss + alpha * physics_loss + beta * hamiltonian_loss

    return total_loss, data_loss, physics_loss, hamiltonian_loss