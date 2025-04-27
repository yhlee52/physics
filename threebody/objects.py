import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 중력상수
G = 1.0

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.force = np.zeros(2, dtype=float)

    def reset_force(self):
        self.force[:] = 0.0

    def add_force(self, other):
        delta = other.position - self.position
        distance = np.linalg.norm(delta)
        if distance == 0:
            return
        force_magnitude = G * self.mass * other.mass / distance**2
        self.force += force_magnitude * delta / distance

    def update(self, dt):
        acceleration = self.force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

    def get_state(self):
        return np.concatenate((self.position, self.velocity))
    
    def set_state(self, state):
        self.position = state[:2]
        self.velocity = state[2:]

def compute_accelerations(states, masses):
    n = len(masses)
    accs = np.zeros((n, 2))
    for i in range(n):
        for j in range(n):
            if i != j:
                delta = states[j, :2] - states[i, :2]
                distance = np.linalg.norm(delta)
                if distance != 0:
                    accs[i] += G * masses[j] * delta / distance**3
    return accs

def rk4_step(bodies, dt):
    n = len(bodies)
    y0 = np.array([b.get_state() for b in bodies])  # (n, 4)
    m = np.array([b.mass for b in bodies])

    def derivatives(y):
        pos = y[:, :2]
        vel = y[:, 2:]
        acc = compute_accelerations(y, m)
        return np.hstack((vel, acc))

    k1 = derivatives(y0)
    k2 = derivatives(y0 + 0.5 * dt * k1)
    k3 = derivatives(y0 + 0.5 * dt * k2)
    k4 = derivatives(y0 + dt * k3)

    y_next = y0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    for i in range(n):
        bodies[i].set_state(y_next[i])

def simulate(bodies, total_time, dt, method="euler"):
    steps = int(total_time / dt)
    history = {i: [] for i in range(len(bodies))}

    for _ in range(steps):
        if method == "euler":
            for body in bodies:
                body.reset_force()
            for i, body in enumerate(bodies):
                for j, other in enumerate(bodies):
                    if i != j:
                        body.add_force(other)
            for i, body in enumerate(bodies):
                body.update(dt)
                history[i].append(body.position.copy())
        elif method == "rk4":
            for i, body in enumerate(bodies):
                history[i].append(body.position.copy())
            rk4_step(bodies, dt)
        else:
            raise ValueError("Unknown method. Choose 'euler' or 'rk4'.")

    return history

def animate_trajectories(history, save_path=None, interval=20):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Three-Body Problem Animation')
    ax.set_aspect('equal')
    ax.grid()

    colors = ['r', 'g', 'b']
    labels = ['Body 1', 'Body 2', 'Body 3']
    paths = [np.array(history[i]) for i in range(len(history))]
    scatters = [ax.plot([], [], 'o', color=colors[i], label=labels[i])[0] for i in range(len(paths))]
    trails = [ax.plot([], [], '-', color=colors[i], lw=0.8)[0] for i in range(len(paths))]

    all_positions = np.vstack([path for path in paths])
    margin = 0.5
    x_min, x_max = np.min(all_positions[:, 0]) - margin, np.max(all_positions[:, 0]) + margin
    y_min, y_max = np.min(all_positions[:, 1]) - margin, np.max(all_positions[:, 1]) + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend()

    def init():
        for scatter, trail in zip(scatters, trails):
            scatter.set_data([], [])
            trail.set_data([], [])
        return scatters + trails

    def update(frame):
        for i in range(len(paths)):
            scatters[i].set_data([paths[i][frame, 0]], [paths[i][frame, 1]])
            trails[i].set_data(paths[i][:frame+1, 0], paths[i][:frame+1, 1])
        return scatters + trails

    ani = animation.FuncAnimation(fig, update, frames=len(paths[0]), init_func=init,
                                  interval=interval, blit=True, repeat=True)

    if save_path:
        ani.save(save_path, writer='ffmpeg', fps=30)
        plt.close(fig)
        return fig, ani
    else:
        plt.show()

    