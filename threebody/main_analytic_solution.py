from objects import Body, simulate, animate_trajectories
from known_solutions import lagrange_condition, euler_condition, fig_condition

if __name__ == "__main__":
    # 초기 조건
    bodies = [
        Body(1.0, [0.0, 0.0], [0.0, 0.3]),
        Body(1.0, [1.0, 0.0], [0.0, -0.3]),
        Body(1.0, [0.5, 0.5], [0.3, -0.3])
    ]

    # Known solution
    bodies = [
        Body(fig_condition["mass"][i], fig_condition["position"][i], fig_condition["velocity"][i]) for i in range(3)
    ]

    history = simulate(bodies, total_time=10, dt=0.01, method="rk4")

    animate_trajectories(history)

