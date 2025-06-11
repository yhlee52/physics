from objects import Body, simulate, animate_trajectories
from known_solutions import lagrange_condition, euler_condition, fig_condition

if __name__ == "__main__":
    # 초기 조건
    # bodies = [
    #     Body(1.0, [-1.0, 0.0], [0.2, 0.3]),
    #     Body(1.2, [ 1.0, 0.0], [-0.5, 0.1]),
    #     Body(1.5, [ 0.0, 0.2], [0.3, -0.4])
    # ]

    # Known solution
    bodies = [
        Body(fig_condition["mass"][i], fig_condition["position"][i], fig_condition["velocity"][i]) for i in range(3)
        #Body(lagrange_condition["mass"][i], lagrange_condition["position"][i], lagrange_condition["velocity"][i]) for i in range(3)
        #Body(euler_condition["mass"][i], euler_condition["position"][i], euler_condition["velocity"][i]) for i in range(3)
    ]

    history = simulate(bodies, total_time=10, dt=0.01, method="rk4")

    animate_trajectories(history)

