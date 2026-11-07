
import numpy as np
import matplotlib.pyplot as plt

def rowing_simulation(
    total_time: float = 20.0,
    dt: float = 0.01,
    mass: float = 766.5,
    k_drag: float = 40.0,
    F_drive: float = 3000.0,
    drive_time: float = 0.9,
    cycle_time: float = 2.0,
    initial_velocity: float = 0.0,
):
    n_steps = int(total_time / dt) + 1
    time = np.linspace(0.0, total_time, n_steps)
    velocity = np.zeros(n_steps)
    thrust_series = np.zeros(n_steps)

    velocity[0] = initial_velocity
    energy_input = 0.0
    energy_drag = 0.0

    for i in range(1, n_steps):
        t = time[i]
        phase = t % cycle_time
        if phase < drive_time:
            F_t = F_drive
        else:
            F_t = 0.0
        thrust_series[i] = F_t

        v_prev = velocity[i - 1]
        F_d = k_drag * v_prev ** 2
        a = (F_t - F_d) / mass
        velocity[i] = v_prev + a * dt
        if velocity[i] < 0.0:
            velocity[i] = 0.0

        P_in = F_t * v_prev
        P_drag = F_d * v_prev
        energy_input += P_in * dt
        energy_drag += P_drag * dt

    E_initial = 0.5 * mass * velocity[0] ** 2
    E_final = 0.5 * mass * velocity[-1] ** 2
    delta_E_k = E_final - E_initial
    efficiency = delta_E_k / energy_input if energy_input > 0.0 else 0.0

    half_idx = n_steps // 2
    avg_velocity = float(np.mean(velocity[half_idx:]))
    max_velocity = float(np.max(velocity))

    return {
        "time": time,
        "velocity": velocity,
        "thrust": thrust_series,
        "energy_input": energy_input,
        "energy_drag": energy_drag,
        "delta_E_k": delta_E_k,
        "efficiency": efficiency,
        "avg_velocity": avg_velocity,
        "max_velocity": max_velocity,
    }

def main() -> None:
    results = rowing_simulation()

    time = results["time"]
    velocity = results["velocity"]
    thrust = results["thrust"]

    plt.figure()
    plt.plot(time, velocity, label="Boat speed")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Boat Velocity vs Time")
    plt.grid(True)
    plt.savefig("/mnt/data/speed_plot.png")
    plt.close()

    plt.figure()
    plt.step(time, thrust, where="post", label="Thrust")
    plt.xlabel("Time (s)")
    plt.ylabel("Thrust (N)")
    plt.title("Thrust vs Time")
    plt.grid(True)
    plt.savefig("/mnt/data/thrust_plot.png")
    plt.close()

    with open("/mnt/data/rowing_results_summary.txt", "w") as fout:
        fout.write("Rowing simulation results\n")
        fout.write("==========================\n")
        fout.write(f"Total simulation time: {results['time'][-1]:.2f} s\n")
        fout.write(f"Average velocity (last half): {results['avg_velocity']:.3f} m/s\n")
        fout.write(f"Maximum velocity: {results['max_velocity']:.3f} m/s\n")
        fout.write(f"Total energy input by rowers: {results['energy_input']:.1f} J\n")
        fout.write(f"Energy dissipated by drag: {results['energy_drag']:.1f} J\n")
        fout.write(f"Change in kinetic energy: {results['delta_E_k']:.1f} J\n")
        fout.write(f"Mechanical efficiency: {results['efficiency']*100:.2f} %\n")

if __name__ == "__main__":
    main()
