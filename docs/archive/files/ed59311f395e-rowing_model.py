"""
Simulation of Mechanical Energy Transfer in Rowing
-------------------------------------------------

This script implements a simple dynamic model of an eight‑person racing shell
propelled by rowers.  The model balances the thrust generated during the
drive phase of each stroke against the hydrodynamic drag on the hull.

The governing equation for the boat’s forward motion is derived from
Newton’s second law.  At each time step we compute the net force on the
shell and update its velocity accordingly.  Mechanical energy input from
the rowers and energy dissipated by drag are accumulated over time, which
enables an estimate of the overall transfer efficiency.

Key assumptions:

* The boat mass (shell + rowers + oars) is constant throughout the
  simulation.  This mass is taken from experimental measurements reported
  in a study of a Mens 8+ racing shell【779142687997843†L270-L287】.
* Hydrodynamic resistance is approximated by a quadratic drag law of the form
  ``F_drag = k_drag * v**2``.  A range of drag coefficients around 8–14 N·s²/m²
  was measured for an 8+ shell【779142687997843†L388-L450】; here we adopt
  ``k_drag = 10.0`` as a representative value.
* Rowers apply a constant thrust ``F_drive`` during the drive portion of each
  stroke and no thrust during the recovery.  This crude forcing function
  reproduces the alternating acceleration and deceleration that characterises
  on‑water rowing.

The output consists of two figures (velocity vs. time and thrust vs. time)
and a plain‑text summary of key statistics.  To run the simulation and
generate the output, execute this script with Python 3.  The results will
be saved in the current directory.
"""

import numpy as np
import matplotlib.pyplot as plt


def rowing_simulation(
    total_time=20.0,
    dt=0.001,
    mass=750.0,
    k_drag=40.0,
    F_drive=3000.0,
    drive_time=None,
    cycle_time=2.0,
    delta=0.45,
    initial_velocity=0.0,
):
    """
    Run the baseline rowing simulation and return time series and summary statistics.

    Model:
        m dv/dt = F_thrust(t) - k_drag v^2

    Parameters
    ----------
    total_time : float
        Duration of the simulation in seconds (default 20 s).
    dt : float
        Time step for numerical integration (default 0.001 s).
    mass : float
        Mass of the system (boat + crew) in kg (default 750).
    k_drag : float
        Quadratic drag coefficient in N·s^2/m^2 (default 40).
    F_drive : float
        Thrust applied during the drive phase in N (default 3000).
    drive_time : float or None
        Duration of the drive in seconds. If None, uses drive_time = delta * cycle_time.
    cycle_time : float
        Stroke period (drive + recovery) in s (default 2.0 s = 30 spm).
    delta : float
        Drive duty fraction of the stroke (portion of the stroke spent driving).
        Ignored if drive_time is provided explicitly.
    initial_velocity : float
        Initial boat speed in m/s (default 0).

    Returns
    -------
    dict with keys:
        "time", "velocity", "thrust",
        "energy_input", "energy_drag", "delta_E_k",
        "efficiency", "avg_velocity", "max_velocity"
    """
    # If drive_time not explicitly given, compute from delta and cycle_time
    if drive_time is None:
        drive_time = delta * cycle_time

    n_steps = int(total_time / dt) + 1
    time = np.linspace(0.0, total_time, n_steps)
    velocity = np.zeros(n_steps)
    thrust_series = np.zeros(n_steps)

    velocity[0] = initial_velocity

    energy_input = 0.0
    energy_drag = 0.0

    for i in range(1, n_steps):
        t = time[i]

        # Stroke phase within [0, cycle_time)
        phase = t % cycle_time
        if phase < drive_time:
            F_t = F_drive
        else:
            F_t = 0.0
        thrust_series[i] = F_t

        v_prev = velocity[i - 1]
        F_d = k_drag * v_prev**2

        # Acceleration and velocity update (explicit Euler)
        a = (F_t - F_d) / mass
        v_new = v_prev + a * dt

        # Prevent numerical drift into negative velocity
        if v_new < 0.0:
            v_new = 0.0

        velocity[i] = v_new

        # Power and energy bookkeeping
        P_in = F_t * v_prev
        P_drag = F_d * v_prev

        energy_input += P_in * dt
        energy_drag += P_drag * dt

    # Kinetic energy change from start to end
    E_initial = 0.5 * mass * velocity[0] ** 2
    E_final = 0.5 * mass * velocity[-1] ** 2
    delta_E_k = E_final - E_initial

    efficiency = delta_E_k / energy_input if energy_input > 0.0 else 0.0

    # Average and max speed over second half
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


def main():
    """
    Run the baseline simulation, save plots, and write a summary file.

    Outputs (in current directory):
        - speed_plot.png
        - thrust_plot.png
        - rowing_results_summary.txt
    """
    results = rowing_simulation()

    time = results["time"]
    velocity = results["velocity"]
    thrust = results["thrust"]

    # --- Figure 1: boat speed vs time ---
    plt.figure()
    plt.plot(time, velocity)
    plt.xlabel("Time (s)")
    plt.ylabel("Boat velocity (m/s)")
    plt.title("Boat velocity vs time (baseline 8+)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("speed_plot.png", dpi=300)

    # --- Figure 2: thrust vs time ---
    plt.figure()
    plt.plot(time, thrust)
    plt.xlabel("Time (s)")
    plt.ylabel("Thrust (N)")
    plt.title("Rowers' thrust vs time (square-wave model)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("thrust_plot.png", dpi=300)

    # --- Text summary for sanity checks ---
    with open("rowing_results_summary.txt", "w") as f:
        f.write("Rowing simulation results (baseline)\n")
        f.write("====================================\n")
        f.write(f"Total simulation time: {time[-1]:.2f} s\n")
        f.write(f"Average velocity (last half): {results['avg_velocity']:.3f} m/s\n")
        f.write(f"Maximum velocity: {results['max_velocity']:.3f} m/s\n")
        f.write(f"Total energy input by rowers: {results['energy_input']:.1f} J\n")
        f.write(f"Energy dissipated by drag: {results['energy_drag']:.1f} J\n")
        f.write(f"Kinetic energy gain: {results['delta_E_k']:.1f} J\n")
        f.write(f"Mechanical efficiency: {results['efficiency']*100:.2f} %\n")


if __name__ == "__main__":
    main()
