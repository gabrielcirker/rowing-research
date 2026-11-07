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
    total_time: float = 20.0,
    dt: float = 0.01,
    mass: float = 766.5,
    k_drag: float = 40.0,
    F_drive: float = 3000.0,
    drive_time: float = 0.9,
    cycle_time: float = 2.0,
    initial_velocity: float = 0.0,
) -> dict:
    """Run the rowing simulation and return time series and summary statistics.

    Parameters
    ----------
    total_time : float
        Duration of the simulation in seconds.
    dt : float
        Time step for numerical integration.
    mass : float
        Mass of the system (shell + rowers + oars) in kilograms.  A typical
        eight‑person shell has a combined mass of about 766.5 kg【779142687997843†L270-L287】.
    k_drag : float
        Quadratic drag coefficient in N·s²/m².  Experimental data for an 8+ shell
        indicate drag constants on the order of 10; we increase the value to 40
        in the default parameters to yield realistic steady‑state speeds of
        roughly 5–6 m/s while preserving the quadratic dependence of drag on
        velocity【779142687997843†L388-L450】.
    F_drive : float
        Peak thrust applied during the drive phase in newtons.  A value of 3000 N
        approximates eight rowers producing about 375 N each during the drive.
    drive_time : float
        Length of the drive phase in seconds.  A typical stroke has a drive
        lasting roughly 0.9 s followed by a recovery to complete a 2 s cycle.
    cycle_time : float
        Total stroke cycle time (drive + recovery) in seconds.
    initial_velocity : float
        Starting boat speed in m/s.

    Returns
    -------
    dict
        A dictionary containing arrays for time, velocity, thrust and summary
        statistics (average velocity, maximum velocity, total energy input,
        kinetic energy gain, drag energy loss and efficiency).
    """
    n_steps = int(total_time / dt) + 1
    time = np.linspace(0.0, total_time, n_steps)
    velocity = np.zeros(n_steps)
    thrust_series = np.zeros(n_steps)

    # initial conditions
    velocity[0] = initial_velocity

    energy_input = 0.0
    energy_drag = 0.0

    for i in range(1, n_steps):
        t = time[i]
        # Determine phase of stroke
        phase = t % cycle_time
        if phase < drive_time:
            F_t = F_drive
        else:
            F_t = 0.0
        thrust_series[i] = F_t

        # hydrodynamic drag force (quadratic in velocity)
        v_prev = velocity[i - 1]
        F_d = k_drag * v_prev ** 2

        # net acceleration
        a = (F_t - F_d) / mass

        # update velocity using explicit Euler method
        velocity[i] = v_prev + a * dt

        # Ensure non‑negative velocity (prevent reversal due to over‑damping)
        if velocity[i] < 0.0:
            velocity[i] = 0.0

        # instantaneous power input and drag power
        P_in = F_t * v_prev
        P_drag = F_d * v_prev

        # accumulate energies
        energy_input += P_in * dt
        energy_drag += P_drag * dt

    # compute kinetic energy change
    E_initial = 0.5 * mass * velocity[0] ** 2
    E_final = 0.5 * mass * velocity[-1] ** 2
    delta_E_k = E_final - E_initial

    # compute efficiency: fraction of rower energy that appears as kinetic energy
    efficiency = delta_E_k / energy_input if energy_input > 0.0 else 0.0

    # compute average velocity over the second half of the simulation (to exclude
    # transient startup effects)
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
    """Run the simulation, save plots and write summary statistics."""
    results = rowing_simulation()

    time = results["time"]
    velocity = results["velocity"]
    thrust = results["thrust"]
    # Create plots
    # Boat velocity vs. time
    plt.figure()
    plt.plot(time, velocity)
    plt.xlabel("Time (s)")
    plt.ylabel("Boat velocity (m/s)")
    plt.title("Boat velocity vs. time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("speed_plot.png", dpi=300)

    # Thrust vs. time
    plt.figure()
    plt.plot(time, thrust)
    plt.xlabel("Time (s)")
    plt.ylabel("Thrust (N)")
    plt.title("Rowers' thrust vs. time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("thrust_plot.png", dpi=300)

    # Write summary results to a text file
    with open("rowing_results_summary.txt", "w") as f:
        f.write("Rowing simulation results\n")
        f.write("==========================\n")
        f.write(f"Total simulation time: {time[-1]:.2f} s\n")
        f.write(f"Average velocity (last half): {results['avg_velocity']:.3f} m/s\n")
        f.write(f"Maximum velocity: {results['max_velocity']:.3f} m/s\n")
        f.write(f"Total energy input by rowers: {results['energy_input']:.1f} J\n")
        f.write(f"Energy dissipated by drag: {results['energy_drag']:.1f} J\n")
        f.write(f"Change in kinetic energy: {results['delta_E_k']:.1f} J\n")
        f.write(f"Mechanical efficiency: {results['efficiency']*100:.2f} %\n")

if __name__ == "__main__":
    main()