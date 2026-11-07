"""
Advanced utilities built on top of the simple rowing model.

This script generates the additional figures used in the manuscript:

    - drag_effect.png            (effect of k_d on speed & efficiency)
    - mass_effect.png            (effect of mass on velocity profile)
    - efficiency_contour.png     (efficiency vs k_d and stroke rate)
    - blade_comparison_plot.png  (hypothetical blade efficiency curves)

It reuses the core integrator from rowing_model.py.

Usage:
    python rowing_advanced_models.py
"""

import numpy as np
import matplotlib.pyplot as plt

from rowing_model import rowing_simulation


def simulate_for_drag_values(
    k_values=(32.0, 40.0, 48.0),
    total_time=20.0,
    dt=0.001,
    mass=750.0,
    F_drive=3000.0,
    cycle_time=2.0,
    delta=0.45,
):
    """Run the baseline model for a set of drag coefficients."""
    results = []
    for k_d in k_values:
        res = rowing_simulation(
            total_time=total_time,
            dt=dt,
            mass=mass,
            k_drag=k_d,
            F_drive=F_drive,
            cycle_time=cycle_time,
            delta=delta,
        )
        res["k_drag"] = k_d
        results.append(res)
    return results


def simulate_for_masses(
    masses=(675.0, 750.0, 825.0),
    total_time=20.0,
    dt=0.001,
    k_drag=40.0,
    F_drive=3000.0,
    cycle_time=2.0,
    delta=0.45,
):
    """Run the baseline model for a set of total masses."""
    results = []
    for m in masses:
        res = rowing_simulation(
            total_time=total_time,
            dt=dt,
            mass=m,
            k_drag=k_drag,
            F_drive=F_drive,
            cycle_time=cycle_time,
            delta=delta,
        )
        res["mass"] = m
        results.append(res)
    return results


def plot_drag_effect():
    """Generate drag_effect.png: average speed and efficiency vs k_d."""
    k_values = (32.0, 40.0, 48.0)
    sims = simulate_for_drag_values(k_values=k_values)

    avg_speeds = [s["avg_velocity"] for s in sims]
    efficiencies = [s["efficiency"] * 100.0 for s in sims]

    fig, ax1 = plt.subplots()

    x = np.arange(len(k_values))
    width = 0.35

    # Bars for average speed
    bar1 = ax1.bar(x - width / 2, avg_speeds, width)
    ax1.set_xlabel("Drag coefficient $k_d$ (N·s$^2$/m$^2$)")
    ax1.set_ylabel("Average speed (m/s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{kv:.0f}" for kv in k_values])
    ax1.set_title("Effect of drag coefficient on speed and efficiency")

    # Second axis for efficiency
    ax2 = ax1.twinx()
    bar2 = ax2.bar(x + width / 2, efficiencies, width)
    ax2.set_ylabel("Efficiency (%)")

    # Annotate bars with numerical values
    for b in bar1:
        height = b.get_height()
        ax1.annotate(
            f"{height:.2f}",
            xy=(b.get_x() + b.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for b in bar2:
        height = b.get_height()
        ax2.annotate(
            f"{height:.1f}",
            xy=(b.get_x() + b.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    plt.savefig("drag_effect.png", dpi=300)


def plot_mass_effect():
    """Generate mass_effect.png: velocity vs time for different masses."""
    masses = (675.0, 750.0, 825.0)
    sims = simulate_for_masses(masses=masses)

    plt.figure()
    for res in sims:
        time = res["time"]
        vel = res["velocity"]
        m = res["mass"]
        plt.plot(time, vel, label=f"m = {m:.0f} kg")

    plt.xlabel("Time (s)")
    plt.ylabel("Boat velocity (m/s)")
    plt.title("Effect of total mass on transient acceleration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mass_effect.png", dpi=300)


def compute_efficiency_grid(
    k_values=(32.0, 36.0, 40.0, 44.0, 48.0),
    stroke_rates=(24.0, 28.0, 32.0, 36.0),
    total_time=30.0,
    dt=0.001,
    mass=750.0,
    F_drive=3000.0,
    delta=0.45,
):
    """
    Compute efficiency for a grid of (k_d, stroke_rate) combinations.

    Returns
    -------
    K, R, eta_grid
        2D meshgrids of k_d, stroke_rate and the corresponding efficiency (%).
    """
    eta_grid = np.zeros((len(stroke_rates), len(k_values)))

    for i, rate in enumerate(stroke_rates):
        cycle_time = 60.0 / rate  # spm -> seconds per stroke
        for j, k_d in enumerate(k_values):
            res = rowing_simulation(
                total_time=total_time,
                dt=dt,
                mass=mass,
                k_drag=k_d,
                F_drive=F_drive,
                cycle_time=cycle_time,
                delta=delta,
            )
            eta_grid[i, j] = res["efficiency"] * 100.0  # percent

    K, R = np.meshgrid(k_values, stroke_rates)
    return K, R, eta_grid


def plot_efficiency_contour():
    """Generate efficiency_contour.png from the efficiency grid."""
    k_values = (32.0, 36.0, 40.0, 44.0, 48.0)
    stroke_rates = (24.0, 28.0, 32.0, 36.0)

    K, R, eta = compute_efficiency_grid(
        k_values=k_values,
        stroke_rates=stroke_rates,
    )

    plt.figure()
    cp = plt.contourf(K, R, eta, levels=15)
    plt.xlabel("Drag coefficient $k_d$ (N·s$^2$/m$^2$)")
    plt.ylabel("Stroke rate (spm)")
    plt.title("Efficiency landscape vs drag and stroke rate")
    cbar = plt.colorbar(cp)
    cbar.set_label("Efficiency (%)")
    plt.tight_layout()
    plt.savefig("efficiency_contour.png", dpi=300)


def plot_blade_efficiency_curves():
    """Generate blade_comparison_plot.png with hypothetical blade curves."""
    # "Slip" or effective loading parameter from 0 to 1
    x = np.linspace(0.0, 1.0, 200)

    # Hypothetical efficiency curves (illustrative only)
    eff_conventional = 0.75 * (1 - (x - 0.6) ** 2 / 0.36)
    eff_conventional = np.clip(eff_conventional, 0.0, None)

    eff_high_aspect = 0.82 * np.exp(-((x - 0.55) ** 2) / (2 * 0.12**2))
    eff_big_blade = 0.78 * np.exp(-((x - 0.50) ** 2) / (2 * 0.18**2))

    plt.figure()
    plt.plot(x, eff_conventional, label="Conventional blade")
    plt.plot(x, eff_high_aspect, label="High-aspect blade")
    plt.plot(x, eff_big_blade, label="Large surface blade")

    plt.xlabel("Non-dimensional slip / loading")
    plt.ylabel("Propulsive efficiency (fraction)")
    plt.title("Hypothetical blade efficiency curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("blade_comparison_plot.png", dpi=300)


def main():
    """Run all advanced analyses and generate the PNG figures."""
    plot_drag_effect()
    plot_mass_effect()
    plot_efficiency_contour()
    plot_blade_efficiency_curves()


if __name__ == "__main__":
    main()
