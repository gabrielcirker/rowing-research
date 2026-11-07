
import numpy as np
import matplotlib.pyplot as plt
from rowing_model import rowing_simulation

baseline_params = {
    "mass": 766.5,
    "k_drag": 40.0,
    "F_drive": 3000.0,
    "drive_time": 0.9,
    "cycle_time": 2.0,
    "total_time": 20.0,
    "dt": 0.01,
}

scenarios = {
    "baseline": baseline_params,
    "low_drag": { **baseline_params, "k_drag": 20.0 },
    "low_mass": { **baseline_params, "mass": 383.25 },
    "high_rate": { **baseline_params, "cycle_time": 1.5 },
}

def run_profile(params):
    res = rowing_simulation(
        total_time=params["total_time"],
        dt=params["dt"],
        mass=params["mass"],
        k_drag=params["k_drag"],
        F_drive=params["F_drive"],
        drive_time=params["drive_time"],
        cycle_time=params["cycle_time"],
        initial_velocity=0.0,
    )
    return res["time"], res["velocity"], res

profiles = {}
results = {}

for name, params in scenarios.items():
    t, v, res = run_profile(params)
    profiles[name] = (t, v)
    results[name] = res

print("Scenario\tAvg vel (m/s)\tMax vel (m/s)\tEfficiency (%)")
for name, res in results.items():
    avg_v = res["avg_velocity"]
    max_v = res["max_velocity"]
    eff = res["efficiency"] * 100.0
    print(f"{name:8s}\t{avg_v:8.3f}\t{max_v:8.3f}\t{eff:8.2f}")

t_base, v_base = profiles["baseline"]
t_ld, v_ld = profiles["low_drag"]
plt.figure()
plt.plot(t_base, v_base, label="Baseline (k_d=40)")
plt.plot(t_ld, v_ld, label="Lower drag (k_d=20)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Effect of drag on velocity")
plt.legend()
plt.grid(True)
plt.savefig("/mnt/data/drag_effect.png")
plt.close()

t_lm, v_lm = profiles["low_mass"]
plt.figure()
plt.plot(t_base, v_base, label=f"Baseline (mass={baseline_params['mass']} kg)")
plt.plot(t_lm, v_lm, label=f"Lighter (mass={scenarios['low_mass']['mass']} kg)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Effect of mass on velocity")
plt.legend()
plt.grid(True)
plt.savefig("/mnt/data/mass_effect.png")
plt.close()

t_hr, v_hr = profiles["high_rate"]
plt.figure()
plt.plot(t_base, v_base, label="Baseline (~30 spm)")
plt.plot(t_hr, v_hr, label="Higher stroke rate (~40 spm)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Effect of stroke rate on velocity")
plt.legend()
plt.grid(True)
plt.savefig("/mnt/data/stroke_effect.png")
plt.close()

print("Saved plots: drag_effect.png, mass_effect.png, stroke_effect.png")
