"""Reduced rowing model reconstructed from manuscript appendix E.1.

This implements the listed surge, prescribed slide and pitch equations; it is
not an empirical validation or a byte-for-byte recovery of the PDF source.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from slide_kinematics import slide_trajectory


@dataclass(frozen=True)
class Parameters:
    hull_mass: float = 250.0
    crew_mass: float = 500.0
    drag: float = 40.0
    period: float = 2.0
    drive_fraction: float = 0.45
    peak_force: float = 3000.0
    beta_a: float = 2.5
    beta_b: float = 3.0
    slide_travel: float = 0.85
    pitch_inertia: float = 1200.0
    pitch_stiffness: float = 5e4
    pitch_damping: float = 2500.0
    trim_moment: float = 1.8e4
    pitch_drag: float = 2.5
    pitch_rate_drag: float = 0.15
    recovery_shape: str = 'cosine'
    recovery_blend: float = 0.0


def thrust(t, p):
    phase = (t % p.period)/p.period
    u = phase/p.drive_fraction
    if u >= 1 or u <= 0:
        return 0.0
    # Beta(2.5, 3) shape normalized at its mode, as in the listing.
    a, b = p.beta_a, p.beta_b
    if a <= 1 or b <= 1: raise ValueError("beta shape parameters must exceed one")
    mode = (a-1)/(a+b-2)
    return p.peak_force * u**(a-1)*(1-u)**(b-1)/(mode**(a-1)*(1-mode)**(b-1))


def simulate(p=Parameters(), duration=20.0, dt=0.01, initial_speed=0.0):
    if min(p.hull_mass, p.crew_mass, p.period, p.pitch_inertia, duration, dt) <= 0 or p.drag < 0:
        raise ValueError('masses, period, inertia and times must be positive; drag nonnegative')
    if not 0 < p.drive_fraction < 1:
        raise ValueError('drive fraction must be in (0, 1)')
    mass = p.hull_mass+p.crew_mass
    def rhs(t, y):
        v, theta, omega = y
        phase = np.array([(t % p.period)/p.period])
        s, _, d2s = slide_trajectory(phase, p.drive_fraction, p.slide_travel, p.recovery_shape, p.recovery_blend)
        drag = p.drag * (1+p.pitch_drag*theta**2+p.pitch_rate_drag*omega**2)*v*abs(v)
        return [(thrust(t,p)-drag-p.crew_mass*d2s[0]/p.period**2)/mass,
                omega, (p.trim_moment*(s[0]-p.slide_travel/2)-p.pitch_stiffness*theta-p.pitch_damping*omega)/p.pitch_inertia]
    t = np.linspace(0, duration, int(np.ceil(duration/dt))+1)
    sol = solve_ivp(rhs, (0,duration), [initial_speed,0,0], t_eval=t, rtol=1e-8, atol=1e-10, max_step=dt)
    if not sol.success: raise RuntimeError(sol.message)
    return {'time': sol.t, 'velocity': sol.y[0], 'pitch': sol.y[1], 'pitch_rate': sol.y[2]}
