"""Slide trajectory reconstructed from the 82-page manuscript, appendix E.1.

Outputs are displacement and its first two derivatives with respect to
stroke phase. This is a prescribed kinematic model, not measured crew motion.
"""
import numpy as np


def slide_trajectory(phase, drive_fraction=0.45, travel=0.85,
                     recovery_shape='cosine', recovery_blend=0.0):
    if not 0 < drive_fraction < 1 or travel < 0:
        raise ValueError('drive fraction must be in (0, 1) and travel nonnegative')
    phase = np.asarray(phase, dtype=float)
    if np.any((phase < 0) | (phase >= 1)):
        raise ValueError('phase must be in [0, 1)')
    s = np.zeros_like(phase)
    ds = np.zeros_like(phase)
    d2s = np.zeros_like(phase)
    drive = phase < drive_fraction
    u = phase[drive] / drive_fraction
    s[drive] = travel * (1-np.cos(np.pi*u))/2
    ds[drive] = travel*np.pi*np.sin(np.pi*u)/(2*drive_fraction)
    d2s[drive] = travel*(np.pi/drive_fraction)**2*np.cos(np.pi*u)/2
    duration = 1-drive_fraction
    u = (phase[~drive]-drive_fraction)/duration
    cosine = (travel*(1+np.cos(np.pi*u))/2,
              -travel*np.pi*np.sin(np.pi*u)/(2*duration),
              -travel*(np.pi/duration)**2*np.cos(np.pi*u)/2)
    poly = (travel*(1-10*u**3+15*u**4-6*u**5),
            -travel*(30*u**2-60*u**3+30*u**4)/duration,
            -travel*(60*u-180*u**2+120*u**3)/duration**2)
    if recovery_shape == 'cosine': weight = 0.0
    elif recovery_shape == 'poly': weight = 1.0
    elif recovery_shape == 'blend': weight = float(np.clip(recovery_blend, 0, 1))
    else: raise ValueError('unknown recovery shape')
    for dest, a, b in zip((s, ds, d2s), cosine, poly):
        dest[~drive] = (1-weight)*a+weight*b
    return s, ds, d2s
