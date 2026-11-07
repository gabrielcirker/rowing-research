"""Fit selected surrogate parameters to synthetic or measured state observations.

This prototype fits hull speed and pitch rate, not uncalibrated IMU acceleration.
The data and noise scales must be supplied by the caller.
"""
from dataclasses import replace
import numpy as np
from scipy.optimize import least_squares
from multibody_model import simulate, Parameters


def fit_surrogate(time, velocity, pitch_rate, base=Parameters(), scales=(.1,.02), dt=.02, initial=(1.5,10000.)):
    t=np.asarray(time,float); v=np.asarray(velocity,float); w=np.asarray(pitch_rate,float)
    if t.ndim!=1 or len(t)<3 or v.shape!=t.shape or w.shape!=t.shape or not np.isclose(t[0],0) or np.any(np.diff(t)<=0):
        raise ValueError('provide matching observations with strictly increasing time starting at zero')
    if not all(np.all(np.isfinite(x)) for x in (t,v,w)) or len(scales)!=2 or min(scales)<=0:
        raise ValueError('observations must be finite and scales positive')
    def residual(x):
        p=replace(base,pitch_drag=x[0],trim_moment=x[1])
        r=simulate(p,duration=t[-1],dt=dt,initial_speed=v[0])
        return np.r_[(np.interp(t,r['time'],r['velocity'])-v)/scales[0],(np.interp(t,r['time'],r['pitch_rate'])-w)/scales[1]]
    return least_squares(residual,initial,bounds=([0.,1000.],[10.,60000.]),x_scale=[1.,10000.],diff_step=1e-3,ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=50)
