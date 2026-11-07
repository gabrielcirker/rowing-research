"""Selected IMU preprocessing reconstructed from manuscript appendix E.2.

Convention: accelerometer reads +g along its z-axis when level and stationary;
gyroscopes use radians/second. This small-angle filter is a research prototype.
"""
import numpy as np

G = 9.80665


def resample_uniform(t, acceleration, angular_rate, fs):
    t = np.asarray(t, dtype=float)
    a = np.asarray(acceleration, dtype=float)
    w = np.asarray(angular_rate, dtype=float)
    if len(t) < 2 or a.shape != (len(t), 3) or w.shape != a.shape or fs <= 0:
        raise ValueError('expected times and matching N x 3 channels with positive sample rate')
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(w)) or np.any(np.diff(t) <= 0):
        raise ValueError('samples must be finite and times strictly increasing')
    new_t = t[0]+np.arange(int(np.floor((t[-1]-t[0])*fs))+1)/fs
    return new_t, np.column_stack([np.interp(new_t,t,a[:,i]) for i in range(3)]), np.column_stack([np.interp(new_t,t,w[:,i]) for i in range(3)])


def complementary_orientation(acceleration, angular_rate, fs, alpha=.02):
    a=np.asarray(acceleration,dtype=float); w=np.asarray(angular_rate,dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or len(a)==0 or w.shape != a.shape or fs <= 0 or not 0 <= alpha <= 1:
        raise ValueError('expected nonempty matching N x 3 arrays, positive sample rate and alpha in [0,1]')
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(w)):
        raise ValueError('samples must be finite')
    roll=np.zeros(len(a)); pitch=np.zeros(len(a))
    roll[0]=np.arctan2(a[0,1],a[0,2])
    pitch[0]=np.arctan2(-a[0,0],np.hypot(a[0,1],a[0,2]))
    for i in range(1,len(a)):
        roll_a=np.arctan2(a[i,1],a[i,2])
        pitch_a=np.arctan2(-a[i,0],np.hypot(a[i,1],a[i,2]))
        roll[i]=(1-alpha)*(roll[i-1]+w[i,0]/fs)+alpha*roll_a
        pitch[i]=(1-alpha)*(pitch[i-1]+w[i,1]/fs)+alpha*pitch_a
    gravity=G*np.column_stack((-np.sin(pitch), np.sin(roll)*np.cos(pitch), np.cos(roll)*np.cos(pitch)))
    return np.column_stack((roll,pitch)), a-gravity
