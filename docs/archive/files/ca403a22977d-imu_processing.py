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


def segment_strokes(t, linear_acceleration, fs, expected_rate_spm=(16.,44.)):
    """Return candidate acceleration-magnitude peaks, not verified catch times."""
    from scipy.signal import butter, sosfiltfilt, find_peaks
    t=np.asarray(t,dtype=float); a=np.asarray(linear_acceleration,dtype=float)
    low,high=expected_rate_spm
    if a.shape != (len(t),3) or len(t)<20 or not 0<low<high or fs<=0:
        raise ValueError('invalid time series, sample rate or stroke-rate bounds')
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(t)) or not np.allclose(np.diff(t),1/fs,rtol=1e-4,atol=1e-8):
        raise ValueError('finite uniformly sampled times and acceleration required')
    band=(.6*low/60,3*high/60)
    if band[1]>=fs/2: raise ValueError('sample rate too low for filter')
    magnitude=np.linalg.norm(a,axis=1)
    sos=butter(3,band,btype='bandpass',fs=fs,output='sos')
    try: filtered=sosfiltfilt(sos,magnitude)
    except ValueError as exc: raise ValueError('time series too short for filter') from exc
    peaks,_=find_peaks(filtered,height=np.median(filtered)+.8*np.std(filtered),distance=max(1,int(.6*60/high*fs)))
    if len(peaks)<3: return peaks
    intervals=np.diff(t[peaks]); keep=np.r_[True,(intervals>.6*60/high)&(intervals<1.4*60/low)]
    return peaks[keep]
