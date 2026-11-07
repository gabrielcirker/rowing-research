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


def stroke_features(t, linear_acceleration, angular_rate, euler, indices, fs):
    """Features between candidate events; these do not measure a validated risk."""
    t=np.asarray(t,float); a=np.asarray(linear_acceleration,float); w=np.asarray(angular_rate,float); angles=np.asarray(euler,float)
    idx=np.asarray(indices,dtype=int); n=len(t)
    if a.shape!=(n,3) or w.shape!=(n,3) or angles.shape!=(n,2) or fs<=0 or np.any(idx<0) or np.any(idx>=n) or np.any(np.diff(idx)<=0):
        raise ValueError('invalid channel dimensions or event indices')
    result={key:[] for key in ('duration','acceleration_rms','jerk_rms','pitch_rms','pitch_rate_rms','accel_high_fraction','gyro_high_fraction')}
    def fraction(x):
        power=np.abs(np.fft.rfft(x-np.mean(x)))**2
        freq=np.fft.rfftfreq(len(x),1/fs)
        return float(power[(freq>=2.5)&(freq<=10)].sum()/(power.sum()+1e-12))
    for start,end in zip(idx[:-1],idx[1:]):
        if end-start<=5: continue
        magnitude=np.linalg.norm(a[start:end],axis=1)
        vals=(t[end]-t[start],np.sqrt(np.mean(magnitude**2)),np.sqrt(np.mean(np.gradient(magnitude,1/fs)**2)),np.sqrt(np.mean(angles[start:end,1]**2)),np.sqrt(np.mean(w[start:end,1]**2)),fraction(magnitude),fraction(w[start:end,1]))
        for key,val in zip(result,vals): result[key].append(float(val))
    return {key:np.asarray(val) for key,val in result.items()}


def relative_disturbance_index(features):
    """Within-record standardized score from the listing; not an absolute risk."""
    keys=('jerk_rms','pitch_rms','accel_high_fraction','gyro_high_fraction')
    if not features or not len(features[keys[0]]): return np.array([])
    columns=np.column_stack([features[k] for k in keys])
    standardized=(columns-columns.mean(axis=0))/(columns.std(axis=0)+1e-12)
    return standardized @ np.array([.35,.35,.15,.15])


def process_signals(t, acceleration, angular_rate, fs=100., alpha=.03):
    """Resample, estimate attitude, locate candidate events and summarize strokes."""
    tt,a,w=resample_uniform(t,acceleration,angular_rate,fs)
    angles,lin=complementary_orientation(a,w,fs,alpha)
    idx=segment_strokes(tt,lin,fs)
    features=stroke_features(tt,lin,w,angles,idx,fs)
    return {'time':tt,'angles':angles,'linear_acceleration':lin,'events':idx,'features':features,'relative_index':relative_disturbance_index(features)}


def synthetic_hull_imu(simulation, imu_x=0., imu_z=.25):
    """Approximate planar synthetic sensor in the filter's positive-g convention.

    Adapted from the manuscript's small-angle expression, with gravity sign
    made consistent with complementary_orientation. Not a full rigid-body IMU.
    """
    t=np.asarray(simulation['time'],float); v=np.asarray(simulation['velocity'],float)
    pitch=np.asarray(simulation['pitch'],float); rate=np.asarray(simulation['pitch_rate'],float)
    if len(t)<3 or any(x.shape!=t.shape for x in (v,pitch,rate)) or np.any(np.diff(t)<=0):
        raise ValueError('simulation channels must match increasing time with at least 3 samples')
    dv=np.gradient(v,t); dr=np.gradient(rate,t)
    a=np.column_stack((dv-rate**2*imu_x-dr*imu_z-G*np.sin(pitch),np.zeros(len(t)),-rate**2*imu_z+dr*imu_x+G*np.cos(pitch)))
    w=np.column_stack((np.zeros(len(t)),rate,np.zeros(len(t))))
    return t,a,w
