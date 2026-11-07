"""Adapted figure reconstruction based on 82-page manuscript pp. 68–70.

The original uses RK4 and negative vertical synthetic gravity. This version uses
adaptive integration and the preprocessing pipeline's positive-g convention.
"""
from dataclasses import replace
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from multibody_model import Parameters,simulate
from imu_processing import synthetic_hull_imu


def main():
    root=Path(__file__).resolve().parents[1]
    p=replace(Parameters(),peak_force=3200,beta_a=2.6,beta_b=3.2,recovery_shape='blend',recovery_blend=.25)
    sim=simulate(p,duration=18,dt=.01)
    t,a,w=synthetic_hull_imu(sim)
    cut=int(.35*len(t)); sl=slice(cut,None)
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(t[sl],sim['velocity'][sl],label='Hull speed')
    ax.set(xlabel='Time (s)',ylabel='Speed (m/s)',title='Synthetic pitch-coupled model')
    right=ax.twinx(); right.plot(t[sl],sim['pitch'][sl],color='tab:orange',label='Pitch'); right.set_ylabel('Pitch (rad)')
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(root/'figures'/'adapted_pitch_and_speed.png',dpi=160); plt.close(fig)
    rng=np.random.default_rng(7); dt=float(np.median(np.diff(t)))
    bias=rng.normal(0,[.15,.15,.01]); walk=np.cumsum(rng.normal(size=(len(t),3))*[.02,.02,.002],axis=0)*np.sqrt(dt)
    truth=np.column_stack((a[:,0],a[:,2],w[:,1])); mock=truth+bias+walk+rng.normal(size=truth.shape)*[.20,.20,.015]
    fig,axes=plt.subplots(3,1,figsize=(7,6),sharex=True)
    for i,(axis,label) in enumerate(zip(axes,['ax (m/s²)','az (m/s²)','gy (rad/s)'])):
        axis.plot(t[sl],truth[sl,i],label='Synthetic'); axis.plot(t[sl],mock[sl,i],alpha=.55,label='Synthetic + noise and drift'); axis.set_ylabel(label);axis.grid(alpha=.3)
    axes[0].legend();axes[-1].set_xlabel('Time (s)');fig.suptitle('Synthetic sensor with simulated measurement errors');fig.tight_layout()
    fig.savefig(root/'figures'/'adapted_synthetic_imu.png',dpi=160);plt.close(fig)
    print('Synthetic only; positive-g convention; original random draw sequence and integrator differ.')

if __name__=='__main__': main()
