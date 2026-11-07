"""Reproduce an explicitly synthetic surrogate and parameter-recovery figure."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from multibody_model import Parameters, simulate
from parameter_fit import fit_surrogate


def main():
    p=Parameters(); r=simulate(p,duration=4,dt=.02); sel=slice(None,None,10)
    fit=fit_surrogate(r['time'][sel],r['velocity'][sel],r['pitch_rate'][sel],dt=.02)
    fig,ax=plt.subplots(2,1,figsize=(7,5),sharex=True)
    ax[0].plot(r['time'],r['velocity']); ax[0].set_ylabel('Hull speed (m/s)')
    ax[1].plot(r['time'],r['pitch_rate']); ax[1].set_ylabel('Pitch rate (rad/s)'); ax[1].set_xlabel('Time (s)')
    for axis in ax: axis.grid(alpha=.3)
    fig.suptitle('Synthetic moving-mass and pitch surrogate')
    fig.tight_layout()
    path=Path(__file__).resolve().parents[1]/'figures'/'synthetic_surrogate.png'
    fig.savefig(path,dpi=160); plt.close(fig)
    print(f'Synthetic fit: pitch-drag coefficient {fit.x[0]:.6f}, trim coefficient {fit.x[1]:.3f}; cost {fit.cost:.3g}')

if __name__=='__main__': main()
