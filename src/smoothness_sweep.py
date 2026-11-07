"""Adapted synthetic version of the appendix E.3 recovery-blend sweep.

Metrics follow the listed operations. This is not the unplotted disturbance
work per distance described in the manuscript text.
"""
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from multibody_model import Parameters,simulate


def main():
    p=replace(Parameters(),peak_force=3200,beta_a=2.6,beta_b=3.2,recovery_shape='blend')
    blends=np.linspace(0,1,7); metrics=[]
    for blend in blends:
        r=simulate(replace(p,recovery_blend=float(blend)),duration=14,dt=.02)
        t=r['time'];v=r['velocity'];a=np.gradient(v,t);jerk=np.gradient(a,t)
        retained=v[int(.2*len(v)):]
        metrics.append((np.sqrt(np.mean(jerk**2)),np.sqrt(np.mean(r['pitch']**2)),np.std(retained)/np.mean(retained)))
    data=np.asarray(metrics)
    fig,axes=plt.subplots(3,1,figsize=(6,7),sharex=True)
    for i,(ax,label) in enumerate(zip(axes,['Jerk RMS (m/s³)','Pitch RMS (rad)','Speed CV'])):
        ax.plot(blends,data[:,i],marker='o');ax.set_ylabel(label);ax.grid(alpha=.3)
    axes[-1].set_xlabel('Recovery blend (0 cosine, 1 polynomial)');fig.suptitle('Adapted synthetic recovery sweep');fig.tight_layout()
    root=Path(__file__).resolve().parents[1];fig.savefig(root/'figures'/'adapted_smoothness_sweep.png',dpi=160);plt.close(fig)
    print('blend,jerk_rms,pitch_rms,speed_cv')
    for x,row in zip(blends,data): print(f'{x:.3f},'+','.join(f'{v:.6f}' for v in row))

if __name__=='__main__': main()
