"""One reproducible synthetic noise and drag-mismatch comparison."""
from dataclasses import replace
import numpy as np
from multibody_model import Parameters,simulate
from parameter_fit import fit_surrogate


def main():
    p=Parameters(); r=simulate(p,duration=4,dt=.02); sel=slice(None,None,10)
    t=r['time'][sel]; rng=np.random.default_rng(2026)
    v=r['velocity'][sel]+rng.normal(0,.01,len(t)); w=r['pitch_rate'][sel]+rng.normal(0,.002,len(t))
    for label,base in [('matched drag',p),('drag 10% high',replace(p,drag=p.drag*1.1))]:
        f=fit_surrogate(t,v,w,base=base,dt=.02)
        print(f'{label}: pitch_drag={f.x[0]:.6f}, trim={f.x[1]:.3f}, cost={f.cost:.6f}, success={f.success}')

if __name__=='__main__': main()
