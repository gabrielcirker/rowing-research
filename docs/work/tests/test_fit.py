import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from multibody_model import Parameters, simulate
from parameter_fit import fit_surrogate

class FitChecks(unittest.TestCase):
    def test_noiseless_synthetic_recovery(self):
        p=Parameters(); r=simulate(p,duration=4,dt=.02)
        sel=slice(None,None,10); fit=fit_surrogate(r['time'][sel],r['velocity'][sel],r['pitch_rate'][sel],dt=.02)
        self.assertTrue(fit.success)
        np.testing.assert_allclose(fit.x,[p.pitch_drag,p.trim_moment],rtol=.01)

if __name__ == '__main__': unittest.main()

class FitRobustnessChecks(unittest.TestCase):
    def test_noise_and_drag_mismatch(self):
        from dataclasses import replace
        p=Parameters(); r=simulate(p,duration=4,dt=.02); sel=slice(None,None,10)
        t=r['time'][sel]; rng=np.random.default_rng(2026)
        v=r['velocity'][sel]+rng.normal(0,.01,len(t)); w=r['pitch_rate'][sel]+rng.normal(0,.002,len(t))
        same=fit_surrogate(t,v,w,base=p,dt=.02)
        mismatch=fit_surrogate(t,v,w,base=replace(p,drag=p.drag*1.1),dt=.02)
        self.assertTrue(same.success and mismatch.success)
        self.assertLess(abs(same.x[0]-p.pitch_drag),.5)
        self.assertLess(mismatch.x[0],.1)
        self.assertGreater(mismatch.cost,same.cost)
