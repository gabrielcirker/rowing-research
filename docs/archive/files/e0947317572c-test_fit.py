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
