import sys
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from multibody_model import Parameters, simulate, thrust
from slide_kinematics import slide_trajectory

class MultibodyChecks(unittest.TestCase):
    def test_unforced_total_momentum(self):
        p=replace(Parameters(), peak_force=0, drag=0, trim_moment=0)
        r=simulate(p, duration=2, dt=.02, initial_speed=3)
        _, ds, _=slide_trajectory((r['time']%p.period)/p.period, p.drive_fraction, p.slide_travel)
        momentum=(p.hull_mass+p.crew_mass)*r['velocity']+p.crew_mass*ds/p.period
        np.testing.assert_allclose(momentum, momentum[0], atol=.02)
        np.testing.assert_allclose(r['pitch'],0,atol=1e-12)
    def test_thrust_shape(self):
        p=Parameters(); mode=(2.5-1)/(2.5+3-2)
        self.assertAlmostEqual(thrust(mode*p.drive_fraction*p.period,p),p.peak_force)
        self.assertEqual(thrust(1.5,p),0)
    def test_step_refinement(self):
        p=Parameters(); a=simulate(p,duration=4,dt=.02); b=simulate(p,duration=4,dt=.01)
        np.testing.assert_allclose(a['velocity'],b['velocity'][::2],atol=2e-4)
        np.testing.assert_allclose(a['pitch'],b['pitch'][::2],atol=2e-5)

if __name__ == '__main__': unittest.main()
