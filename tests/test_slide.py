import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from slide_kinematics import slide_trajectory

class SlideChecks(unittest.TestCase):
    def test_endpoints_and_velocity(self):
        s, ds, _ = slide_trajectory(np.array([0., .45]))
        np.testing.assert_allclose(s, [0, .85], atol=1e-12)
        np.testing.assert_allclose(ds, [0, 0], atol=1e-12)
    def test_analytic_derivatives(self):
        x = np.array([.1, .25, .6, .8]); h = 1e-5
        for shape in ('cosine', 'poly', 'blend'):
            s, ds, d2s = slide_trajectory(x, recovery_shape=shape, recovery_blend=.4)
            plus = slide_trajectory(x+h, recovery_shape=shape, recovery_blend=.4)
            minus = slide_trajectory(x-h, recovery_shape=shape, recovery_blend=.4)
            np.testing.assert_allclose((plus[0]-minus[0])/(2*h), ds, rtol=1e-7, atol=1e-8)
            np.testing.assert_allclose((plus[1]-minus[1])/(2*h), d2s, rtol=1e-7, atol=1e-7)

if __name__ == '__main__': unittest.main()
