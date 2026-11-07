import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rowing_model import rowing_simulation

class RowingChecks(unittest.TestCase):
    def test_energy_residual_converges(self):
        results = [rowing_simulation(dt=dt) for dt in (0.01, 0.005, 0.001)]
        residuals = [abs(r['energy_input']-r['energy_drag']-r['delta_E_k']) for r in results]
        self.assertGreater(residuals[0], residuals[1])
        self.assertGreater(residuals[1], residuals[2])
        self.assertLess(abs(results[0]['avg_velocity']-results[2]['avg_velocity']), 0.01)

    def test_zero_forces(self):
        r = rowing_simulation(F_drive=0, k_drag=0, initial_velocity=2)
        self.assertTrue(all(abs(v-2)<1e-12 for v in r['velocity']))
        self.assertEqual(r['delta_E_k'], 0)

if __name__ == '__main__': unittest.main()
