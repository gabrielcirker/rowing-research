import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from imu_processing import G, complementary_orientation, resample_uniform

class IMUChecks(unittest.TestCase):
    def test_stationary_pitch(self):
        p=.2; a=np.tile([-G*np.sin(p),0,G*np.cos(p)],(100,1)); w=np.zeros_like(a)
        angles, lin=complementary_orientation(a,w,100)
        np.testing.assert_allclose(angles[:,1],p,atol=1e-12)
        np.testing.assert_allclose(lin,0,atol=1e-12)
    def test_resampling_and_duplicate_times(self):
        t=np.array([0.,.1,.2]); x=np.column_stack((t,2*t,3*t))
        tt,a,_=resample_uniform(t,x,x,20)
        np.testing.assert_allclose(a[:,0],tt)
        with self.assertRaises(ValueError): resample_uniform([0,0],x[:2],x[:2],20)

if __name__ == '__main__': unittest.main()
