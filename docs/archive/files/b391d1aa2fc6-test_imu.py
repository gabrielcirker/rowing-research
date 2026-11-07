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

class SegmentationChecks(unittest.TestCase):
    def test_synthetic_periodic_pulses(self):
        from imu_processing import segment_strokes
        t=np.arange(0,30,.01)
        pulses=np.zeros_like(t)
        for center in np.arange(1,30,2): pulses+=np.exp(-((t-center)/.12)**2)
        a=np.column_stack((pulses,np.zeros_like(t),np.zeros_like(t)))
        peaks=segment_strokes(t,a,100)
        self.assertGreaterEqual(len(peaks),13)
        self.assertLess(np.max(np.abs(np.diff(t[peaks])-2)),.1)

class FeatureChecks(unittest.TestCase):
    def test_event_features_and_relative_index(self):
        from imu_processing import stroke_features,relative_disturbance_index
        t=np.arange(0,6,.01); a=np.zeros((len(t),3)); w=np.zeros_like(a); e=np.zeros((len(t),2))
        f=stroke_features(t,a,w,e,np.array([0,200,400]),100)
        np.testing.assert_allclose(f['duration'],[2,2])
        np.testing.assert_allclose(relative_disturbance_index(f),[0,0])

class SyntheticIMUChecks(unittest.TestCase):
    def test_static_gravity_convention(self):
        from imu_processing import synthetic_hull_imu
        t=np.arange(0,1,.01); zeros=np.zeros_like(t)
        _,a,w=synthetic_hull_imu({'time':t,'velocity':zeros,'pitch':zeros,'pitch_rate':zeros})
        np.testing.assert_allclose(a,np.tile([0,0,G],(len(t),1)),atol=1e-12)
        _,lin=complementary_orientation(a,w,100)
        np.testing.assert_allclose(lin,0,atol=1e-12)
