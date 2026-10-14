import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/protocol'))
from route_replay import Fix,replay,fidelity
from calibrate_drift import calibrate

class ReplayTests(unittest.TestCase):
    def run_route(self,policy,level='L0',**kw):
        truth=np.column_stack([np.arange(40)*100,np.zeros(40)])
        return replay(list(range(40)),truth,calibrate()['levels'][level],91001,policy,
            lambda frame,estimate,radius:dict(texture=.4,raw=.6,cal=.7),
            lambda frame,estimate,radius:Fix(False,None),**kw)

    def test_l0_and_matched_attempts(self):
        for policy in ('P0_equal','P1_random'):
            r=self.run_route(policy,exact_attempts=7)
            self.assertEqual(r['metrics']['attempts'],7)
            self.assertEqual(r['metrics']['fraction_within_100m'],1)

    def test_repeatable(self):
        self.assertEqual(self.run_route('P4_calibrated','L2'),self.run_route('P4_calibrated','L2'))

    def test_fidelity_fails_empty_and_no_accepts(self):
        with self.assertRaises(ValueError):fidelity([])
        rows=[dict(attempt_id=i,live_outcome='rejected',lookup_outcome='rejected') for i in range(2000)]
        self.assertFalse(fidelity(rows)['pass'])
        rows[0].update(live_outcome='accepted_correct',lookup_outcome='accepted_correct',position_difference_m=0)
        self.assertTrue(fidelity(rows)['pass'])

    def test_calibration(self):
        for x in calibrate()['levels'].values():
            self.assertAlmostEqual(x['simulated_ar1_rms_1km_m'],x['target_ar1_rms_1km_m'],delta=.5)

if __name__=='__main__':unittest.main()
