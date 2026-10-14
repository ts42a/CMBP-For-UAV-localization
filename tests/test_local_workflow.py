import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/protocol'),str(ROOT/'scripts/predictor'),str(ROOT/'scripts/localization')]
spec=importlib.util.spec_from_file_location('cmbl_launcher',ROOT/'test.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
from replay_io import CachedLocalizer,paired_route_interval,paired_h4_interval
from hypothesis_report import decide
from route_replay import Fix,replay
from runtime_config import validate_protocol,load_settings
from fitted_predictor import FittedPredictor
from evaluate_predictors import fit_predict

class LocalWorkflowTests(unittest.TestCase):
    def test_sample_is_exact_repeatable_contiguous(self):
        rows=[dict(region_folder=r,filename=f'{r}_{i}.jpg',num=str(i)) for r in launcher.REGIONS for i in range(100)]
        a=launcher.select_frames(rows,8,3)
        self.assertEqual(a,launcher.select_frames(rows,8,3));self.assertEqual(len(a['frames']),80)
        for r in launcher.REGIONS:
            nums=[f['num'] for f in a['frames'] if f['region']==r]
            self.assertEqual(nums,list(range(nums[0],nums[0]+8)))
        for percent in (-1,0,float('nan'),101):
            with self.assertRaises(ValueError):launcher.select_frames(rows,percent,3)

    def test_runtime_types_and_unknown_protocol(self):
        p={'schema_version':'unknown'}
        with self.assertRaises(ValueError):validate_protocol(p)
        p={'schema_version':'predictor_protocol_v5','inputs':{'splits':'a','scheme':'b'},
           'metrics':{'bootstrap':'1000 reps seed 3'},'controls':{'shuffle_permutations':True}}
        with self.assertRaises(ValueError):validate_protocol(p)

    def test_durable_cache_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            calls=[]
            def live(*args):calls.append(args);return Fix(True,(1.,2.))
            c=CachedLocalizer(d,{'model':'a'},live);self.assertEqual(c('f',(0,0),100),Fix(True,(1,2)))
            c=CachedLocalizer(d,{'model':'a'},live);c('f',(0,0),100)
            self.assertEqual(len(calls),1)
            c('f',(.00001,0),100);self.assertEqual(len(calls),2)
            with self.assertRaises(ValueError):CachedLocalizer(d,{'model':'b'},live)

    def test_online_fit_equals_batch_and_rejects_leakage(self):
        rng=np.random.default_rng(4);n=90
        d=SimpleNamespace(y=np.arange(n)%2,group=np.array(['G1']*60+['G2']*30),
            near=rng.normal(size=(n,384)),win=rng.normal(size=(n,384)),tex=rng.normal(size=(n,12)),
            extra=rng.normal(size=(n,2)))
        tr=np.arange(60);te=np.arange(60,90)
        for kind in ('M1_texture','M2_dino'):
            m=FittedPredictor(kind,d,tr,'G2')
            np.testing.assert_allclose(m.data(d,te),fit_predict(kind,d,tr,te),rtol=0,atol=1e-12)
            with self.assertRaises(ValueError):FittedPredictor(kind,d,np.arange(90),'G2')

    def test_paired_blocks_and_oracle(self):
        r=paired_route_interval({'a':[0]*20},{'a':[300]*20},reps=30)
        self.assertEqual(r['ci95'],[1,1])
        from calibrate_drift import calibrate
        truth=np.column_stack([np.arange(10)*100,np.zeros(10)])
        r=replay(list(range(10)),truth,calibrate()['levels']['L0'],3,'P6_oracle',
            lambda *x:dict(texture=.5,raw=.5,cal=.5),lambda frame,*x:Fix(True,tuple(truth[frame])),
            budget=.2,oracle=lambda *x:True)
        self.assertEqual(r['metrics']['attempts'],2)

    def test_matched_cap_and_primary_seed_blocks(self):
        from calibrate_drift import calibrate
        truth=np.column_stack([np.arange(10)*100,np.zeros(10)])
        result=replay(list(range(10)),truth,calibrate()['levels']['L0'],3,'P4_calibrated',
            lambda *x:dict(texture=.8,raw=.8,cal=.8),lambda *x:Fix(False,None),attempt_cap=3)
        self.assertEqual(result['metrics']['attempts'],3)
        h4=paired_h4_interval({'r':[[0]*20]*5},{'r':[[300]*20]*5},reps=30)
        self.assertEqual(h4['ci95'],[1,1])

    def test_local_hypothesis_never_confirmatory(self):
        c={'eligible_groups':6,'difference':.03,'ci95':[.01,.05]}
        self.assertEqual(decide([c],'H1')['verdict'],'exploratory_only')
        self.assertEqual(decide([c],'H1',True)['verdict'],'pass')
        self.assertEqual(decide([dict(c,eligible_groups=4)],'H1',True)['verdict'],'inconclusive')

    def test_single_writer_lock(self):
        with tempfile.TemporaryDirectory() as d:
            lock=launcher.acquire_lock(d)
            try:
                with self.assertRaises(RuntimeError):launcher.acquire_lock(d)
            finally:lock.close()

if __name__=='__main__':unittest.main()
