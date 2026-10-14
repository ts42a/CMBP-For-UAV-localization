import csv
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/localization'),str(ROOT/'scripts/protocol')]
from integrity import read_csv, validate_outcomes, prepare_run, append_frame_atomic, sha256, encoded_keys, validate_feature_pair
from stage11_acceptance import assess, STAGE11_TAG, LEGACY_STAGE11
from safe_import import stage_zip


def row(k=0):
    return dict(region='01',filename='a.jpg',variant='prior',radius_m='100',offset_k=str(k),
        config_sha256='a'*64,prereg_sha256='b'*64,center_lat='1',center_lon='2',
        offset_east_m='0',offset_north_m='0',runtime_s='1',frame_setup_s='1',frame_cpu_s='1',
        fine_inliers='0',n_window_tiles='0',outcome_primary='rejected',accepted='0',accept_rule='patch_20',
        pred_lat='',pred_lon='',geodesic_error_m='',used_homography='0',geo_verify_status='empty_window')


class OutcomeTests(unittest.TestCase):
    def gate(self,rows,pilot=None):
        return assess(rows,{'a.jpg'},[row()] if pilot is None else pilot,
            {'search_window':{'radii_m':[100]},'prior_offset':{'offsets_per_frame_per_radius':3},'tolerances_m':{'primary':100}},
            {'region':'01','config_sha256':'a'*64,'prereg_sha256':'b'*64,'accept_rule':'patch_20'},
            {'cuda_available':True,'tf32_matmul':False,'tf32_cudnn':False,'bundle_commit':LEGACY_STAGE11},STAGE11_TAG)

    def test_valid_grid_and_rewrite_alias(self):
        self.assertTrue(self.gate([row(k) for k in range(3)])['pass'])

    def test_bad_grid_and_no_overlap(self):
        good=[row(k) for k in range(3)]
        for bad in ([row(k) for k in range(1,4)],good[:-1],good+[row()], [dict(r,variant='wrong') for r in good]):
            with self.subTest(bad=bad),self.assertRaises(ValueError):self.gate(bad)
        with self.assertRaises(ValueError):self.gate(good,[])

    def test_unknown_and_inconsistent_outcomes(self):
        for change in ({'outcome_primary':''},{'outcome_primary':'typo'}, {'accepted':'1'},
                       {'pred_lat':'nan','pred_lon':'2'},{'config_sha256':'bad'}):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_outcomes([dict(row(),**change)])

    def test_truncated_record(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'out.csv';p.write_text('region,filename,variant,radius_m,offset_k,outcome_primary\n01,a.jpg,prior,100,0')
            with self.assertRaises(ValueError):read_csv(p)

    def test_resume_preserves_manifest_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run_manifest.json'
            m={'started_at':'first','experiment':{'config':'a','code':'b','frames':['a']}}
            prepare_run(d,m);old=p.read_bytes()
            prepare_run(d,dict(m,started_at='second'))
            self.assertEqual(old,p.read_bytes())
            for field in ('config','code','frames'):
                changed={'started_at':'third','experiment':dict(m['experiment'],**{field:'different'})}
                with self.assertRaises(ValueError):prepare_run(d,changed)
            self.assertEqual(old,p.read_bytes())

    def test_atomic_frame_failure_retains_previous_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.csv';p.write_text('key,value\na,1')
            old=p.read_bytes()
            with patch('integrity.os.replace',side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):append_frame_atomic(p,[{'key':'b','value':'2'}],['key','value'])
            self.assertEqual(old,p.read_bytes())
            append_frame_atomic(p,[{'key':'b','value':'2'}],['key','value'])
            self.assertEqual(len(read_csv(p)),2)

    def test_archive_rejects_workspace_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ('scripts/label_windowed.py','results/full/labels_v1/01/../../evil','C:/evil'):
                z=Path(d)/'x.zip'
                with zipfile.ZipFile(z,'w') as f:f.writestr(name,'bad')
                with self.assertRaises(ValueError):stage_zip(z,Path(d)/'stage','results/full/labels_v1/01/')

    def test_windows_transient_lock_retries_atomic_commit(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.csv';p.write_text('key,value\na,1\n')
            replace=os.replace
            locked=PermissionError('temporary sharing conflict');locked.winerror=5
            calls=[]
            def transient(src,dst):
                calls.append(1)
                if len(calls)<3:raise locked
                replace(src,dst)
            with patch('integrity.os.replace',side_effect=transient),patch('integrity.time.sleep'):
                append_frame_atomic(p,[{'key':'b','value':'2'}],['key','value'])
            self.assertEqual(len(calls),3)
            self.assertEqual(read_csv(p),[{'key':'a','value':'1'},{'key':'b','value':'2'}])

    def test_windows_persistent_lock_fails_without_data_loss(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.csv';p.write_text('key,value\na,1\n');old=p.read_bytes()
            locked=PermissionError('persistent lock');locked.winerror=32
            with patch('integrity.os.replace',side_effect=locked) as replace,patch('integrity.time.sleep'):
                with self.assertRaises(PermissionError):
                    append_frame_atomic(p,[{'key':'b','value':'2'}],['key','value'])
            self.assertEqual(replace.call_count,21)
            self.assertEqual(p.read_bytes(),old)
            self.assertEqual(list(Path(d).glob('*.tmp')),[])


class FeatureTests(unittest.TestCase):
    def test_key_alignment_and_domains(self):
        import numpy as np
        import pandas as pd
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            records=[dict(row(k),label=k,group_7='G1') for k in (0,1)]
            df=pd.DataFrame(records)
            df.to_csv(p/'features.csv',index=False)
            np.savez(p/'embeddings.npz',near=np.zeros((2,384)),window=np.zeros((2,384)),attempt_keys=encoded_keys(records))
            m={'features_csv_sha256':sha256(p/'features.csv'),'embeddings_npz_sha256':sha256(p/'embeddings.npz')}
            (p/'run_manifest.json').write_text(json.dumps(m))
            with np.load(p/'embeddings.npz') as emb:
                validate_feature_pair(p,df,emb,{'01':'G1'})
                with self.assertRaises(ValueError):validate_feature_pair(p,df.iloc[::-1],emb,{'01':'G1'})
                with self.assertRaises(ValueError):validate_feature_pair(p,df,emb,{'01':'G2'})
                df=df.assign(label=[.5,1.0])
                with self.assertRaises(ValueError):validate_feature_pair(p,df,emb,{'01':'G1'})

    def test_reordered_cache_rejected_despite_matching_count(self):
        import numpy as np
        from dataclasses import asdict
        from dino_embedder import load_embedding_cache
        from tile_index import Tile
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cache.npz'
            tiles=[Tile(i,0,i*512,1024,i*512+1024,512,i*512+512,0,0) for i in range(2)]
            meta={'meta':{'model_name':'dinov2_vits14','image_size':224,'normalize':'l2_on_output'},'extra':{'tiles':[asdict(t) for t in tiles]}}
            p.with_suffix('.meta.json').write_text(json.dumps(meta))
            np.savez(p,embeddings=np.zeros((2,384)),tile_ids=np.array([1,0]))
            with self.assertRaises(ValueError):load_embedding_cache(p,tiles=tiles)


if __name__=='__main__':unittest.main()
