"""Read-only Kaggle status snapshot; never starts or changes a kernel."""
import json
from datetime import datetime, timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT=Path(__file__).resolve().parents[2]

def main():
    api=KaggleApi();api.authenticate()
    owned=[]
    for page in range(1,101):
        batch=api.kernels_list(mine=True,page=page,page_size=100) or []
        owned.extend(json.loads(str(k)) if not isinstance(k,dict) else k for k in batch)
        if len(batch)<100:
            break
    else:
        raise RuntimeError('Owned-kernel listing incomplete; cannot classify missing')
    refs={k['ref'] for k in owned}
    records=[]
    for region in ('01','02','03','04','05','06','08','09','10','11'):
        kernel=f'tosarker/cmbl-s11-labels-{region}'
        record={'region':region,'kernel':kernel}
        try:
            response=api.kernels_status(kernel)
            if not isinstance(response, dict):
                response=json.loads(str(response))
            record['remote']=response
            record['status']='finished' if str(response.get('status','')).lower()=='complete' else 'partial'
        except Exception as exc:
            # Permission-denied does not establish that a kernel does not exist.
            record.update(status='missing' if kernel not in refs else 'inaccessible',error_type=type(exc).__name__,detail=str(exc))
        folder=ROOT/'results/full/labels_v1'/region
        record['locally_fetched']=(folder/'outcomes.csv').exists()
        records.append(record)
        print(region,record['status'],flush=True)
    out={'checked_utc':datetime.now(timezone.utc).isoformat(),'owned_kernel_refs':sorted(refs),'regions':records}
    (ROOT/'reports/stage11_status_2026-10-14.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8')

if __name__=='__main__':main()
