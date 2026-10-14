"""Run the fixed Stage 10 workload only on a real CUDA RTX 4080 cloud host."""
import argparse
import subprocess
import sys
from pathlib import Path

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',required=True)
    ap.add_argument('--emb-root',required=True)
    ap.add_argument('--torch-home',required=True)
    args=ap.parse_args()
    import torch
    if not torch.cuda.is_available() or '4080' not in torch.cuda.get_device_name(0):
        raise SystemExit('Requires an actual CUDA RTX 4080. No benchmark launched.')
    root=Path(__file__).resolve().parents[2]
    command=[sys.executable,str(root/'kaggle/stage10_runner.py'),'--code-root',str(root),
        '--data-root',args.data_root,'--emb-root',args.emb_root,'--torch-home',args.torch_home,
        '--out','results/stage10/rtx4080_bench','--skip-embed-bench']
    subprocess.check_call(command,cwd=root)
    print('Return the complete results/stage10/rtx4080_bench directory to the PC, then run launch_gate.py.')
