"""Save consecutive real S6 inputs, including motion and recurrent ONNX state.
Run from SparseDrive with PYTHONPATH=. using its environment.
This small set is a compilation smoke test, not full INT8 accuracy calibration.
"""
import argparse
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,'/home/fjm/tong/SparseDrive')
from tools.onnx_stage1 import Runner, arguments
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--onnx',default='onnx/s6/s6.onnx')
p.add_argument('--config',default='projects/configs/ep1/s6_det_occ_decoder4_fpn3_fastbev.py')
p.add_argument('--output',default='/home/fjm/tong/Compilation/app/sparse/calibration')
p.add_argument('--frames',type=int,default=4)
a=p.parse_args()
if a.frames < 2: p.error('Use at least a first and a history frame')
r=Runner(arguments('S6 calibration').parse_args(['--onnx',a.onnx,'--config',a.config,'--device','cpu']))
root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
original=r.session; rows=[]
class Capture:
    def get_outputs(self): return original.get_outputs()
    def run(self,names,feed):
        folder=root/('%04d'%len(rows));folder.mkdir(exist_ok=True)
        for old in folder.glob('*.bin'):
            if old.stem not in feed:old.unlink()
        for key,value in feed.items():value.astype(np.float32).tofile(folder/(key+'.bin'))
        rows.append(' '.join(str(folder/(v.name+'.bin')) for v in r.inputs))
        print('Calibration frame',len(rows)-1,'history_valid',feed['history_valid'],flush=True)
        return original.run(names,feed)
r.session=Capture()
for i in range(a.frames): r.infer(i)
(root/'dataset.txt').write_text('\n'.join(rows)+'\n')
