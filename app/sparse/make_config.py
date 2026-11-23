"""Generate SDK input definitions from S6 ONNX; image preprocessing is upstream."""
from pathlib import Path
import os
import onnx,yaml
p=Path(__file__).resolve().parent
run=Path(os.environ['S6_RUN_DIR'])
m=onnx.load(str(run/'source.onnx'))
c=dict(quant=dict(dataset=dict(calib_path=str((p/Path(os.environ.get('S6_CALIBRATION_DIR','calibration'))/'dataset.txt').resolve()),quant_batch_size=1),activation=dict(algorithm='MaxMin',symmetric=False,quant_type='S8'),weight=dict(algorithm='MaxMin',quant_type='S8',per_channel=True)),executor=dict(enable_gpu=False,cpu_settings=dict(thread_num=4)),omg=dict(enable_split_model_by_batch=False),inputs=[dict(input_name=v.name,calib_type='Binary',input_type='Binary',data_type='FP32',reverse_rgb_order=False,data_layout='NCHW',shape=[d.dim_value for d in v.type.tensor_type.shape.dim]) for v in m.graph.input])
names = {v['input_name'] for v in c['inputs']}
outputs = {v.name for v in m.graph.output}
s7_front = 'occ_depth_context' in outputs
s7_tail = names == {'occ_bev'} and outputs == {'occ_logits'}
if not (s7_front or s7_tail):
    assert ('projection_mat' in names) == ('occ_ego2img' in names), 'S6 matrices must both be fixed or both be runtime inputs'
if not (s7_front or s7_tail) and 'projection_mat' not in names:
    # Frozen matrices let OMG fuse geometry into a mixed-precision Gemm
    # whose FP16/INT8-per-channel signature this SDK cannot compile.
    c['passes'] = [dict(pass_name='FuseMatMulAddPass', enabled=False)]
(run/'input_cfg.yml').write_text(yaml.safe_dump(c,sort_keys=False))
