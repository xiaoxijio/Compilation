"""Apply exact S6 SDK rewrites, check them in ORT, and select geometric FP16."""
from pathlib import Path
import os
import sys
import numpy as np,onnx,onnxruntime as ort,yaml
from onnx import helper
from s6_graph import lower_broadcasts,avoid_qat_pattern,geometry_precision_tensors,numeric_comparison_masks
sys.path.insert(0,'/home/fjm/tong/SparseDrive')
from tools.export_ep1_npu import align_cache_rows
from tools.s6_sampling_layout import move_sampling_mask
p=Path(__file__).resolve().parent
run=Path(os.environ['S6_RUN_DIR'])
source=run/'source.onnx';target=run/'model.onnx'
m=onnx.load(str(source))
assert not any(n.domain=='extern_cpu' or 'Scatter' in n.op_type for n in m.graph.node)
m,masks=move_sampling_mask(m)
m=lower_broadcasts(m);m,clips=avoid_qat_pattern(m);m,comparisons=numeric_comparison_masks(m)
onnx.checker.check_model(m);onnx.save(m,str(target))
c=yaml.safe_load((run/'input_cfg.yml').read_text());selected=geometry_precision_tensors(m)
c['quant']['quant_tensors']=[dict(tensor_name=x,quant_type='FP16') for x in selected]
(run/'input_cfg.yml').write_text(yaml.safe_dump(c,sort_keys=False))
opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL;opts.enable_mem_pattern=False;opts.enable_mem_reuse=False;opts.enable_cpu_mem_arena=False
sessions=[]
for graph in (onnx.load(str(source)),m):
    arg=next(n for n in graph.graph.node if n.op_type=='ArgMax')
    graph.graph.output.append(helper.make_tensor_value_info(arg.input[0],onnx.TensorProto.FLOAT,[1,128,128,16,18]))
    sessions.append(ort.InferenceSession(graph.SerializeToString(),opts,providers=['CPUExecutionProvider']))
lines=['S6 compiler rewrites: masks=%d, Clip-to-MinMax=%d, numeric masks=%d, FP16 geometry tensors=%d'%(len(masks),clips,comparisons,len(selected))]
for line in Path(c['quant']['dataset']['calib_path']).read_text().splitlines():
    files=line.split();feed={v.name:np.fromfile(f,np.float32).reshape(v.shape) for v,f in zip(sessions[0].get_inputs(),files)}
    print('Checking frame',Path(files[0]).parent.name,flush=True)
    a,b=[s.run(None,feed) for s in sessions]
    aligned=align_cache_rows(a[:7],b[:7]);errors=[]
    for i,(x,y) in enumerate(zip(a[:7],aligned)):
        if i==3:continue
        np.testing.assert_allclose(x,y,rtol=.003,atol=.003);errors.append(float(np.max(abs(x-y))))
    np.testing.assert_allclose(a[-1],b[-1],rtol=.003,atol=.003)
    different=a[3]!=b[3];gap=np.sort(a[-1],axis=-1)[...,-1]-np.sort(a[-1],axis=-1)[...,-2]
    local_error=np.max(abs(a[-1]-b[-1]),axis=-1)
    assert np.all(gap[different]<=2*local_error[different]+1e-6),'Non-tied OCC category change'
    lines.append('frame=%s max_detector_error=%.8g max_occ_logit_error=%.8g occ_class_changes=%d'%(Path(files[0]).parent.name,max(errors),local_error.max(),different.sum()))
(run/'rewrite_validation.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines),flush=True)
