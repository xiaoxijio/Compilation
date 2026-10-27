#!/usr/bin/env python3
"""Prepare cached-map model inputs and preserve exact base-1024 components."""
import argparse
from pathlib import Path
import subprocess
import sys
import onnx
import numpy as np
import yaml
from model_paths import resolve_onnx, find_npz, config_path

HERE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx', type=Path, default=Path('g3'))
    p.add_argument('--npz', type=Path, action='append')
    p.add_argument('--out-dir', type=Path, default=None)
    p.add_argument('--config', type=Path, default=None)
    p.add_argument('--base-config', type=Path, help='Optional custom base YAML; otherwise use built-in S8 defaults')
    a = p.parse_args()
    try:
        a.onnx = resolve_onnx(a.onnx, HERE)
        a.npz = a.npz or find_npz(a.onnx)
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    a.out_dir = a.out_dir or HERE / 'calibration' / a.onnx.stem
    a.config = a.config or config_path(a.onnx, HERE)
    for field in ('config', 'base_config'):
        path = getattr(a, field)
        if path is None:
            continue
        if path.parent == Path('.'):
            path = HERE / 'input_cfg' / path.name
        setattr(a, field, path.resolve())
    model = onnx.load(str(a.onnx), load_external_data=False)
    props = {x.key:x.value for x in model.metadata_props}
    if props.get('geometry_mode') != 'mapping':
        p.error('Expected --runtime-mapping ONNX')
    if props.get('mapping_version') != '3':
        p.error('仅支持版本 3 的零深度填充映射模型，请使用当前导出脚本重新导出')
    expected = {value.name for value in model.graph.input} | {'mapping_version'}
    for path in a.npz:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != expected:
                p.error(f'{path}: NPZ 字段必须与当前模型输入及版本标记完全匹配')
            if data['mapping_version'].shape != () or data['mapping_version'].item() != 3:
                p.error(f'{path}: 映射版本必须为 3')
    cmd = [sys.executable, str(HERE/'prepare_calibration.py'), '--onnx', str(a.onnx),
           '--out-dir', str(a.out_dir), '--config', str(a.config), '--overwrite']
    if a.base_config is not None:
        cmd += ['--base-config', str(a.base_config)]
    for path in a.npz:
        cmd += ['--npz', str(path)]
    result = subprocess.run(cmd)
    if result.returncode:
        raise SystemExit(result.returncode)
    cfg = yaml.safe_load(a.config.read_text())
    names = {v.name for v in model.graph.input if v.name != 'img'}
    existing = cfg['quant'].get('quant_tensors', [])
    cfg['quant']['quant_tensors'] = [t for t in existing if t['tensor_name'] not in names] + [
        dict(tensor_name=name, quant_type='FP16') for name in sorted(names)]
    a.config.write_text('# Runtime mapping components must retain at least FP16 precision; do not quantize them to S8.\n' + yaml.safe_dump(cfg, sort_keys=False))
    print('Requested FP16 mapping precision; SDK may retain FP32 on integer-index paths.')

if __name__ == '__main__':
    main()
