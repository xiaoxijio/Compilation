#!/usr/bin/env python3
"""Build a multi-input calibration YAML/list/binaries from exported input NPZs."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import numpy as np
import onnx
import yaml
from model_paths import resolve_onnx, find_npz, config_path

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_base_config(path, parser):
    if path is None:
        return {
            'quant': {
                'dataset': {'calib_path': 'calibration/dataset_bevdet.txt', 'quant_batch_size': 1},
                'activation': {'algorithm': 'MaxMin', 'symmetric': False, 'quant_type': 'S8'},
                'weight': {'algorithm': 'MaxMin', 'quant_type': 'S8', 'per_channel': True},
            },
            'executor': {'cpu_settings': {'thread_num': 4}, 'gpu_settings': {'dev_id': 0}},
            'omg': {'enable_split_model_by_batch': False},
        }
    try:
        config = yaml.safe_load(path.read_text())
        if not isinstance(config, dict) or not isinstance(config.get('quant', {}).get('dataset'), dict):
            raise ValueError('Expected a YAML mapping with quant.dataset')
        return config
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(f'Cannot load --base-config {path}: {exc}')


def prepare_image_list(dataset, parser):
    source = HERE / '../../data/ImageNet/dataset_bevdet.txt'
    try:
        entries = source.read_text().split()
        rows = []
        for entry in entries:
            if entry.startswith('../../../../data/'):
                entry = '../../data/' + entry[len('../../../../data/'):]
            binary = (HERE / entry).resolve(strict=True)
            rows.append(os.path.relpath(binary, HERE))
        if not rows:
            raise ValueError('Image calibration list is empty')
    except (OSError, ValueError) as exc:
        parser.error(f'Cannot prepare image calibration list from {source}: {exc}')
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text('\n'.join(rows) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', type=Path, default=Path('g3'))
    parser.add_argument('--npz', type=Path, action='append',
                        help='Repeat for multiple real calibration samples; each NPZ must contain all ONNX inputs')
    parser.add_argument('--base-config', type=Path, help='Optional custom base YAML; otherwise use built-in S8 defaults')
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing model calibration directory and config')
    parser.add_argument('--fixed-only', action='store_true', help='Initialize only single-image models; used by 00_init.sh')
    args = parser.parse_args()
    try:
        args.onnx = resolve_onnx(args.onnx, HERE)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.out_dir = args.out_dir or HERE / 'calibration' / args.onnx.stem
    args.config = args.config or config_path(args.onnx, HERE)
    for field in ('config', 'base_config'):
        path = getattr(args, field)
        if path is None:
            continue
        if path.parent == Path('.'):
            path = HERE / 'input_cfg' / path.name
        setattr(args, field, path.resolve())
    model = onnx.load(str(args.onnx), load_external_data=False)
    inputs = []
    for value in model.graph.input:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or not shape or any(n <= 0 for n in shape):
            parser.error('Expected fixed-shape FP32 inputs: ' + value.name)
        inputs.append((value.name, shape))
    fixed = len(inputs) == 1 and inputs[0][0] == 'img'
    if args.fixed_only and not fixed:
        return
    if fixed and not args.npz:
        # The original fixed-geometry workflow used these image binaries directly.
        if args.config.exists():
            print('Using existing config:', args.config)
            return
        config = read_base_config(args.base_config, parser)
        dataset = HERE / config['quant']['dataset']['calib_path']
        if args.base_config is None and not dataset.exists():
            prepare_image_list(dataset, parser)
        if not dataset.is_file() or not dataset.read_text().split():
            parser.error('Missing/empty image calibration list: ' + str(dataset))
        expected_bytes = int(np.prod(inputs[0][1])) * 4
        for name in dataset.read_text().split():
            binary = HERE / name
            if not binary.is_file() or binary.stat().st_size != expected_bytes:
                parser.error('Missing or invalid FP32 image calibration binary: ' + str(binary))
        if not args.config.is_relative_to((HERE / 'input_cfg').resolve()):
            parser.error('--config must be inside ' + str(HERE / 'input_cfg'))
        config['inputs'] = [dict(input_name='img', calib_type='Binary', input_type='Binary',
            data_type='FP32', reverse_rgb_order=False, data_layout='NCHW', shape=inputs[0][1])]
        args.config.parent.mkdir(parents=True, exist_ok=True)
        args.config.write_text('# Fixed geometry: use preprocessed image binaries, no NPZ required.\n' + yaml.safe_dump(config, sort_keys=False))
        print('Prepared fixed-geometry config:', args.config)
        print('Image calibration list:', dataset)
        return
    try:
        args.npz = args.npz or find_npz(args.onnx)
    except ValueError as exc:
        parser.error(str(exc))
    # Validate every NPZ before writing configuration or calibration files.
    samples = []
    for path in args.npz:
        with np.load(path, allow_pickle=False) as data:
            sample = []
            for name, shape in inputs:
                if name not in data or list(data[name].shape) != shape or data[name].dtype != np.float32:
                    parser.error(f'{path}: {name} must be float32 with shape {shape}')
                array = np.ascontiguousarray(data[name], dtype='<f4')
                if not np.isfinite(array).all():
                    parser.error(f'{path}: nonfinite values in {name}')
                sample.append(array)
            samples.append(sample)
    out = args.out_dir.resolve()
    cfg = args.config.resolve()
    calibration_root = (HERE / 'calibration').resolve()
    if out == calibration_root or not out.is_relative_to(calibration_root):
        parser.error('--out-dir must be inside ' + str(HERE / 'calibration'))
    if not cfg.is_relative_to((HERE / 'input_cfg').resolve()):
        parser.error('--config must be inside ' + str(HERE / 'input_cfg'))
    if (cfg.exists() or out.exists()) and not args.overwrite:
        parser.error('Output config/directory already exists; choose new --config and --out-dir to preserve existing files')
    config = read_base_config(args.base_config, parser)
    if out.exists():
        if not out.is_dir():
            parser.error('--out-dir exists but is not a directory: ' + str(out))
        shutil.rmtree(out)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True)
    rows = []
    for i, sample in enumerate(samples):
        directory = out / f'sample_{i:03d}'
        directory.mkdir()
        files = []
        for index, ((name, shape), array) in enumerate(zip(inputs, sample)):
            path = directory / f'{index}_{name}.bin'
            array.tofile(path)
            files.append(os.path.relpath(path, HERE))
        rows.append(' '.join(files))
    dataset = out / 'dataset.txt'
    dataset.write_text('\n'.join(rows) + '\n')
    config['quant']['dataset']['calib_path'] = os.path.relpath(dataset, HERE)
    config['inputs'] = [dict(input_name=name, calib_type='Binary', input_type='Binary',
        data_type='FP32', reverse_rgb_order=False, data_layout='NCHW', shape=shape)
        for name, shape in inputs]
    cfg.write_text('# 多输入 ONNX 校准配置；样本数：' + str(len(samples)) + '\n'
                   '# 输入已预处理，不在此重复处理图像。\n'
                   + yaml.safe_dump(config, sort_keys=False))
    manifest = dict(onnx=str(args.onnx.resolve()), onnx_sha256=digest(args.onnx),
                    input_order=[name for name, _ in inputs],
                    sample_npz=[dict(path=str(p.resolve()), sha256=digest(p)) for p in args.npz],
                    sample_count=len(samples), config=str(cfg),
                    note='Export examples can check compilation; one frame is not representative accuracy calibration.')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Config:', cfg)
    print('Dataset:', dataset)
    print('Samples:', len(samples), '(one example is for bring-up, not accuracy acceptance)')
    print('Input order:', ', '.join(name for name, _ in inputs))


if __name__ == '__main__':
    main()
