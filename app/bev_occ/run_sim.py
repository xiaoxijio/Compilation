#!/usr/bin/env python3
"""Run the official SDK simulator in a unique directory and archive its results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4
from sim_report import write_summary
from model_paths import resolve_onnx, config_path

HERE = Path(__file__).resolve().parent


def toolchain_root():
    if os.environ.get('NPU_TOOLCHAIN_PATH'):
        return Path(os.environ['NPU_TOOLCHAIN_PATH']).resolve()
    for parent in HERE.parents:
        candidate = parent / 'npu_toolchain'
        if (candidate / 'linux/simulator/lib/libotsimulator.so').is_file():
            return candidate
    raise ValueError('Set NPU_TOOLCHAIN_PATH to the SDK npu_toolchain directory')


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_args = parser.add_mutually_exclusive_group()
    model_args.add_argument('--model', type=Path, help='Path to a compiled .ortm')
    model_args.add_argument('--model-name', help='Basename in tmp_models, e.g. bev_occ_mapping; default g3')
    parser.add_argument('--tag', default=None, help='Experiment name, e.g. baseline/g1/g2/g3/g4')
    parser.add_argument('--mode', choices=('perf', 'infer', 'dump'), default='perf',
                        help='perf: estimate cycles/FPS; infer: run one real input; dump: also dump intermediate layers')
    parser.add_argument('--input', type=Path, help='One preprocessed FP32 6x3x256x704 .bin; required for infer/dump')
    parser.add_argument('--out-root', type=Path, default=HERE / 'results')
    parser.add_argument('--timeout', type=int, default=1800, help='PC process timeout in seconds')
    parser.add_argument('--compile-config', type=Path, help='Optional exact OMG YAML used for this model (archived)')
    parser.add_argument('--onnx', type=Path, help='Model name or ONNX path; auto-select ORTM, config and result tag')
    args = parser.parse_args()
    if args.compile_config is not None and args.compile_config.parent == Path('.'):
        args.compile_config = HERE / 'input_cfg' / args.compile_config.name
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.mode == 'perf' and args.input is not None:
        parser.error('perf estimates do not consume image data; use --mode infer with --input')
    if args.mode != 'perf' and args.input is None:
        parser.error('--input is required for infer/dump')
    try:
        toolchain = toolchain_root()
        if args.onnx:
            args.onnx = resolve_onnx(args.onnx, HERE)
        model_name = args.model_name or (args.onnx.stem if args.onnx else 'g3')
        for suffix in ('.onnx', '.ortm'):
            if model_name.endswith(suffix):
                model_name = model_name[:-len(suffix)]
                break
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', model_name):
            parser.error('--model-name must be a basename, not a path')
        model = (args.model or HERE / 'tmp_models' / (model_name + '.ortm')).resolve(strict=True)
        if model.suffix != '.ortm' or '_fake_quant' in model.stem:
            parser.error('Use the compiled .ortm, not ONNX or the _fake_quant.ortm')
        binary = HERE / 'build/ortm_run_sim_model'
        if not binary.is_file():
            parser.error('Run bash 04_build.sh first')
        if args.tag is None:
            args.tag = model.stem
        if args.onnx is None and args.model is None:
            candidate = HERE / 'tmp_models' / (model_name + '.onnx')
            if candidate.is_file():
                args.onnx = candidate
        config_origin = 'explicit' if args.compile_config else None
        if args.compile_config is None:
            saved = model.with_suffix('.compile.yml')
            current = config_path(model, HERE)
            if saved.is_file():
                args.compile_config, config_origin = saved, 'archived_at_compile'
            elif current.is_file():
                args.compile_config, config_origin = current, 'current_config_not_verified_against_ortm'
        model_identity = identity(model)
        input_identity = identity(args.input) if args.input else None
        # The exported joint model has a single normalized FP32 image input.
        if input_identity and input_identity['bytes'] != 6 * 3 * 256 * 704 * 4:
            parser.error('Expected one FP32 [6,3,256,704] input binary (12976128 bytes)')
        config_identity = identity(args.compile_config) if args.compile_config else None
        onnx_identity = identity(args.onnx) if args.onnx else None
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    safe_tag = re.sub(r'[^A-Za-z0-9_.-]', '_', args.tag)[:80] or 'run'
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid4().hex[:8]
    run_dir = args.out_root.resolve() / safe_tag / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    command = [str(binary), '-m', str(model)]
    if args.mode == 'perf':
        command += ['-p']
    else:
        command += ['-i', input_identity['path'], '-o', str(run_dir / 'outputs')]
        if args.mode == 'dump':
            command += ['-d']
    manifest = dict(tag=args.tag, run_id=run_id, mode=args.mode, status='running',
                    started_at_utc=datetime.now(timezone.utc).isoformat(), model=model_identity,
                    input=input_identity, compile_config=config_identity, compile_config_origin=config_origin, onnx=onnx_identity,
                    runner=identity(binary), toolchain=str(toolchain), command=command,
                    scope='single model; no application postprocessing',
                    note='perf mode uses no image input and does not verify numerical correctness.')
    if args.compile_config:
        (run_dir / 'compile_config.yml').write_bytes(args.compile_config.read_bytes())
    info = model.with_name(model.stem + '_info.json')
    if info.is_file():
        manifest['compiler_info'] = identity(info)
        (run_dir / 'compiler_info.json').write_bytes(info.read_bytes())
    manifest_path = run_dir / 'run_metadata.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(toolchain / 'linux/simulator/lib') + (
        ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    print('Mode:', args.mode, '\nResults:', run_dir, flush=True)
    started = time.monotonic()
    result = 1
    try:
        with (run_dir / 'simulator.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=run_dir, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                result = process.wait(timeout=args.timeout)
                manifest['status'] = 'complete' if result == 0 else 'failed'
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                manifest['status'] = 'timeout' if isinstance(exc, subprocess.TimeoutExpired) else 'interrupted'
                result = 124 if isinstance(exc, subprocess.TimeoutExpired) else 130
    except OSError as exc:
        manifest['status'], manifest['error'] = 'failed', str(exc)
    manifest['pc_process_wall_seconds'] = time.monotonic() - started
    manifest['returncode'] = result
    manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    # Runtime success alone is insufficient: require a real estimate or all outputs.
    summary = write_summary(run_dir, manifest)
    if result == 0 and args.mode == 'perf' and 'estimated_single_core_fps' not in summary:
        result, manifest['status'] = 2, 'missing_profile'
    if result == 0 and args.mode != 'perf':
        log = (run_dir / 'simulator.log').read_text(errors='replace')
        tensors = re.findall(r'\bdump output path: (.+)', log)
        output_section = log.split('outputs:\n', 1)[-1].split('[I]', 1)[0]
        descriptors = re.findall(r'\s*(\d+): dtype: (\w+) shape: ([\dx]+) stride: (\d+) size: (\d+)', output_section)
        manifest['output_descriptors'] = [dict(index=int(i), dtype=dtype, shape=shape,
            stride=int(stride), runtime_bytes=int(size)) for i, dtype, shape, stride, size in descriptors]
        if (not descriptors or len(tensors) != len(descriptors)
                or any(not Path(p.strip()).is_file() or Path(p.strip()).stat().st_size == 0 for p in tensors)):
            result, manifest['status'] = 2, 'missing_outputs'
        manifest['output_files'] = tensors
    manifest['returncode'] = result
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    summary = write_summary(run_dir, manifest)
    print((run_dir / 'summary.txt').read_text().split('Top nodes')[0], flush=True)
    print('Details:', run_dir / 'summary.txt')
    if result:
        print('Simulation did not complete successfully; inspect:', run_dir / 'simulator.log', file=sys.stderr)
    return result


if __name__ == '__main__':
    sys.exit(main())
