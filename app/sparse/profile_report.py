"""Profile a compiled SparseDrive ORTM and retain each simulator run."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'bev_occ'))
from sim_report import read_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = HERE / 'models' / (os.getenv('S6_VARIANT', 's6') + '.ortm')
    parser.add_argument('model', type=Path, nargs='?', default=default)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    tag = args.model.stem
    model = args.model.resolve()
    if not model.is_file() or model.suffix != '.ortm' or model.stem.endswith('_fake_quant'):
        parser.error('Provide a compiled .ortm file')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', tag):
        parser.error('The model filename must use letters, digits, underscores, dots or hyphens')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')

    toolchain = Path(os.getenv('NPU_TOOLCHAIN_PATH', '/home/fjm/tong/npu_toolchain'))
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = ':'.join(filter(None, (
        str(toolchain / 'linux/simulator/lib'), env.get('LD_LIBRARY_PATH'))))
    destination = HERE / 'results' / tag
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = Path(tempfile.mkdtemp(prefix=timestamp + '_perf_', dir=destination))
    digest = hashlib.sha256()
    with model.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    print(f'Results: {run}', flush=True)
    start = time.monotonic()
    status = 'failed'
    profile = {}
    with (run / 'simulator.log').open('w') as log:
        try:
            subprocess.run([str(HERE / 'build/ortm_run_sim_model'), '-m', str(model), '-p'],
                           cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=args.timeout)
            profile = read_profile(run / 'oritek_perf_sim.csv')
            if 'estimated_single_core_fps' in profile:
                status = 'complete'
        except (OSError, subprocess.SubprocessError) as error:
            print(error, file=log)

    lines = ['SparseDrive PC simulator', f'Tag: {tag}', f'Status: {status}',
             f'Model: {model}', f'Model SHA256: {digest.hexdigest()}', 'Mode: perf',
             'SDK simulator estimate. Physical board latency requires a board run.']
    if status == 'complete':
        lines.extend([
            f"Estimated single-core FPS: {profile['estimated_single_core_fps']:.5f}",
            f"Estimated single-core latency: {profile['estimated_single_core_latency_ms']:.4f} ms",
            f"NN clock cycles: {profile.get('nn_clock_cycles', 'not reported')}",
            f"Profile nodes: {profile.get('nodes', 'not reported')}"])
    lines.extend([f'PC process wall time (diagnostic only): {time.monotonic() - start:.3f} s',
                  '', 'Top nodes by estimated cycles:'])
    lines.extend(f"{row['cycles']:.0f}\t{row['hardware']}\t{row['node']}"
                 for row in profile.get('top_nodes', []))
    lines.extend(['', 'Raw log: simulator.log', 'Raw profile: oritek_perf_sim.csv'])
    summary = '\n'.join(lines) + '\n'
    (run / 'summary.txt').write_text(summary)
    print(summary, end='')
    return 0 if status == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
