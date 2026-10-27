#!/usr/bin/env python3
"""Summarize the latest successful performance run for each experiment tag."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--baseline', default='nchw')
    args = parser.parse_args()
    latest = {}
    for path in sorted(args.root.glob('*/*/summary.json')):
        data = json.loads(path.read_text())
        if data.get('status') == 'complete' and data.get('mode') == 'perf' and data.get('estimated_single_core_latency_ms', 0) > 0:
            latest[data['tag']] = data
    if not latest:
        parser.error('No successful perf summaries under ' + str(args.root))
    baseline = latest.get(args.baseline)
    if baseline is None:
        parser.error('Baseline tag not found: ' + args.baseline)
    columns = ['tag', 'estimated_latency_ms', 'estimated_fps', 'speedup_vs_baseline',
               'latency_reduction_percent', 'nn_clock_cycles', 'ortm_sha256', 'run_dir']
    rows = []
    for tag in [args.baseline] + sorted(set(latest) - {args.baseline}):
        data = latest[tag]
        ms = data['estimated_single_core_latency_ms']
        base_ms = baseline['estimated_single_core_latency_ms']
        rows.append(dict(tag=tag, estimated_latency_ms=ms, estimated_fps=data['estimated_single_core_fps'],
                         speedup_vs_baseline=base_ms / ms,
                         latency_reduction_percent=(1 - ms / base_ms) * 100,
                         nn_clock_cycles=data.get('nn_clock_cycles'),
                         ortm_sha256=data['model']['sha256'], run_dir=data['run_dir']))
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['SDK simulation estimates; not measured board performance.',
             'Latest successful perf run per tag. Compare only identical target/quantization/toolchain conditions.',
             'Tag\tEstimated ms\tEstimated FPS\tSpeedup\tLatency reduction']
    for row in rows:
        lines.append(f"{row['tag']}\t{row['estimated_latency_ms']:.4f}\t{row['estimated_fps']:.5f}\t"
                     f"{row['speedup_vs_baseline']:.3f}x\t{row['latency_reduction_percent']:.2f}%")
    (args.root / 'comparison.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('CSV:', (args.root / 'comparison.csv').resolve())


if __name__ == '__main__':
    main()
