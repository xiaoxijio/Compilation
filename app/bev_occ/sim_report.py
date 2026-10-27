"""Parse the SDK simulator CSV without treating PC wall time as target latency."""
import csv
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def read_profile(path):
    text = Path(path).read_text(encoding='utf-8-sig', errors='replace')
    result = {}
    for key, label in [('graphs', 'graphs'), ('nodes', 'nodes'),
                       ('nn_clock_cycles', 'NN clock cycles'),
                       ('read_band_width_raw', 'NN read band width'),
                       ('write_band_width_raw', 'NN write band width')]:
        match = re.search(r'^\s*([\d.eE+\-]+)\s*,\s*' + label + r'\s*$', text, re.M)
        if match:
            result[key] = int(float(match.group(1)))
    match = re.search(r'([\d.eE+\-]+)\s+fps in LQ560 single core', text)
    if match:
        fps = float(match.group(1))
        if math.isfinite(fps) and fps > 0:
            result['estimated_single_core_fps'] = fps
            result['estimated_single_core_latency_ms'] = 1000. / fps
    layers = []
    for row in csv.DictReader(io.StringIO(text.split('Profiler Stats Info')[0].strip())):
        try:
            cycles = float(row['CYCLE'])
            if not math.isfinite(cycles) or cycles < 0:
                continue
            layers.append(dict(cycles=cycles, hardware=row['HARDWARE'],
                               node=row['NODE'].replace('\x00', '').replace('\\u0000', '')))
        except (KeyError, ValueError, TypeError):
            continue
    result['top_nodes'] = sorted(layers, key=lambda row: row['cycles'], reverse=True)[:20]
    grouped = defaultdict(float)
    for row in layers:
        grouped[row['hardware'] + ': ' + row['node']] += row['cycles']
    result['cycles_by_kernel'] = dict(sorted(grouped.items(), key=lambda item: item[1], reverse=True))
    result['parsed_layer_rows'] = len(layers)
    return result


def write_summary(run_dir, manifest):
    run_dir = Path(run_dir)
    profile = run_dir / 'oritek_perf_sim.csv'
    data = dict(tag=manifest['tag'], mode=manifest['mode'], status=manifest['status'],
                model=manifest['model'], run_dir=str(run_dir.resolve()),
                pc_process_wall_seconds=manifest.get('pc_process_wall_seconds'),
                interpretation='Simulator estimate; not measured board FPS or application end-to-end latency.')
    if profile.is_file():
        data.update(read_profile(profile))
    log_path = run_dir / 'simulator.log'
    if log_path.is_file():
        match = re.search(r'Simulator version:\s*(\S+)', log_path.read_text(errors='replace'))
        if match:
            data['simulator_version'] = match.group(1)
    lines = ['BEVDet + FlashOCC PC simulator',
             'Tag: ' + manifest['tag'], 'Status: ' + manifest['status'],
             'Model: ' + manifest['model']['path'],
             'Model SHA256: ' + manifest['model']['sha256'],
             'Mode: ' + manifest['mode'],
             'Simulator estimates below are NOT measured board performance.']
    if 'estimated_single_core_fps' in data:
        lines += [f"Estimated single-core FPS: {data['estimated_single_core_fps']:.5f}",
                  f"Estimated single-core latency: {data['estimated_single_core_latency_ms']:.4f} ms",
                  f"NN clock cycles: {data.get('nn_clock_cycles', 'not reported')}",
                  f"Profile nodes: {data.get('nodes', 'not reported')}"]
    elif manifest['mode'] == 'perf':
        lines.append('No valid FPS estimate found. Inspect simulator.log; do not use PC wall time as target latency.')
    if manifest.get('pc_process_wall_seconds') is not None:
        lines.append(f"PC process wall time (diagnostic only): {manifest['pc_process_wall_seconds']:.3f} s")
    lines += ['', 'Top nodes by estimated cycles (kernel names may be fused):']
    for row in data.get('top_nodes', []):
        lines.append(f"{row['cycles']:.0f}\t{row['hardware']}\t{row['node']}")
    lines += ['', 'Raw log: simulator.log', 'Raw profile: oritek_perf_sim.csv (perf mode)',
              'No image preprocessing, detection decode/NMS or OCC argmax is benchmarked here.']
    (run_dir / 'summary.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    (run_dir / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return data
