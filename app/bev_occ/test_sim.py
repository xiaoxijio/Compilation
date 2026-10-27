"""Standard-library tests; no simulator execution needed for these regressions."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_sim
from model_paths import resolve_onnx, find_npz, config_path
from sim_report import read_profile

PROFILE = '''UID,CYCLE,HARDWARE,READBW,WRITEBW,NODE
0,100,CUBE,0,0,conv
0,2e2,VEC,0,0,copy

Profiler Stats Info
1,graphs
2,nodes
300,NN clock cycles
400,NN read band width
500,NN write band width
5 fps in LQ560 single core
'''


class SimulatorTests(unittest.TestCase):
    def test_model_resolution_prefers_exact_npz_and_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / 'tmp_models'
            models.mkdir()
            model = models / 'g3.onnx'
            model.touch()
            self.assertEqual(resolve_onnx('g3', root), model)
            self.assertEqual(resolve_onnx('g3.onnx', root), model)
            self.assertEqual(config_path(model, root), root / 'input_cfg/g3.yml')
            (models / 'g30.inputs.npz').touch()
            with self.assertRaises(ValueError):
                find_npz(model)
            (models / 'g3_a.npz').touch()
            self.assertEqual(find_npz(model), [models / 'g3_a.npz'])
            (models / 'g3_b.npz').touch()
            with self.assertRaises(ValueError):
                find_npz(model)
            (models / 'g3.inputs.npz').touch()
            self.assertEqual(find_npz(model), [models / 'g3.inputs.npz'])

    def test_profile_footer_is_not_a_node_and_units_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.csv'
            path.write_text(PROFILE)
            data = read_profile(path)
            self.assertEqual(data['nn_clock_cycles'], 300)
            self.assertEqual(data['estimated_single_core_latency_ms'], 200.)
            self.assertEqual(data['parsed_layer_rows'], 2)
            self.assertEqual(data['top_nodes'][0]['node'], 'copy')
            path.write_text('UID,CYCLE,HARDWARE,READBW,WRITEBW,NODE\n')
            self.assertNotIn('estimated_single_core_fps', read_profile(path))

    def invoke(self, root, program, mode='perf'):
        runner = root / 'build/ortm_run_sim_model'
        runner.parent.mkdir(exist_ok=True)
        runner.write_text('#!/usr/bin/env python3\n' + program)
        runner.chmod(0o755)
        model = root / 'model with spaces.ortm'
        model.write_bytes(b'fixture-model')
        argv = ['run_sim.py', '--model', str(model), '--out-root', str(root / 'results'), '--tag', 'test']
        if mode == 'infer':
            data = root / 'input.bin'
            with data.open('wb') as stream:
                stream.truncate(6 * 3 * 256 * 704 * 4)
            argv += ['--mode', 'infer', '--input', str(data)]
        with patch.object(run_sim, 'HERE', root), patch.object(run_sim, 'toolchain_root', return_value=root), patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_sim.main()

    def test_runs_archive_and_external_failure_is_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            program = 'from pathlib import Path\nPath("oritek_perf_sim.csv").write_text(' + repr(PROFILE) + ')\n'
            self.assertEqual(self.invoke(root, program), 0)
            self.assertEqual(self.invoke(root, program), 0)
            summaries = list(root.glob('results/test/*/summary.json'))
            self.assertEqual(len(summaries), 2)
            self.assertEqual(self.invoke(root, 'raise SystemExit(7)\n'), 7)
            self.assertEqual(self.invoke(root, 'pass\n'), 2)
            statuses = [json.loads(p.read_text())['status'] for p in root.glob('results/test/*/run_metadata.json')]
            self.assertIn('failed', statuses)
            self.assertIn('missing_profile', statuses)

    def test_inference_requires_all_outputs_including_occ(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            program = '''import sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index('-o') + 1])
out.mkdir()
print('outputs:')
for index in range(7):
    print(f'    {index}: dtype: F32 shape: 1x2 stride: 8 size: 8')
print('[I] execute done')
for index in range(7):
    path = out / f'output_{index}.bin'
    path.write_bytes(b'12345678')
    print('dump output path:', path)
'''
            self.assertEqual(self.invoke(root, program, mode='infer'), 0)
            missing_occ = program.replace('for index in range(7):\n    path', 'for index in range(6):\n    path')
            self.assertEqual(self.invoke(root, missing_occ, mode='infer'), 2)


if __name__ == '__main__':
    unittest.main()
