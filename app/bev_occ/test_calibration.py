"""Calibration CLI regression tests; run with the SDK Python environment."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import onnx
import yaml
from onnx import helper, TensorProto

HERE = Path(__file__).resolve().parent


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = self.root / 'samples/app/bev_occ'
        self.app.mkdir(parents=True)
        for name in ('prepare_calibration.py', 'prepare_mapping_calibration.py', 'model_paths.py'):
            shutil.copy2(HERE / name, self.app / name)
        (self.app / 'tmp_models').mkdir()

    def model(self, name, mapping=False):
        inputs = [helper.make_tensor_value_info('img', TensorProto.FLOAT, [1, 3, 2, 2])]
        arrays = {'img': np.ones((1, 3, 2, 2), dtype=np.float32)}
        if mapping:
            inputs.append(helper.make_tensor_value_info('ranks', TensorProto.FLOAT, [2, 1]))
            arrays.update(ranks=np.zeros((2, 1), dtype=np.float32), mapping_version=np.array(3, dtype=np.int32))
        model = helper.make_model(helper.make_graph([], name, inputs, []))
        if mapping:
            helper.set_model_props(model, {'geometry_mode': 'mapping', 'mapping_version': '3'})
            np.savez(self.app / 'tmp_models' / (name + '.inputs.npz'), **arrays)
        onnx.save(model, self.app / 'tmp_models' / (name + '.onnx'))

    def run_cli(self, script, *args):
        return subprocess.run([sys.executable, str(self.app / script), *args],
                              cwd=self.app, capture_output=True, text=True,
                              env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})

    def test_mapping_from_empty_directories_without_base_template(self):
        self.model('g3', mapping=True)
        result = self.run_cli('prepare_mapping_calibration.py', '--onnx', 'g3')
        self.assertEqual(result.returncode, 0, result.stderr)
        cfg = yaml.safe_load((self.app / 'input_cfg/g3.yml').read_text())
        self.assertEqual(cfg['quant']['activation']['quant_type'], 'S8')
        self.assertEqual(cfg['quant']['quant_tensors'], [{'tensor_name': 'ranks', 'quant_type': 'FP16'}])
        self.assertFalse((self.app / 'input_cfg/input_cfg.yml').exists())
        manifest = json.loads((self.app / 'calibration/g3/manifest.json').read_text())
        self.assertEqual(manifest['sample_count'], 1)

    def test_mapping_overwrites_outputs_but_invalid_input_preserves_them(self):
        self.model('g3', mapping=True)
        result = self.run_cli('prepare_mapping_calibration.py', '--onnx', 'g3')
        self.assertEqual(result.returncode, 0, result.stderr)
        out = self.app / 'calibration/g3'
        stale = out / 'sample_999/stale.bin'
        stale.parent.mkdir()
        stale.write_bytes(b'old')
        cfg = self.app / 'input_cfg/g3.yml'
        cfg.write_text('old config')
        result = self.run_cli('prepare_mapping_calibration.py', '--onnx', 'g3')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(stale.exists())
        self.assertEqual(yaml.safe_load(cfg.read_text())['quant']['quant_tensors'][0]['quant_type'], 'FP16')
        previous = cfg.read_bytes()
        np.savez(self.app / 'tmp_models/g3.inputs.npz', img=np.zeros((1,), dtype=np.float32))
        result = self.run_cli('prepare_mapping_calibration.py', '--onnx', 'g3')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(cfg.read_bytes(), previous)
        self.assertTrue((out / 'dataset.txt').is_file())

    def test_fixed_model_creates_image_list_without_npz_or_template(self):
        self.model('fixed')
        data = self.root / 'samples/data'
        (data / 'ImageNet').mkdir(parents=True)
        np.ones((1, 3, 2, 2), dtype=np.float32).tofile(data / 'image.bin')
        (data / 'ImageNet/dataset_bevdet.txt').write_text('../../../../data/image.bin\n')
        result = self.run_cli('prepare_calibration.py', '--onnx', 'fixed')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.app / 'input_cfg/fixed.yml').is_file())
        self.assertEqual((self.app / 'calibration/dataset_bevdet.txt').read_text(), '../../data/image.bin\n')

    def test_explicit_missing_template_reports_error_without_traceback(self):
        self.model('g3', mapping=True)
        result = self.run_cli('prepare_mapping_calibration.py', '--onnx', 'g3', '--base-config', 'missing.yml')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Cannot load --base-config', result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        self.assertFalse((self.app / 'calibration/g3').exists())


if __name__ == '__main__':
    unittest.main()
