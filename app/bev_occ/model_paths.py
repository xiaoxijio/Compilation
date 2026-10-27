"""Resolve model names and associated local artifacts without guessing ambiguous files."""
from pathlib import Path


def resolve_onnx(value, here):
    path = Path(value)
    if path.parent == Path('.'):
        path = here / 'tmp_models' / (path.name if path.suffix == '.onnx' else path.name + '.onnx')
    return path.resolve(strict=True)


def find_npz(model):
    exact = model.with_suffix('.inputs.npz')
    if exact.is_file():
        return [exact]
    exact = model.with_suffix('.npz')
    if exact.is_file():
        return [exact]
    matches = sorted(p for p in model.parent.glob(model.stem + '*.npz')
                     if p.name[len(model.stem):len(model.stem)+1] in ('_', '-', '.'))
    if len(matches) != 1:
        raise ValueError(f'{model.stem}: expected one matching NPZ, found {len(matches)}; use --npz explicitly')
    return matches


def config_path(model, here):
    return here / 'input_cfg' / (model.stem + '.yml')
