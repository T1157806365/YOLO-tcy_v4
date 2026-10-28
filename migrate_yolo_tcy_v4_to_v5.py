#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a clean YOLO-tcy_v5 from a local v4 BEST checkpoint.

Run on the user's own server, using the existing 'tcy' environment.
No source files are changed or deleted. V5 is built in a temporary folder and
published only after copying and checking the checkpoint and Python syntax.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import py_compile
import shutil
import sys
import time
from collections import deque
from pathlib import Path

SRC_DEFAULT = Path('/mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v4')
DST_DEFAULT = Path('/mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5')
RUN_REL = Path('runs/rlsfa_gtga_bhlr_gmls_v2/yolo26m_UAVCB_GMLSv2_DMLS-Correction_Fine_Scale_BHLRv2-P3_ADD_rgb1920_sem640_tir640_seed0_scratch_bs32_r6_bhlrnew_newbest')
ENTRYPOINTS = (
    'train_rlsfa_gtga_bhlr_v2.py',
    'val_rlsfa_gtga_bhlr_v2.py',
    'visualize_rlsfa_coarsefine_gtga_bhlr_add.py',
    'eval_alignment_iou_displacement_GMLS_v2.py',
    'models/gtga_bhlr_factory.py',
    'models/rgbt_rlsfa_gtga_centered_bhlr_model.py',
)
# Factory retains legacy branches so checkpoint loading logic stays unchanged.
# We don't copy legacy model variants that are imported ONLY inside those branches.
FACTORY_KEEP_IMPORTS = {'models.rgbt_rlsfa_gtga_centered_bhlr_model'}


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def size_tree(path: Path) -> int:
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [x for x in dirs if not (Path(root) / x).is_symlink()]
        for file in files:
            p = Path(root) / file
            if not p.is_symlink():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    return total


def norm_rel(src: Path, p: Path) -> Path:
    try:
        rel = p.relative_to(src)
    except ValueError:
        raise RuntimeError('Refusing to copy a path outside source: ' + str(p))
    if '..' in rel.parts:
        raise RuntimeError('Unsafe relative path: ' + str(rel))
    return rel


def resolve_module(src: Path, name: str) -> list[Path]:
    if not name or not name.startswith(('models', 'engine', 'datasets', 'utils', 'configs')):
        return []
    parts = name.split('.')
    out = []
    for i in range(1, len(parts)):
        ip = src.joinpath(*parts[:i], '__init__.py')
        if ip.is_file():
            out.append(ip)
    py = src.joinpath(*parts).with_suffix('.py')
    pkg = src.joinpath(*parts, '__init__.py')
    if py.is_file():
        out.append(py)
    elif pkg.is_file():
        out.append(pkg)
    return out


def imported_modules(src: Path, filename: Path) -> list[str]:
    try:
        tree = ast.parse(filename.read_text(encoding='utf-8-sig'))
    except (UnicodeDecodeError, SyntaxError) as e:
        raise RuntimeError(f'Source Python parsing failed: {filename}: {e}')
    rel = filename.relative_to(src)
    package = list(rel.parts[:-1])
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level > len(package):
                    continue
                prefix = package[:len(package) - node.level + 1]
                if node.module:
                    prefix += node.module.split('.')
                base = '.'.join(prefix)
            else:
                base = node.module or ''
            if base:
                modules.append(base)
            for name in node.names:
                if name.name != '*':
                    modules.append((base + '.' if base else '') + name.name)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == 'import_module' and node.args
              and isinstance(node.args[0], ast.Constant)
              and isinstance(node.args[0].value, str)):
            modules.append(node.args[0].value)
    return modules


def dependency_closure(src: Path) -> tuple[list[Path], list[str]]:
    queue = deque()
    missed = []
    for seed in ENTRYPOINTS:
        p = src / seed
        if p.is_file():
            queue.append(p)
        else:
            missed.append(seed)
    required = {'models/rgbt_rlsfa_gtga_centered_bhlr_model.py',
                'models/gtga_bhlr_factory.py',
                'train_rlsfa_gtga_bhlr_v2.py', 'val_rlsfa_gtga_bhlr_v2.py'}
    if any(not (src / s).is_file() for s in required):
        raise FileNotFoundError('Missing an essential source entrypoint; review v4 layout.')
    seen = set()
    while queue:
        p = queue.popleft()
        rel = norm_rel(src, p)
        if rel in seen:
            continue
        seen.add(rel)
        for mod in imported_modules(src, p):
            if rel.as_posix() == 'models/gtga_bhlr_factory.py' and mod.startswith('models.rgbt_'):
                if mod not in FACTORY_KEEP_IMPORTS:
                    continue
            for dep in resolve_module(src, mod):
                if dep.relative_to(src) not in seen:
                    queue.append(dep)
    return sorted(seen), missed


def write_yaml(path: Path, value: dict) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False,
                                    default_flow_style=False), encoding='utf-8')


def update_paths(obj, src: Path, dst: Path):
    if isinstance(obj, dict):
        return {k: update_paths(v, src, dst) for k, v in obj.items()}
    if isinstance(obj, list):
        return [update_paths(x, src, dst) for x in obj]
    if isinstance(obj, str):
        prefix = str(src)
        return str(dst) + obj[len(prefix):] if (obj == prefix or obj.startswith(prefix + '/')) else obj
    return obj


def copy_one(src: Path, stage: Path, rel: Path) -> None:
    from_file = src / rel
    to_file = stage / rel
    to_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(from_file, to_file)


def extra_weights(src: Path, model_name: str, force_names: list[str]) -> list[Path]:
    result = set()
    for raw in force_names:
        p = Path(raw)
        if not p.is_absolute():
            p = src / p
        if not p.is_file() or not p.is_relative_to(src):
            raise FileNotFoundError('Extra weight not found within v4: ' + str(p))
        result.add(p.relative_to(src))
    excluded = {'runs', '.git', 'logs', 'pid', '__pycache__', 'datasets'}
    wanted = model_name.lower().replace('-', '').replace('_', '')
    for root, dirs, filenames in os.walk(src):
        dirs[:] = [d for d in dirs if d not in excluded and not d.startswith('.YOLO-')]
        for name in filenames:
            if not name.lower().endswith(('.pt', '.pth', '.safetensors', '.ckpt')):
                continue
            normalized = name.lower().replace('-', '').replace('_', '')
            if wanted in normalized:
                result.add((Path(root) / name).relative_to(src))
    return sorted(result)


def copy_reference_configs(src: Path, stage: Path, run_cfg: dict, original_cfg_path: Path):
    import yaml
    # Include dataset YAMLs referenced by the saved checkpoint; these can be absolute v4 paths.
    dataset_files = []
    cfg_data = run_cfg.get('data', {}) or {}
    for key, basename in [('rgb', 'UAV-CB-RGB.yaml'), ('tir', 'UAV-CB-TIR.yaml')]:
        raw = cfg_data.get(key, '')
        candidate = Path(str(raw)).expanduser() if raw else src / 'configs/datasets' / basename
        if not candidate.is_absolute():
            candidate = src / candidate
        # Always prefer the source project's named dataset config.
        backup = src / 'configs/datasets' / basename
        chosen = candidate if candidate.is_file() and candidate.is_relative_to(src) else backup
        if not chosen.is_file():
            raise FileNotFoundError(f'Cannot locate original {key} dataset YAML: {candidate} / {backup}')
        dest = stage / 'configs/datasets' / basename
        dest.parent.mkdir(parents=True, exist_ok=True)
        parsed = yaml.safe_load(chosen.read_text(encoding='utf-8'))
        parsed = update_paths(parsed, src, stage.parent / 'YOLO-tcy_v5')
        write_yaml(dest, parsed)
        dataset_files.append({'modality': key, 'source': str(chosen), 'dest': str(dest.relative_to(stage))})
    if original_cfg_path.is_file():
        archival_yaml = stage / 'docs/source_v4_training.yaml'
        archival_yaml.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_cfg_path, archival_yaml)
    # The saved configuration, not the mutable source YAML, is authoritative.
    canonical = update_paths(copy.deepcopy(run_cfg), src, stage.parent / 'YOLO-tcy_v5')
    canonical.setdefault('data', {})['rgb'] = str(stage.parent / 'YOLO-tcy_v5/configs/datasets/UAV-CB-RGB.yaml')
    canonical.setdefault('data', {})['tir'] = str(stage.parent / 'YOLO-tcy_v5/configs/datasets/UAV-CB-TIR.yaml')
    canonical.setdefault('output', {})['project'] = 'runs/v5_experiments'
    canonical['output']['name'] = 'gmls_v2_bhlrv2_retrain'
    canonical.setdefault('train', {})['resume'] = None
    # A common typo in old human-edited YAML is 'ture' (string) instead of true (boolean).
    if str(canonical.get('model', {}).get('pretrained', '')).lower() == 'ture':
        canonical['model']['pretrained'] = True
    write_yaml(stage / 'configs/experiments/best_baseline.yaml', canonical)
    return dataset_files


def create_helpers(stage: Path, run_rel: Path, weights: list[Path]):
    best = run_rel / 'weights/best.pt'
    for folder in ['logs', 'pid', 'docs', 'scripts', 'runs/v5_experiments']:
        (stage / folder).mkdir(parents=True, exist_ok=True)
    lines = [
        '#!/usr/bin/env bash', 'set -euo pipefail',
        'cd "$(dirname "$(realpath "$0")")/.."',
        'python val_rlsfa_gtga_bhlr_v2.py \\',
        f'  --weights "{best.as_posix()}" \\',
        '  --rgb configs/datasets/UAV-CB-RGB.yaml \\',
        '  --tir configs/datasets/UAV-CB-TIR.yaml \\',
        '  --split val --device "${1:-3}" --batch 4 --workers 4 \\',
        '  --conf 0.001 --iou 0.70 --max-det 300 \\',
        '  --save-dir runs/v5_validation',
    ]
    sh = stage / 'scripts/validate_best.sh'
    sh.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    sh.chmod(0o755)
    train = stage / 'scripts/train_baseline.sh'
    train.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$(realpath "$0")")/.."\n'
                     'python train_rlsfa_gtga_bhlr_v2.py --cfg configs/experiments/best_baseline.yaml\n',
                     encoding='utf-8')
    train.chmod(0o755)
    verify = stage / 'verify_v5.py'
    verify.write_text('''#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import sys
import torch
from models.gtga_bhlr_factory import load_model_from_checkpoint
p=Path(__file__).resolve().parent
m=json.loads((p/'docs/migration_manifest.json').read_text())
w=p/m['best_rel']
assert w.is_file(), w
h=hashlib.sha256()
with w.open('rb') as f:
    for block in iter(lambda:f.read(4*1024*1024), b''):
        h.update(block)
assert h.hexdigest()==m['best_sha256']
model, checkpoint, config = load_model_from_checkpoint(w, device=torch.device('cpu'), strict=True)
assert Path(sys.modules['models.gtga_bhlr_factory'].__file__).resolve().is_relative_to(p)
print('[PASS] Checkpoint SHA256 and strict CPU model load passed')
print('Model:', getattr(model,'model_name', type(model).__name__))
print('Data paths embedded in original checkpoint (may refer to v4):', config.get('data'))
print('Use scripts/validate_best.sh to pass v5 dataset YAMLs explicitly.')
''', encoding='utf-8')
    return {'validation_script': 'scripts/validate_best.sh',
            'training_script': 'scripts/train_baseline.sh',
            'best_rel': best.as_posix()}


def main():
    parser = argparse.ArgumentParser(description='Safely migrate one v4 best model to a clean v5')
    parser.add_argument('--src', type=Path, default=SRC_DEFAULT)
    parser.add_argument('--dst', type=Path, default=DST_DEFAULT)
    parser.add_argument('--dry-run', action='store_true', help='Show plan only; do not create v5')
    parser.add_argument('--extra-weight', action='append', default=[],
                        help='Extra v4-relative checkpoint to copy (repeatable)')
    a = parser.parse_args()
    src = a.src.expanduser().resolve()
    dst = a.dst.expanduser().absolute()
    if src == dst or src in dst.parents or dst in src.parents:
        raise RuntimeError('Source and destination must be distinct sibling paths')
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f'REFUSING TO OVERWRITE existing v5: {dst}')
    if src.name != 'YOLO-tcy_v4' or dst.name != 'YOLO-tcy_v5':
        raise ValueError('Expected YOLO-tcy_v4 -> YOLO-tcy_v5 (use exact folder names)')
    run_src = src / RUN_REL
    best = run_src / 'weights/best.pt'
    if not best.is_file():
        raise FileNotFoundError(f'Trained BEST checkpoint missing: {best}')
    print('[SOURCE]', src)
    print('[DESTINATION]', dst)
    print('[RUN]', RUN_REL)
    print('[CHECKPOINT]', best)
    import torch
    # Original project checkpoint is user-owned/trusted. weights_only=False is needed
    # for existing configs and optimizer state, not for untrusted third-party .pt files.
    ckpt = torch.load(best, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt:
        raise ValueError('Unexpected checkpoint structure: expected model_state_dict')
    cfg = ckpt.get('config')
    if not isinstance(cfg, dict):
        raise ValueError('BEST checkpoint does not contain its original config dict')
    variant = str(cfg.get('model', {}).get('variant', '')).lower()
    modelname = str(cfg.get('model', {}).get('name', 'yolo26m'))
    if variant != 'rlsfa_gtga_centered_bhlr_v2' or modelname.lower() != 'yolo26m':
        raise RuntimeError(f'Unexpected variant/model: {variant}/{modelname}; refusing unverified migration')
    del ckpt
    files, missed_optional = dependency_closure(src)
    weight_paths = extra_weights(src, modelname, a.extra_weight)
    other_config = src / 'configs/models/yolo26/yolo26m.yaml'
    orig_cfg = src / 'configs/experiments/gmls_v2/uavcb_gmls_v2_yolo26m_1920_640.yaml'
    if other_config.is_file():
        files.append(other_config.relative_to(src))
    for name in ('requirements.txt', 'pyproject.toml', 'environment.yml'):
        if (src / name).is_file():
            files.append(Path(name))
    for p in weight_paths:
        files.append(p)
    files = sorted(set(files))
    run_size = size_tree(run_src)
    estimated = run_size + sum((src / f).stat().st_size for f in files)
    print(f'[PLAN] Core Python/weight/config files: {len(files)}')
    print(f'[PLAN] Pretrained/extra weights: {[p.as_posix() for p in weight_paths] or "NONE FOUND"}')
    print(f'[PLAN] Run folder size approximately {run_size/(1024**3):.3f} GiB')
    if missed_optional:
        print('[NOTE] Optional entrypoints absent:', ', '.join(missed_optional))
    if not weight_paths:
        print('[WARNING] No matching YOLO26m pretrained weight found outside runs.')
        print('          Existing best.pt remains fully preserved. Add --extra-weight if needed.')
    print('[DEPENDENCIES]')
    for f in files:
        print('  ', f.as_posix())
    if a.dry_run:
        print('[DRY RUN] No files copied.')
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dst.parent).free
    if estimated > free * 0.95:
        raise RuntimeError(f'Insufficient free disk: need approx {estimated/(1024**3):.2f} GiB, free {free/(1024**3):.2f} GiB')
    stage = dst.parent / ('.YOLO-tcy_v5_staging_' + str(os.getpid()))
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir()
    try:
        for index, rel in enumerate(files, 1):
            copy_one(src, stage, rel)
        dataset_files = copy_reference_configs(src, stage, cfg, orig_cfg)
        print('[COPY] Complete experiment directory (including weights, logs, metrics)...')
        run_dst = stage / RUN_REL
        run_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(run_src, run_dst, symlinks=True)
        orig_hash = digest(best)
        copied_hash = digest(run_dst / 'weights/best.pt')
        if orig_hash != copied_hash:
            raise RuntimeError('BEST checkpoint checksum mismatch! V5 not published.')
        py_files = list(stage.rglob('*.py'))
        for py in py_files:
            try:
                py_compile.compile(str(py), doraise=True)
            except py_compile.PyCompileError as e:
                raise RuntimeError(f'Python syntax error in {py.relative_to(stage)}: {e}')
        # Remove transient pycache produced by compilation.
        for cache in list(stage.rglob('__pycache__')):
            shutil.rmtree(cache, ignore_errors=True)
        helpers = create_helpers(stage, RUN_REL, weight_paths)
        old_path_mentions = []
        for rel in files:
            candidate = stage / rel
            if candidate.suffix.lower() in {'.py', '.yaml', '.yml', '.json'}:
                try:
                    if str(src) in candidate.read_text(encoding='utf-8'):
                        old_path_mentions.append(rel.as_posix())
                except UnicodeDecodeError:
                    pass
        if old_path_mentions:
            print('[WARNING] These copied source files mention absolute v4 paths; review if needed:')
            for mention in old_path_mentions:
                print('   ', mention)
        env_lines = [f'python={sys.version.split()[0]}', f'torch={torch.__version__}']
        try:
            import importlib.metadata as md
            for pkg in ('ultralytics', 'numpy', 'opencv-python', 'pyyaml'):
                try:
                    env_lines.append(f'{pkg}={md.version(pkg)}')
                except md.PackageNotFoundError:
                    pass
        except ImportError:
            pass
        (stage / 'docs/source_environment.txt').write_text('\n'.join(env_lines) + '\n', encoding='utf-8')
        manifest = {
            'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'source_root': str(src), 'dest_root': str(dst),
            'checkpoint_original': str(best),
            'best_rel': helpers['best_rel'], 'best_sha256': orig_hash,
            'source_config_variant': variant,
            'source_config_model': modelname,
            'source_config_is_from_checkpoint': True,
            'total_core_files': len(files),
            'core_files': [p.as_posix() for p in files],
            'pretrained_weight_files': [p.as_posix() for p in weight_paths],
            'missing_optional_entrypoints': missed_optional,
            'source_files_with_old_absolute_paths': old_path_mentions,
            'datasets': dataset_files,
            'run_size_bytes': run_size,
            'script_paths': helpers,
            'checkpoint_metadata_note': 'Original best.pt config is intentionally UNMODIFIED and may contain v4 dataset paths; pass v5 YAML explicitly or use scripts/validate_best.sh',
        }
        (stage / 'docs/migration_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        readme = f'''# YOLO-tcy_v5 — Clean GMLS-v2 baseline

Only the best GMLS-v2+DMLS correction+Fine Scale+BHLR-v2(P3)+ADD run from v4 is copied.

## What is preserved?
- Source Python files needed by the chosen model and training/evaluation commands.
- Baseline experiment config reconstructed from the TRAINED checkpoint: `configs/experiments/best_baseline.yaml`.
- Original v4 YAML archived at `docs/source_v4_training.yaml` for provenance (it may differ from trained config).
- RGB/TIR dataset YAML copied into `configs/datasets` (original image data stays external).
- Matching YOLO26m pretrained weights found in v4 (see migration manifest).
- FULL experiment folder at `{RUN_REL.as_posix()}` (best/last weights, logs, and recorded results).

## Verify without training
```bash
conda activate tcy
cd {dst}
python verify_v5.py
bash scripts/validate_best.sh 3
```

## Retrain only when intended
```bash
conda activate tcy
cd {dst}
# Edit configs/experiments/best_baseline.yaml (train.device especially) before training
bash scripts/train_baseline.sh
```

The saved checkpoint metadata is unmodified. Its embedded dataset paths may still point to v4.
The validation helper overrides RGB/TIR YAML so it uses the v5 copies.
Do not delete v4 until strict load and validation metrics match.

## Provenance
See `docs/migration_manifest.json` for SHA256 and copied file paths.
'''
        (stage / 'README.md').write_text(readme, encoding='utf-8')
        # Rename atomically only when the target does not exist.
        if dst.exists():
            raise FileExistsError(dst)
        stage.rename(dst)
        print('\n[SUCCESS] Created:', dst)
        print('[VERIFIED] best.pt SHA256:', orig_hash)
        print('[VERIFIED] Syntax checked:', len(py_files), 'Python files')
        print('[RUN] cd', dst)
        print('[RUN] python verify_v5.py')
        print('[RUN] bash scripts/validate_best.sh 3')
        print('[INFO] v4 left untouched. Dataset image folders not duplicated.')
    except Exception:
        print('[FAILED] Original v4 is untouched. Incomplete staging folder:', stage, file=sys.stderr)
        print('[FAILED] Inspect/remove staging folder ONLY after verifying it is not needed.', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
