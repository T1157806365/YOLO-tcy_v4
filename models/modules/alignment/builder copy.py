"""Clean alignment builder for the GTGA project."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Type

from .base import BaseAlignment
from .identity import IdentityAlignment
from .gtga import GTGAAlignment


BUILTIN_ALIGNMENTS = {
    'identity': IdentityAlignment,
    'gtga': GTGAAlignment,
}


def _load_target(target: str) -> Type[BaseAlignment]:
    if ':' not in target:
        raise ValueError(
            "alignment.target must use 'package.module:ClassName', "
            f'got: {target}'
        )

    module_name, class_name = target.split(':', 1)
    module = importlib.import_module(module_name)

    if not hasattr(module, class_name):
        raise AttributeError(f'{module_name} has no class {class_name}')

    cls = getattr(module, class_name)

    if not isinstance(cls, type):
        raise TypeError(f'{target} is not a Python class')
    if not issubclass(cls, BaseAlignment):
        raise TypeError(f'{target} must inherit BaseAlignment')

    return cls


def build_alignment(cfg: Dict[str, Any] | None = None) -> BaseAlignment:
    cfg = dict(cfg or {})

    if not bool(cfg.get('enabled', False)):
        return IdentityAlignment(cfg=cfg)

    align_type = str(cfg.get('type', 'identity')).strip().lower()

    if align_type in BUILTIN_ALIGNMENTS:
        return BUILTIN_ALIGNMENTS[align_type](cfg=cfg)

    target = cfg.get('target', None)
    if not target:
        raise NotImplementedError(
            f'Unknown alignment.type={align_type!r}. '
            "Current clean built-ins are: identity, gtga."
        )

    return _load_target(str(target))(cfg=cfg)
