from __future__ import annotations

from runtime import DEFAULT_MODEL_DIR, load_runtime_bundle

_BUNDLE = None


def _load():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = load_runtime_bundle(DEFAULT_MODEL_DIR)
    return _BUNDLE


def pcyber_score(row: dict) -> float:
    return float(_load().score_row(row).get("pcyber", 0.0))
