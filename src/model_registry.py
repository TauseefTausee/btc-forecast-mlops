"""Tiny file-based model registry living on the shared volume / PVC.

Layout under MODEL_DIR:

    registry.json             -> pointer to the production version + history
    last_run.json             -> metrics of the most recent validation run
    versions/<version>/model/        -> xgb_h1..7.json + metadata.json
    versions/<version>/metrics.json

Writes go to a temp file followed by os.replace, so API pods reading the
same volume never see a half-written file. A lock file serialises writers
(Airflow, Kubeflow, or several init containers starting at once).
"""

import fcntl
import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.forecaster import METADATA_FILE, BTCForecaster

REGISTRY_FILE = "registry.json"
LAST_RUN_FILE = "last_run.json"
MODEL_SUBDIR = "model"
METRICS_FILE = "metrics.json"
KEEP_VERSIONS = 10


def new_version() -> str:
    return "v" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _try_flock(fh, op) -> bool:
    """flock, but tolerate filesystems without lock support (e.g. 9p mounts)."""
    try:
        fcntl.flock(fh, op)
        return True
    except OSError:
        return False


@contextmanager
def file_lock(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        locked = _try_flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if locked:
                _try_flock(fh, fcntl.LOCK_UN)


def registry_lock(model_dir: Path):
    return file_lock(Path(model_dir) / ".registry.lock")


def save_candidate(model, metrics: dict, model_dir: Path, version: str = None) -> str:
    version = version or new_version()
    vdir = Path(model_dir) / "versions" / version
    vdir.mkdir(parents=True, exist_ok=True)
    model.save(vdir / MODEL_SUBDIR)
    _write_json(vdir / METRICS_FILE, {**metrics, "version": version})
    _write_json(Path(model_dir) / LAST_RUN_FILE, {**metrics, "version": version})
    return version


def promote(version: str, model_dir: Path) -> dict:
    model_dir = Path(model_dir)
    if not (model_dir / "versions" / version / MODEL_SUBDIR / METADATA_FILE).exists():
        raise FileNotFoundError(f"unknown model version {version}")
    with registry_lock(model_dir):
        reg = read_json(model_dir / REGISTRY_FILE, {"current": None, "history": []})
        reg["current"] = version
        reg["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        reg["history"] = ([version] + [v for v in reg.get("history", []) if v != version])[:50]
        _write_json(model_dir / REGISTRY_FILE, reg)
        _prune(model_dir, keep=set(reg["history"][:KEEP_VERSIONS]))
    return reg


def _prune(model_dir: Path, keep: set) -> None:
    vroot = model_dir / "versions"
    versions = sorted(p.name for p in vroot.iterdir() if p.is_dir())
    # keep registered versions plus the newest few candidates
    keep = keep | set(versions[-KEEP_VERSIONS:])
    for name in versions:
        if name not in keep:
            shutil.rmtree(vroot / name, ignore_errors=True)


def current_version(model_dir: Path):
    reg = read_json(Path(model_dir) / REGISTRY_FILE, {})
    return reg.get("current")


def load_version(version: str, model_dir: Path):
    vdir = Path(model_dir) / "versions" / version
    model = BTCForecaster.load(vdir / MODEL_SUBDIR)
    metrics = read_json(vdir / METRICS_FILE, {})
    return model, metrics


def load_current(model_dir: Path):
    version = current_version(model_dir)
    if not version:
        raise FileNotFoundError(f"no registered model in {model_dir}")
    model, metrics = load_version(version, model_dir)
    return version, model, metrics
