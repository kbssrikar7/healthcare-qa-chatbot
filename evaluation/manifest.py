"""
Reproducibility manifest for evaluation and benchmark runs (plan B4 / F1).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _git_sha(project_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    try:
        h.update(path.read_bytes())
        return h.hexdigest()[:16]
    except Exception:
        return None


def build_manifest(
    project_root: Path,
    *,
    extra: Optional[Dict[str, Any]] = None,
    kb_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    pr = Path(project_root).resolve()
    kb = Path(kb_dir) if kb_dir else pr / "data" / "knowledge_base_v2"
    meta_path = kb / "embedding_metadata.json"
    m: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": _git_sha(pr),
        "kb_dir": str(kb),
        "embedding_metadata_sha256_prefix": _file_sha256(meta_path),
        "env_snapshot": {
            "ENVIRONMENT": os.getenv("ENVIRONMENT", ""),
            "CHROMA_COLLECTION": os.getenv("CHROMA_COLLECTION", ""),
            "KB_PERSIST_DIR": os.getenv("KB_PERSIST_DIR", ""),
        },
    }
    if extra:
        m["extra"] = extra
    return m


def write_manifest(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
