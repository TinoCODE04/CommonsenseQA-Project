"""Configure project-local caches before importing Hugging Face libraries."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_cache():
    hf_home = PROJECT_ROOT / ".cache" / "huggingface"
    paths = {
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_home / "hub",
        "HF_DATASETS_CACHE": hf_home / "datasets",
        "HF_XET_CACHE": hf_home / "xet",
    }
    os.environ.update({name: str(path) for name, path in paths.items()})
    return paths
