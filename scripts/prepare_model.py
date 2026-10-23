r"""Cache RoBERTa-Base and check a real CommonsenseQA forward/backward pass.

Run from the project directory:
    .\.venv\Scripts\python.exe .\scripts\prepare_model.py

Subsequent runs reuse the recorded snapshot and verify it locally.
This script never performs an optimizer step or saves fine-tuned weights.
"""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from project_env import PROJECT_ROOT, configure_cache

CACHE_PATHS = configure_cache()

import torch
from datasets import load_from_disk
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForMultipleChoice, AutoTokenizer, set_seed

MODEL_ID = "FacebookAI/roberta-base"
REPORT_PATH = PROJECT_ROOT / "reports" / "model-info.json"
ALLOWED_FILES = {
    "config.json", "tokenizer_config.json", "tokenizer.json",
    "special_tokens_map.json", "vocab.json", "merges.txt", "model.safetensors",
}
REQUIRED_FILES = ALLOWED_FILES - {"special_tokens_map.json"}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def get_snapshot():
    if REPORT_PATH.exists():
        previous = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        if previous["source"]["model_id"] != MODEL_ID:
            raise ValueError("The saved model report refers to another model.")
        snapshot = (PROJECT_ROOT / previous["local_snapshot"]).resolve()
        if not snapshot.is_relative_to(CACHE_PATHS["HF_HUB_CACHE"].resolve()):
            raise ValueError("Recorded model snapshot is outside this project's cache.")
        print("Reusing and checking the saved local model snapshot.", flush=True)
        return snapshot, previous["source"]["revision"], previous["files"]

    info = HfApi().model_info(MODEL_ID, token=False, files_metadata=True)
    expected_files = {}
    for entry in info.siblings:
        if entry.rfilename in ALLOWED_FILES:
            expected_files[entry.rfilename] = {
                "size_bytes": entry.size,
                "source_lfs_sha256": getattr(entry.lfs, "sha256", None),
            }
    if not REQUIRED_FILES.issubset(expected_files):
        raise ValueError("The model repository is missing required weight/tokenizer files.")
    print(f"Downloading {MODEL_ID} at revision {info.sha}", flush=True)
    snapshot = Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=info.sha,
        cache_dir=str(CACHE_PATHS["HF_HUB_CACHE"]),
        allow_patterns=sorted(expected_files),
        token=False,
        max_workers=2,
    )).resolve()
    return snapshot, info.sha, expected_files


def main():
    dataset_path = PROJECT_ROOT / "data" / "commonsense_qa"
    if not dataset_path.exists():
        raise RuntimeError("Prepare the local CommonsenseQA data first.")
    snapshot, revision, expected_files = get_snapshot()
    checked_files = {}
    for name, expected in sorted(expected_files.items()):
        path = snapshot / name
        actual_size = path.stat().st_size
        actual_hash = file_hash(path)
        if expected.get("size_bytes") is not None and actual_size != expected["size_bytes"]:
            raise ValueError(f"File size mismatch: {name}")
        expected_hash = expected.get("sha256") or expected.get("source_lfs_sha256")
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"SHA256 mismatch: {name}")
        checked_files[name] = {**expected, "size_bytes": actual_size, "sha256": actual_hash}
    print("Downloaded file sizes and available source checksums verified.", flush=True)

    torch.set_num_threads(4)
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
    )
    model, loading_info = AutoModelForMultipleChoice.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, output_loading_info=True,
    )
    # Transformers 5 returns sets for some loading-report fields.
    loading_info = {
        name: sorted(value) if isinstance(value, (set, frozenset)) else value
        for name, value in loading_info.items()
    }
    missing = loading_info.get("missing_keys", [])
    missing_encoder = [name for name in missing if not name.startswith(("classifier.", "roberta.pooler."))]
    if missing_encoder or loading_info.get("mismatched_keys") or loading_info.get("error_msgs"):
        raise ValueError(f"Pretrained encoder loading failed: {loading_info}")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("Expected a CPU model.")

    dataset = load_from_disk(str(dataset_path))
    row = dataset["train"][0]
    labels = row["choices"]["label"]
    answers = row["choices"]["text"]
    label_index = labels.index(row["answerKey"])
    encoded = tokenizer(
        [row["question"]] * len(answers), answers,
        padding=True, truncation=True, max_length=64,
        return_token_type_ids=False, return_tensors="pt",
    )
    batch = {name: tensor.unsqueeze(0) for name, tensor in encoded.items()}
    model.train()
    output = model(**batch, labels=torch.tensor([label_index], dtype=torch.long))
    if tuple(output.logits.shape) != (1, 5) or not torch.isfinite(output.loss).item():
        raise ValueError("Invalid multiple-choice model output.")
    output.loss.backward()
    gradients = [(name, parameter.grad) for name, parameter in model.named_parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all().item() for _, gradient in gradients):
        raise ValueError("Missing or non-finite gradients.")
    encoder_name, encoder_gradient = next((name, gradient) for name, gradient in gradients if name.startswith("roberta.encoder."))
    if encoder_gradient.norm().item() == 0 or model.classifier.weight.grad.norm().item() == 0:
        raise ValueError("The encoder or classification head did not receive a gradient.")

    report = {
        "source": {
            "model_id": MODEL_ID,
            "url": "https://huggingface.co/FacebookAI/roberta-base",
            "revision": revision,
            "license": "MIT",
        },
        "local_snapshot": snapshot.relative_to(PROJECT_ROOT).as_posix(),
        "files": checked_files,
        "packages": {name: version(name) for name in ("torch", "transformers", "tokenizers", "safetensors")},
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "device": "cpu",
        "dtype": str(next(model.parameters()).dtype),
        "loading_info": loading_info,
        "checks": {
            "file_integrity": "passed",
            "pretrained_encoder_loading": "passed",
            "real_training_example_id": row["id"],
            "correct_answer_label": row["answerKey"],
            "correct_answer_index": label_index,
            "input_shape": list(batch["input_ids"].shape),
            "logits_shape": list(output.logits.shape),
            "loss": float(output.loss.detach()),
            "finite_gradients": "passed",
            "gradient_parameter_count": len(gradients),
            "encoder_gradient_example": encoder_name,
            "encoder_gradient_norm": float(encoder_gradient.norm()),
            "classifier_gradient_norm": float(model.classifier.weight.grad.norm()),
            "optimizer_steps": 0,
        },
        "initialization_note": "The multiple-choice classifier and any missing pooler weights are randomly initialized with seed 42.",
        "training_status": "Forward/backward validation only. No parameter updates or fine-tuned checkpoint.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    env_path = PROJECT_ROOT / "environment-info.json"
    environment = json.loads(env_path.read_text(encoding="utf-8"))
    environment["pretrained_weights_downloaded"] = True
    environment["model_report"] = "reports/model-info.json"
    environment["model_snapshot"] = report["local_snapshot"]
    env_path.write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print("RoBERTa-Base passed the real-example forward/backward check; no optimizer step was run.", flush=True)


if __name__ == "__main__":
    main()
