r"""Download, inspect, and save the official CommonsenseQA dataset.

Run from the project directory:
    .\.venv\Scripts\python.exe .\scripts\prepare_data.py

Existing saved data is verified and reused without downloading it again.
The official rows, labels, order, and splits are preserved.
"""

from collections import Counter
import hashlib
from importlib.metadata import version
import json

from project_env import PROJECT_ROOT, configure_cache

CACHE_PATHS = configure_cache()

# Cache configuration must precede these imports.
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi

DATASET_ID = "tau/commonsense_qa"
DATA_DIR = PROJECT_ROOT / "data" / "commonsense_qa"
REPORT_PATH = PROJECT_ROOT / "reports" / "data-info.json"
SAMPLES_PATH = PROJECT_ROOT / "reports" / "data-samples.json"
EXPECTED_COUNTS = {"train": 9741, "validation": 1221, "test": 1140}
LABELS = {"A", "B", "C", "D", "E"}


def content_hash(split):
    digest = hashlib.sha256()
    for row in split:
        serialized = json.dumps(row, ensure_ascii=False, sort_keys=True)
        digest.update((serialized + "\n").encode("utf-8"))
    return digest.hexdigest()


def audit_split(name, split):
    errors = Counter()
    answer_counts = Counter()
    ids = []
    questions = []
    repeated_choice_text_rows = 0
    repeated_correct_answer_text_rows = 0
    for row in split:
        qid = row.get("id")
        question = row.get("question")
        if not isinstance(qid, str) or not qid.strip():
            errors["invalid_id"] += 1
        ids.append(qid)
        if not isinstance(question, str) or not question.strip():
            errors["empty_question"] += 1
        else:
            questions.append(question.strip())
        choices = row.get("choices")
        if not isinstance(choices, dict):
            errors["invalid_choices_structure"] += 1
            continue
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        if len(labels) != 5 or set(labels) != LABELS or len(texts) != 5:
            errors["invalid_five_choices"] += 1
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            errors["empty_choice"] += 1
        if len(texts) != len(set(texts)):
            repeated_choice_text_rows += 1
        answer = row.get("answerKey")
        if name == "test":
            if answer not in ("", None):
                errors["unexpected_public_test_answer"] += 1
        elif answer not in labels or answer not in LABELS:
            errors["invalid_or_missing_answer"] += 1
        answer_counts[answer or "UNLABELED"] += 1
        if answer in labels and len(texts) == len(labels):
            if texts.count(texts[labels.index(answer)]) > 1:
                repeated_correct_answer_text_rows += 1
    duplicate_ids = len(ids) - len(set(ids))
    if duplicate_ids:
        errors["duplicate_ids"] = duplicate_ids
    summary = {
        "rows": len(split),
        "columns": split.column_names,
        "structural_errors": dict(errors),
        "answer_distribution": dict(sorted(answer_counts.items())),
        "duplicate_id_rows": duplicate_ids,
        "repeated_question_text_rows_after_strip": len(questions) - len(set(questions)),
        "repeated_choice_text_rows": repeated_choice_text_rows,
        "repeated_correct_answer_text_rows": repeated_correct_answer_text_rows if name != "test" else None,
        "content_sha256": content_hash(split),
    }
    if errors:
        raise ValueError(f"{name} failed structural checks: {dict(errors)}")
    return summary, set(ids), set(questions)


def main():
    previous_report = None
    if DATA_DIR.exists():
        if not REPORT_PATH.exists():
            raise RuntimeError("Saved data exists without its provenance report; inspect it before reuse.")
        previous_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        revision = previous_report["source"]["revision"]
        dataset = load_from_disk(str(DATA_DIR))
        print("Reusing and checking saved local data.", flush=True)
    else:
        info = HfApi().dataset_info(DATASET_ID, token=False)
        revision = info.sha
        print(f"Downloading {DATASET_ID} at revision {revision}", flush=True)
        dataset = load_dataset(
            DATASET_ID,
            name="default",
            revision=revision,
            cache_dir=str(CACHE_PATHS["HF_DATASETS_CACHE"]),
            token=False,
        )

    if set(dataset) != set(EXPECTED_COUNTS):
        raise ValueError(f"Unexpected dataset splits: {list(dataset)}")
    summaries, id_sets, question_sets = {}, {}, {}
    for name, expected in EXPECTED_COUNTS.items():
        if len(dataset[name]) != expected:
            raise ValueError(f"{name}: expected {expected} rows, got {len(dataset[name])}")
        summaries[name], id_sets[name], question_sets[name] = audit_split(name, dataset[name])
        if previous_report and summaries[name]["content_sha256"] != previous_report["splits"][name]["content_sha256"]:
            raise ValueError(f"Saved {name} content differs from its recorded hash.")
        print(f"Checked {name}: {len(dataset[name])} rows.", flush=True)

    overlaps = {}
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        id_overlap = len(id_sets[left] & id_sets[right])
        if id_overlap:
            raise ValueError(f"Overlapping IDs between {left} and {right}: {id_overlap}")
        overlaps[f"{left}__{right}"] = {
            "shared_ids": id_overlap,
            "shared_question_texts_after_strip": len(question_sets[left] & question_sets[right]),
        }

    if previous_report is None:
        DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(DATA_DIR))
    reloaded = load_from_disk(str(DATA_DIR))
    for name in EXPECTED_COUNTS:
        if content_hash(reloaded[name]) != summaries[name]["content_sha256"]:
            raise ValueError(f"Round-trip content mismatch in {name}")

    report = {
        "source": {
            "dataset_id": DATASET_ID,
            "url": "https://huggingface.co/datasets/tau/commonsense_qa",
            "configuration": "default",
            "revision": revision,
            "license": "MIT",
        },
        "local_data_directory": DATA_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "cache_paths": {name: path.relative_to(PROJECT_ROOT).as_posix() for name, path in CACHE_PATHS.items()},
        "packages": {name: version(name) for name in ("datasets", "huggingface-hub", "pyarrow")},
        "splits": summaries,
        "cross_split_overlaps": overlaps,
        "saved_data_round_trip": "passed",
        "modifications": "None. Official rows, fields, order, and splits are preserved.",
        "evaluation_note": "Public test answers are absent. Report validation accuracy as development accuracy.",
    }
    samples = {}
    for name in EXPECTED_COUNTS:
        row = reloaded[name][0]
        answer = row["answerKey"]
        samples[name] = {
            "original_row": row,
            "label_index_for_training": row["choices"]["label"].index(answer) if answer else None,
        }
    samples["repeated_choice_examples"] = {}
    for name in EXPECTED_COUNTS:
        examples = []
        for row in reloaded[name]:
            texts = row["choices"]["text"]
            if len(texts) != len(set(texts)):
                examples.append(row)
                if len(examples) == 3:
                    break
        samples["repeated_choice_examples"][name] = examples
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SAMPLES_PATH.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print("Saved data and verified that local reload preserves every row.", flush=True)


if __name__ == "__main__":
    main()
