r"""Run an offline, reproducible CommonsenseQA training experiment.

    .\.venv\Scripts\python.exe .\scripts\train.py --config configs/trial.json
"""

import argparse
from collections import Counter
import gc
import json
import math
import os
import platform
from time import perf_counter

from project_env import PROJECT_ROOT, configure_cache

configure_cache()
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import psutil
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForMultipleChoice, AutoTokenizer, DataCollatorForMultipleChoice,
    TrainingArguments, set_seed,
)

from mcqa import compute_accuracy, encode_example
from mcqa_trainer import MultipleChoiceTrainer
from prepare_data import content_hash
from prepare_model import file_hash
from training_monitor import TrainingMonitor


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def select_rows(split, count, seed):
    if count == 0:
        return split
    if count < 0 or count > len(split):
        raise ValueError(f"Invalid subset size: {count}")
    return split.shuffle(seed=seed, keep_in_memory=True).select(range(count))


def run(config):
    output_dir = (PROJECT_ROOT / config["output_directory"]).resolve()
    if not output_dir.is_relative_to(PROJECT_ROOT / "outputs") or output_dir == PROJECT_ROOT / "outputs":
        raise ValueError("Use a separate experiment directory inside outputs/.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Experiment directory is not empty: {output_dir}")
    for field in ("batch_size", "eval_batch_size", "gradient_accumulation_steps", "epochs", "threads"):
        if config[field] <= 0:
            raise ValueError(f"{field} must be positive.")

    source_model = json.loads((PROJECT_ROOT / "reports/model-info.json").read_text(encoding="utf-8"))
    source_data = json.loads((PROJECT_ROOT / "reports/data-info.json").read_text(encoding="utf-8"))
    snapshot = PROJECT_ROOT / source_model["local_snapshot"]
    if file_hash(snapshot / "model.safetensors") != source_model["files"]["model.safetensors"]["sha256"]:
        raise ValueError("The starting pretrained weights differ from the recorded snapshot.")
    dataset = load_from_disk(str(PROJECT_ROOT / source_data["local_data_directory"]))
    for name in ("train", "validation"):
        if content_hash(dataset[name]) != source_data["splits"][name]["content_sha256"]:
            raise ValueError(f"Local {name} data differs from its recorded source.")
    train_rows = select_rows(dataset["train"], config["train_size"], config["seed"])
    validation_rows = select_rows(dataset["validation"], config["validation_size"], config["validation_seed"])
    train_ids, validation_ids = list(train_rows["id"]), list(validation_rows["id"])
    if set(train_ids) & set(validation_ids):
        raise ValueError("Training and validation IDs overlap.")

    torch.set_num_threads(config["threads"])
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    train_data = train_rows.map(
        lambda row: encode_example(row, tokenizer, config["max_length"]),
        remove_columns=train_rows.column_names, keep_in_memory=True, load_from_cache_file=False,
        desc="Encoding training subset",
    )
    validation_data = validation_rows.map(
        lambda row: encode_example(row, tokenizer, config["max_length"]),
        remove_columns=validation_rows.column_names, keep_in_memory=True, load_from_cache_file=False,
        desc="Encoding validation subset",
    )
    effective_batch = config["batch_size"] * config["gradient_accumulation_steps"]
    expected_steps = math.ceil(math.ceil(len(train_data) / config["batch_size"]) / config["gradient_accumulation_steps"]) * config["epochs"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "run-config.json", config)
    save_json(output_dir / "subset-ids.json", {"train": train_ids, "validation": validation_ids})
    monitor = TrainingMonitor()
    monitor.start()
    try:
        model = AutoModelForMultipleChoice.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        )
        encoder_before = model.roberta.encoder.layer[0].attention.self.query.weight.detach().clone()
        classifier_before = model.classifier.weight.detach().clone()
        collator = DataCollatorForMultipleChoice(tokenizer)
        args = TrainingArguments(
            output_dir=str(output_dir), use_cpu=True, fp16=False, bf16=False,
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["eval_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            num_train_epochs=config["epochs"], learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            warmup_steps=math.ceil(expected_steps * config["warmup_fraction"]),
            lr_scheduler_type="linear", optim="adamw_torch", max_grad_norm=1.0,
            eval_strategy="no", save_strategy="epoch", save_total_limit=1,
            save_only_model=False, logging_strategy="steps", logging_steps=1,
            seed=config["seed"], data_seed=config["seed"],
            dataloader_num_workers=0, dataloader_pin_memory=False,
            report_to="none", push_to_hub=False, disable_tqdm=True,
            skip_memory_metrics=True,
        )
        trainer = MultipleChoiceTrainer(
            model=model, args=args, train_dataset=train_data, eval_dataset=validation_data,
            processing_class=tokenizer, data_collator=collator,
            compute_metrics=compute_accuracy, callbacks=[monitor],
        )
        print("Measuring the untrained task head on the fixed validation subset.", flush=True)
        baseline = trainer.predict(validation_data, metric_key_prefix="dev_before")
        print(f"Starting {expected_steps} optimizer steps; effective batch size {effective_batch}.", flush=True)
        started = perf_counter()
        training_result = trainer.train()
        training_wall_seconds = perf_counter() - started
        if trainer.state.global_step != expected_steps:
            raise RuntimeError(f"Expected {expected_steps} steps, got {trainer.state.global_step}")
        encoder_delta = float((model.roberta.encoder.layer[0].attention.self.query.weight.detach() - encoder_before).abs().max())
        classifier_delta = float((model.classifier.weight.detach() - classifier_before).abs().max())
        if not encoder_delta > 0 or not classifier_delta > 0:
            raise RuntimeError("The encoder and classifier must both update during full fine-tuning.")
        after = trainer.predict(validation_data, metric_key_prefix="dev_after")
        if not np.isfinite(after.predictions).all():
            raise RuntimeError("Non-finite validation predictions.")
        checkpoint = output_dir / f"checkpoint-{trainer.state.global_step}"
        required = ["model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json", "tokenizer_config.json"]
        if not all((checkpoint / name).is_file() for name in required):
            raise RuntimeError("The training checkpoint is incomplete.")
        train_metrics = dict(training_result.metrics)
        log_history = list(trainer.state.log_history)
        if not all(math.isfinite(entry["loss"]) for entry in log_history if "loss" in entry):
            raise RuntimeError("Non-finite training loss.")
        probe_features = [validation_data[i] for i in range(2)]
        original_probe = collator(probe_features)
        expected_logits = np.asarray(after.predictions[:2]).copy()
        del trainer, model, training_result, encoder_before, classifier_before
        gc.collect()

        print("Reloading the saved model and tokenizer offline.", flush=True)
        saved_tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
        reloaded = AutoModelForMultipleChoice.from_pretrained(
            str(checkpoint), local_files_only=True, use_safetensors=True,
        )
        reloaded.eval()
        saved_features = [encode_example(validation_rows[i], saved_tokenizer, config["max_length"]) for i in range(2)]
        saved_probe = DataCollatorForMultipleChoice(saved_tokenizer)(saved_features)
        if not all(torch.equal(original_probe[name], saved_probe[name]) for name in original_probe):
            raise RuntimeError("Saved tokenizer changed the probe inputs or labels.")
        with torch.no_grad():
            reloaded_logits = reloaded(**saved_probe).logits.numpy()
        reload_error = float(np.max(np.abs(reloaded_logits - expected_logits)))
        if not np.allclose(reloaded_logits, expected_logits, atol=1e-6, rtol=1e-5):
            raise RuntimeError("Checkpoint reload changed validation predictions.")
    finally:
        monitor.stop()

    steady_steps = monitor.step_times[2:] or monitor.step_times
    steady_questions_per_second = effective_batch / float(np.mean(steady_steps))
    records = []
    for row, logits in zip(validation_rows, after.predictions):
        predicted_index = int(np.argmax(logits))
        predicted_label = row["choices"]["label"][predicted_index]
        records.append({
            "id": row["id"], "question": row["question"], "choices": row["choices"],
            "answerKey": row["answerKey"], "prediction": predicted_label,
            "correct": predicted_label == row["answerKey"], "logits": logits.tolist(),
        })
    prediction_path = PROJECT_ROOT / "reports" / f"{output_dir.name}-predictions.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    report = {
        "status": "completed", "purpose": config["purpose"], "config": config,
        "sources": {"model": source_model["source"], "dataset": source_data["source"]},
        "hardware": {"processor": platform.processor(), "logical_cpus": psutil.cpu_count(),
                     "physical_cpus": psutil.cpu_count(logical=False), "total_ram_bytes": psutil.virtual_memory().total},
        "selected_question_ids": {"train": train_ids, "validation": validation_ids},
        "training_answer_distribution": dict(Counter(train_rows["answerKey"])),
        "validation_answer_distribution": dict(Counter(validation_rows["answerKey"])),
        "effective_batch_size": effective_batch, "optimizer_steps": expected_steps,
        "baseline_validation_metrics": baseline.metrics,
        "after_training_validation_metrics": after.metrics,
        "training_metrics": train_metrics, "training_wall_seconds": training_wall_seconds,
        "training_log_history": log_history, "resources": monitor.summary(),
        "steady_training_questions_per_second": steady_questions_per_second,
        "estimated_full_training_compute_seconds_for_three_epochs": len(dataset["train"]) * 3 / steady_questions_per_second,
        "estimate_note": "Extrapolation from this small fixed subset, excluding evaluation/checkpoint overhead; CPU load and sequence lengths affect the actual time.",
        "checkpoint": checkpoint.relative_to(PROJECT_ROOT).as_posix(),
        "checkpoint_size_bytes": sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file()),
        "checks": {"encoder_max_parameter_change": encoder_delta,
                   "classifier_max_parameter_change": classifier_delta,
                   "gradient_accumulation_loss_normalization": "per-microbatch mean divided by actual accumulation steps",
                   "checkpoint_files_present": True, "offline_checkpoint_reload": "passed",
                   "reload_max_absolute_logit_difference": reload_error},
        "evaluation_note": f"Development accuracy on {len(validation_rows)} validation questions. Interpret according to the recorded run purpose and training subset; this is not a public test-set score.",
    }
    report_path = PROJECT_ROOT / "reports" / f"{output_dir.name}.json"
    save_json(report_path, report)
    print(json.dumps({"report": str(report_path), "training_wall_seconds": training_wall_seconds,
                      "dev_accuracy_before": baseline.metrics["dev_before_accuracy"],
                      "dev_accuracy_after": after.metrics["dev_after_accuracy"],
                      "peak_private_gib": monitor.peak_private_bytes / 1024**3,
                      "estimated_full_training_compute_hours_for_three_epochs": report["estimated_full_training_compute_seconds_for_three_epochs"] / 3600,
                      "reload_max_absolute_logit_difference": reload_error}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trial.json")
    arguments = parser.parse_args()
    config_path = (PROJECT_ROOT / arguments.config).resolve()
    run(json.loads(config_path.read_text(encoding="utf-8")))
