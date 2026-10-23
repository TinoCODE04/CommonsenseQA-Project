"""Run the saved multiple-choice checkpoint, without training or gold labels."""

from dataclasses import dataclass
import math
from pathlib import Path
from threading import Lock
from time import perf_counter

LABELS = tuple("ABCDE")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "outputs/full-roberta-base/checkpoint-1827"


def validate_inputs(question, choices):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("请填写英文问题，再点击开始预测。")
    if len(question) > 4000:
        raise ValueError("问题超过 4000 个字符，请缩短后重试。")
    if len(choices) != 5:
        raise ValueError("请提供 A–E 共五个选项。")
    normalized = []
    for label, text in zip(LABELS, choices):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"请填写选项 {label}，五个选项都需要有内容。")
        if len(text) > 1000:
            raise ValueError(f"选项 {label} 超过 1000 个字符，请缩短后重试。")
        normalized.append(text.strip())
    return question.strip(), normalized


def scores_from_logits(logits):
    values = [float(value) for value in logits]
    if len(values) != 5 or not all(math.isfinite(value) for value in values):
        raise ValueError("模型输出异常，请检查本地检查点后重试。")
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    return {label: value / total for label, value in zip(LABELS, exps)}


def tokenize_question(tokenizer, question, choices, max_length=128):
    """One question, five pairs: [1, 5, sequence_length]; no answer key."""
    question, choices = validate_inputs(question, choices)
    tokens = tokenizer(
        [question] * 5, choices, padding=True, truncation=True,
        max_length=max_length, return_token_type_ids=False, return_tensors="pt",
    )
    return {key: value.unsqueeze(0) for key, value in tokens.items()}


@dataclass(frozen=True)
class Prediction:
    label: str
    answer: str
    scores: dict
    logits: list
    inference_seconds: float
    loading_seconds: float


class ModelRunner:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, max_length=128, threads=4):
        self.checkpoint = Path(checkpoint)
        self.max_length = max_length
        self.threads = threads
        self._lock = Lock()
        self._model = None
        self._tokenizer = None

    def predict(self, question, choices):
        question, choices = validate_inputs(question, choices)
        with self._lock:
            loading_seconds = 0.0
            if self._model is None:
                required = ("config.json", "model.safetensors", "tokenizer.json")
                if any(not (self.checkpoint / name).is_file() for name in required):
                    raise FileNotFoundError(
                        f"找不到完整的训练检查点：{self.checkpoint}。请保留模型和分词器文件。"
                    )
                started = perf_counter()
                import torch
                from transformers import AutoModelForMultipleChoice, AutoTokenizer

                torch.set_num_threads(self.threads)
                tokenizer = AutoTokenizer.from_pretrained(
                    self.checkpoint, local_files_only=True,
                )
                model = AutoModelForMultipleChoice.from_pretrained(
                    self.checkpoint, local_files_only=True,
                    use_safetensors=True, dtype=torch.float32,
                ).to("cpu").eval()
                self._tokenizer, self._model = tokenizer, model
                loading_seconds = perf_counter() - started

            import torch

            started = perf_counter()
            tokens = tokenize_question(self._tokenizer, question, choices, self.max_length)
            with torch.inference_mode():
                logits = self._model(**tokens).logits[0].tolist()
            elapsed = perf_counter() - started
            scores = scores_from_logits(logits)
            label = max(scores, key=scores.get)
            return Prediction(label, choices[LABELS.index(label)], scores, logits,
                              elapsed, loading_seconds)
