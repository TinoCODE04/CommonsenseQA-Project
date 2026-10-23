"""Read experiment evidence; never overwrite reports or checkpoints."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

from web.inference import ROOT, LABELS, scores_from_logits


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"缺少实验文件：{path}。请从完整的项目目录启动。") from exc


def read_predictions(path):
    if not path.is_file():
        raise FileNotFoundError(f"缺少逐题预测文件：{path}。请保留 reports 目录。")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"逐题文件中出现重复 ID：{row['id']}")
        seen.add(row["id"])
        if row["choices"]["label"] != list(LABELS) or len(row["choices"]["text"]) != 5:
            raise ValueError("逐题文件的选项结构与本实验不一致。")
        scores = scores_from_logits(row["logits"])
        if (row["answerKey"] not in LABELS
                or row["prediction"] != max(scores, key=scores.get)
                or row["correct"] != (row["prediction"] == row["answerKey"])):
            raise ValueError("逐题文件的答案、得分与正确标记不一致。")
    return rows


@dataclass
class Experiment:
    root: Path
    report: dict
    data_info: dict
    rows: list
    before: dict
    correct: int
    before_correct: int

    @property
    def accuracy(self):
        return self.correct / len(self.rows)

    @property
    def before_accuracy(self):
        return self.before_correct / len(self.rows)

    def filter_rows(self, outcome="全部", query=""):
        query = (query or "").strip().casefold()
        return [row for row in self.rows
                if (outcome == "全部" or row["correct"] == (outcome == "答对"))
                and (not query or query in row["question"].casefold()
                     or query in row["id"].casefold())]


def load_experiment(root=ROOT):
    root = Path(root)
    report = read_json(root / "reports/full-roberta-base.json")
    data_info = read_json(root / "reports/data-info.json")
    rows = read_predictions(root / "reports/full-roberta-base-predictions.jsonl")
    before_rows = read_predictions(root / "reports/full-roberta-base-before-predictions-recheck.jsonl")
    before = {row["id"]: row for row in before_rows}
    if not rows or {row["id"] for row in rows} != set(before):
        raise ValueError("微调前后逐题文件的题目 ID 不一致。")
    for row in rows:
        original = before[row["id"]]
        if any(row[key] != original[key] for key in ("question", "choices", "answerKey")):
            raise ValueError("微调前后记录中的原题或参考答案不一致。")
    correct = sum(row["correct"] for row in rows)
    before_correct = sum(row["correct"] for row in before_rows)
    for actual, recorded in (
        (correct / len(rows), report["after_training_validation_metrics"]["dev_after_accuracy"]),
        (before_correct / len(rows), report["baseline_validation_metrics"]["dev_before_accuracy"]),
    ):
        if not math.isclose(actual, recorded, abs_tol=1e-10):
            raise ValueError("逐题计数与原始报告准确率不一致，请先检查实验文件。")
    if len(rows) != data_info["splits"]["validation"]["rows"]:
        raise ValueError("预测题数与验证集题数不一致。")
    return Experiment(root, report, data_info, rows, before, correct, before_correct)
