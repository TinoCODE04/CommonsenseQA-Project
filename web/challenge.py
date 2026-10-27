"""Immutable, per-browser challenge rounds. Gold labels never enter model inputs."""

from dataclasses import dataclass, replace
import random

from web.inference import LABELS, Prediction


@dataclass(frozen=True)
class Attempt:
    question_id: str
    human_label: str
    prediction: Prediction
    reference_label: str


@dataclass(frozen=True)
class Round:
    question_ids: tuple
    position: int = 0
    attempts: tuple = ()

    @property
    def answered(self):
        return len(self.attempts) > self.position

    @property
    def finished(self):
        return len(self.attempts) == len(self.question_ids)

    @property
    def human_correct(self):
        return sum(a.human_label == a.reference_label for a in self.attempts)

    @property
    def model_correct(self):
        return sum(a.prediction.label == a.reference_label for a in self.attempts)


def start_round(rows, size=10, rng=None):
    ids = list(dict.fromkeys(row["id"] for row in rows))
    if not 1 <= size <= len(ids):
        raise ValueError("题库中没有足够的不重复题目。")
    return Round(tuple((rng or random.SystemRandom()).sample(ids, size)))


def submit_answer(session, label, by_id, runner):
    if session is None:
        raise ValueError("请先开始一轮挑战。")
    if session.answered:
        return session  # Idempotent: an answered question cannot be rescored.
    if label not in LABELS:
        raise ValueError("请先选择 A–E 中的一个答案，再提交。")
    row = by_id[session.question_ids[session.position]]
    prediction = runner.predict(row["question"], list(row["choices"]["text"]))
    if prediction.label not in LABELS:
        raise ValueError("模型输出的选项无效，请重试。")
    attempt = Attempt(row["id"], label, prediction, row["answerKey"])
    return replace(session, attempts=(*session.attempts, attempt))


def next_question(session):
    if session is None or not session.answered:
        raise ValueError("请先提交当前题目的答案。")
    if session.finished:
        raise ValueError("本轮已完成，可以开始新一轮。")
    return replace(session, position=session.position + 1)
