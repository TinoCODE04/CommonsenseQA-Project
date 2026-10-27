"""Round transitions and scoring, including retries and duplicate submissions."""

import importlib
import random
import unittest

from web.inference import Prediction


def feature():
    try:
        return importlib.import_module("web.challenge")
    except ModuleNotFoundError as exc:
        raise AssertionError("Challenge feature is missing") from exc


def rows():
    return [dict(id=f"q{i}", question=f"Question {i}?",
                 choices={"label": list("ABCDE"), "text": ["first", "second", "third", "fourth", "fifth"]},
                 answerKey="C") for i in range(12)]


class Runner:
    def __init__(self):
        self.calls = []
        self.fail = False

    def predict(self, question, choices):
        self.calls.append((question, choices))
        if self.fail:
            raise RuntimeError("inference interrupted")
        return Prediction("B", choices[1], dict(zip("ABCDE", [.05, .8, .05, .05, .05])),
                          [0, 3, 0, 0, 0], .1, 0)


class ChallengeTests(unittest.TestCase):
    def setUp(self):
        self.rows = rows()
        self.by_id = {row["id"]: row for row in self.rows}
        self.runner = Runner()

    def start(self):
        return feature().start_round(self.rows, rng=random.Random(42))

    def test_start_has_ten_unique_questions_and_no_premature_score(self):
        session = self.start()
        self.assertEqual(len(session.question_ids), 10)
        self.assertEqual(len(set(session.question_ids)), 10)
        self.assertEqual(session.attempts, ())
        self.assertEqual((session.human_correct, session.model_correct), (0, 0))
        self.assertFalse(session.finished)

    def test_submission_scores_letters_even_when_option_text_is_duplicated(self):
        session = self.start()
        row = self.by_id[session.question_ids[0]]
        row["choices"]["text"] = ["same"] * 5
        updated = feature().submit_answer(session, "C", self.by_id, self.runner)
        self.assertEqual((updated.human_correct, updated.model_correct), (1, 0))
        self.assertEqual(self.runner.calls, [(row["question"], ["same"] * 5)])
        self.assertEqual(session.attempts, ())

    def test_duplicate_submission_does_not_repeat_model_call_or_change_score(self):
        first = feature().submit_answer(self.start(), "C", self.by_id, self.runner)
        second = feature().submit_answer(first, "A", self.by_id, self.runner)
        self.assertEqual(first, second)
        self.assertEqual(len(self.runner.calls), 1)
        self.assertEqual(second.human_correct, 1)

    def test_missing_or_invalid_choice_does_not_call_model(self):
        for label in [None, "", "Z"]:
            with self.subTest(label=label), self.assertRaises(ValueError):
                feature().submit_answer(self.start(), label, self.by_id, self.runner)
        self.assertEqual(self.runner.calls, [])

    def test_next_requires_an_answer_then_clears_answered_phase(self):
        session = self.start()
        with self.assertRaises(ValueError):
            feature().next_question(session)
        answered = feature().submit_answer(session, "A", self.by_id, self.runner)
        next_round = feature().next_question(answered)
        self.assertEqual(next_round.position, 1)
        self.assertFalse(next_round.answered)
        self.assertEqual(len(next_round.attempts), 1)

    def test_failed_inference_keeps_round_retryable_without_scoring(self):
        session = self.start()
        self.runner.fail = True
        with self.assertRaises(RuntimeError):
            feature().submit_answer(session, "C", self.by_id, self.runner)
        self.assertFalse(session.answered)
        self.assertEqual(session.attempts, ())
        self.runner.fail = False
        updated = feature().submit_answer(session, "C", self.by_id, self.runner)
        self.assertEqual(len(updated.attempts), 1)

    def test_full_round_scores_ten_questions_and_new_round_is_independent(self):
        session = self.start()
        for index in range(10):
            session = feature().submit_answer(session, "C", self.by_id, self.runner)
            if index < 9:
                session = feature().next_question(session)
        self.assertTrue(session.finished)
        self.assertEqual((session.human_correct, session.model_correct), (10, 0))
        with self.assertRaises(ValueError):
            feature().next_question(session)
        new_round = self.start()
        self.assertEqual(new_round.attempts, ())
        self.assertEqual(len(self.runner.calls), 10)

    def test_too_few_unique_questions_fail_instead_of_repeating(self):
        with self.assertRaises(ValueError):
            feature().start_round(self.rows[:9])

    def test_feedback_and_review_hide_answers_until_submission(self):
        try:
            ui = importlib.import_module("web.challenge_ui")
        except ModuleNotFoundError as exc:
            raise AssertionError("Challenge UI is missing") from exc
        session = self.start()
        hidden = ui.render_feedback(session, self.by_id)
        self.assertNotIn("参考答案", hidden)
        self.assertNotIn("score-list", hidden)
        self.assertEqual(ui.render_review(session, self.by_id), "")
        submitted = feature().submit_answer(session, "A", self.by_id, self.runner)
        visible = ui.render_feedback(submitted, self.by_id)
        self.assertIn("参考答案", visible)
        self.assertIn("score-list", visible)
        self.assertEqual(ui.render_review(submitted, self.by_id), "")

    def test_question_display_escapes_dataset_text(self):
        try:
            ui = importlib.import_module("web.challenge_ui")
        except ModuleNotFoundError as exc:
            raise AssertionError("Challenge UI is missing") from exc
        session = self.start()
        row = dict(self.by_id[session.question_ids[0]], question="<script>bad()</script>")
        html = ui.render_question(session, row)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()
