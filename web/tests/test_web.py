"""Behavior checks for the course demo, without starting a web server."""

import importlib
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


def feature(name):
    try:
        return importlib.import_module(f"web.{name}")
    except ModuleNotFoundError as exc:
        raise AssertionError(f"Missing web feature: {name}") from exc


class InputTests(unittest.TestCase):
    def test_requires_question_and_all_five_answers(self):
        validate = feature("inference").validate_inputs
        for question, choices in [(" ", list("abcde")), ("Why?", ["a"] * 4),
                                  ("Why?", ["a", "b", "", "d", "e"])]:
            with self.subTest(question=question, choices=choices):
                with self.assertRaises(ValueError):
                    validate(question, choices)

    def test_duplicate_choice_text_keeps_all_letter_positions(self):
        question, choices = feature("inference").validate_inputs(" Where? ", ["bank"] * 5)
        self.assertEqual(question, "Where?")
        self.assertEqual(choices, ["bank"] * 5)

    def test_extreme_logits_return_finite_normalized_scores(self):
        result = feature("inference").scores_from_logits([1000, 1001, -1000, 0, 1])
        self.assertEqual(list(result), list("ABCDE"))
        self.assertTrue(all(math.isfinite(x) for x in result.values()))
        self.assertAlmostEqual(sum(result.values()), 1)
        self.assertEqual(max(result, key=result.get), "B")

    def test_nonfinite_logits_fail_instead_of_displaying_bad_scores(self):
        for logits in [[0, 1, 2, 3, float("nan")], [1, 2, 3]]:
            with self.assertRaises(ValueError):
                feature("inference").scores_from_logits(logits)

    def test_too_long_input_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "4000"):
            feature("inference").validate_inputs("x" * 4001, list("abcde"))

    def test_actual_tokenizer_preserves_five_choice_axis_without_labels(self):
        from transformers import AutoTokenizer
        inference = feature("inference")
        tokenizer = AutoTokenizer.from_pretrained(inference.DEFAULT_CHECKPOINT, local_files_only=True)
        choices = ["bank", "bookstore", "library", "mall", "park"]
        encoded = inference.tokenize_question(tokenizer, "Where can I buy a book?", choices)
        self.assertEqual(tuple(encoded["input_ids"].shape[:2]), (1, 5))
        self.assertLessEqual(encoded["input_ids"].shape[-1], 128)
        self.assertNotIn("labels", encoded)
        for index, answer in enumerate(choices):
            text = tokenizer.decode(encoded["input_ids"][0, index], skip_special_tokens=True)
            self.assertIn(answer, text)


class EvidenceTests(unittest.TestCase):
    def test_clearing_example_selection_returns_six_empty_inputs(self):
        self.assertEqual(feature("presentation").example_values(None), ("",) * 6)

    def test_real_records_recount_to_verified_metrics(self):
        exp = feature("experiment").load_experiment(ROOT)
        self.assertEqual(len(exp.rows), 1221)
        self.assertEqual(exp.correct, 720)
        self.assertEqual(exp.before_correct, 219)
        self.assertAlmostEqual(exp.accuracy, 720 / 1221)

    def test_filters_and_search_share_same_record_population(self):
        exp = feature("experiment").load_experiment(ROOT)
        self.assertEqual(len(exp.filter_rows("答错", "")), 501)
        self.assertEqual(len(exp.filter_rows("答对", "")), 720)
        self.assertEqual(exp.filter_rows("全部", "__no_such_question__"), [])
        row = exp.rows[0]
        self.assertEqual(exp.filter_rows("全部", row["id"]), [row])

    def test_case_html_escapes_dataset_content(self):
        exp = feature("experiment").load_experiment(ROOT)
        row = dict(exp.rows[0], question='<script>alert("bad")</script>')
        html = feature("presentation").render_case(row, exp.before[row["id"]])
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)


class AppTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejected_challenge_request_restores_controls_without_scoring(self):
        import random
        from web.challenge import start_round
        demo = feature("app").build_app()
        answer_id = next(key for key, fn in demo.fns.items() if fn.fn and fn.fn.__name__ == "answer")
        failures = [dep for dep in demo.config["dependencies"]
                    if dep["trigger_after"] == answer_id and dep["trigger_only_on_failure"]]
        self.assertEqual(len(failures), 1, "Rejected queued predictions need a failure recovery event")
        recovery = demo.fns[failures[0]["id"]]
        self.assertFalse(recovery.queue)
        session = start_round(feature("experiment").load_experiment().rows, rng=random.Random(42))
        restored = recovery.fn(session, "C")
        self.assertEqual(restored[0], session)
        self.assertEqual(session.attempts, ())
        self.assertTrue(restored[3]["interactive"])
        self.assertEqual(restored[3]["value"], "C")
        self.assertTrue(restored[6]["interactive"])
        self.assertFalse(restored[7]["interactive"])
        self.assertTrue(restored[8]["interactive"])

    async def test_app_builds_and_prediction_accepts_six_unlabeled_text_inputs(self):
        import asyncio
        asyncio.get_running_loop().set_debug(False)
        demo = feature("app").build_app()
        endpoints = [fn for fn in demo.fns.values() if fn.api_name == "predict"]
        self.assertEqual(len(endpoints), 1)
        self.assertEqual(len(endpoints[0].inputs), 6)
        labels = [component["props"].get("label") for component in demo.config["components"]
                  if component["type"] == "tabitem"]
        self.assertIn("人机答题挑战", labels)
        self.assertIn("微调变化分析", labels)
        self.assertEqual(demo.title, "常识问答实验台 · CommonsenseQA")


if __name__ == "__main__":
    unittest.main()
