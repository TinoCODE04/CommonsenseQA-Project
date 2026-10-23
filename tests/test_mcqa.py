"""Catch label misalignment, unlabeled training rows, and incorrect batching."""

import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from project_env import configure_cache

configure_cache()
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
from transformers import AutoTokenizer, DataCollatorForMultipleChoice


class MultipleChoiceDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report = json.loads((ROOT / "reports/model-info.json").read_text(encoding="utf-8"))
        cls.tokenizer = AutoTokenizer.from_pretrained(
            str(ROOT / report["local_snapshot"]), local_files_only=True,
        )

    def helpers(self):
        self.assertIsNotNone(
            importlib.util.find_spec("mcqa"),
            "The training data helpers have not been implemented.",
        )
        return importlib.import_module("mcqa")

    def example(self):
        return {
            "id": "fixture", "question": "Which fruit should I pick?",
            "question_concept": "fruit",
            "choices": {
                "label": ["C", "A", "E", "B", "D"],
                "text": ["shared", "apple", "pear", "plum", "shared"],
            },
            "answerKey": "D",
        }

    def test_label_uses_letter_order_even_when_answer_text_is_duplicated(self):
        helpers = self.helpers()
        encoded = helpers.encode_example(self.example(), self.tokenizer, 128)
        self.assertEqual(encoded["labels"], 4)
        self.assertEqual(len(encoded["input_ids"]), 5)
        self.assertIn("apple", self.tokenizer.decode(encoded["input_ids"][1]))

    def test_unlabeled_public_test_rows_cannot_enter_supervised_training(self):
        helpers = self.helpers()
        for missing in ("", None):
            with self.subTest(answer=missing):
                row = self.example()
                row["answerKey"] = missing
                with self.assertRaises(ValueError):
                    helpers.encode_example(row, self.tokenizer, 128)

    def test_incomplete_choice_sets_are_rejected(self):
        helpers = self.helpers()
        row = self.example()
        row["choices"]["label"] = row["choices"]["label"][:4]
        row["choices"]["text"] = row["choices"]["text"][:4]
        with self.assertRaises(ValueError):
            helpers.encode_example(row, self.tokenizer, 128)

    def test_variable_length_rows_collate_to_five_choices_with_matching_labels(self):
        helpers = self.helpers()
        short_row = self.example()
        long_row = self.example()
        long_row["question"] = "Please consider the question carefully. " * 12
        long_row["answerKey"] = "B"
        features = [helpers.encode_example(row, self.tokenizer, 128) for row in (short_row, long_row)]
        batch = DataCollatorForMultipleChoice(self.tokenizer)(features)
        self.assertEqual(tuple(batch["input_ids"].shape[:2]), (2, 5))
        self.assertEqual(batch["labels"].tolist(), [4, 3])
        self.assertTrue((batch["attention_mask"][0] == 0).any().item())
        padded_ids = batch["input_ids"][batch["attention_mask"] == 0]
        self.assertTrue((padded_ids == self.tokenizer.pad_token_id).all().item())

    def test_accuracy_counts_correct_options_instead_of_highest_label_numbers(self):
        helpers = self.helpers()
        logits = np.array([[0, 8, 0, 0, 0], [9, 0, 0, 0, 0], [0, 0, 0, 0, 7]])
        result = helpers.compute_accuracy((logits, np.array([1, 2, 4])))
        self.assertAlmostEqual(result["accuracy"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
