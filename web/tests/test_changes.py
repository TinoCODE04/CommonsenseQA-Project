"""Partition existing evidence and show paired scores without new model inference."""

import importlib
from types import SimpleNamespace
import unittest

from web.experiment import load_experiment


def feature():
    try:
        return importlib.import_module("web.changes")
    except ModuleNotFoundError as exc:
        raise AssertionError("Change analysis feature is missing") from exc


def experiment():
    exp = load_experiment()
    template = exp.rows[0]
    after, before = [], {}
    for name, old, new in [("improved", False, True), ("regressed", True, False),
                           ("stable-right", True, True), ("stable-wrong", False, False)]:
        row = dict(template, id=name, correct=new)
        after.append(row)
        before[name] = dict(row, correct=old)
    return SimpleNamespace(rows=after, before=before)


class ChangeTests(unittest.TestCase):
    def test_all_four_transitions_are_disjoint(self):
        analysis = feature().ChangeAnalysis(experiment())
        self.assertEqual(analysis.counts, {"错→对": 1, "对→错": 1, "一直答对": 1, "一直答错": 1})
        expected = {"错→对": "improved", "对→错": "regressed",
                    "一直答对": "stable-right", "一直答错": "stable-wrong"}
        for kind, record_id in expected.items():
            self.assertEqual([r["id"] for r in analysis.filter_rows(kind)], [record_id])

    def test_real_partition_reconciles_original_report(self):
        analysis = feature().ChangeAnalysis(load_experiment())
        c = analysis.counts
        self.assertEqual(sum(c.values()), 1221)
        self.assertEqual(c["错→对"] - c["对→错"], 501)
        self.assertEqual(c["一直答对"] + c["对→错"], 219)
        self.assertEqual(c["一直答对"] + c["错→对"], 720)

    def test_search_combines_transition_and_query_and_can_be_empty(self):
        analysis = feature().ChangeAnalysis(experiment())
        self.assertEqual([r["id"] for r in analysis.filter_rows("全部", "IMPROVED")], ["improved"])
        self.assertEqual(analysis.filter_rows("对→错", "improved"), [])
        self.assertEqual(analysis.filter_rows("全部", "__missing__"), [])

    def test_paired_case_escapes_question_and_displays_both_predictions(self):
        exp = load_experiment()
        row = dict(exp.rows[0], question="<script>bad()</script>")
        html = feature().render_change_case(row, exp.before[row["id"]])
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("微调前", html)
        self.assertIn("微调后", html)
        self.assertIn("事后重建", html)

    def test_no_selected_case_has_no_stale_scores(self):
        html = feature().render_change_case(None, None)
        self.assertIn("没有匹配", html)
        self.assertNotIn("score-list", html)


if __name__ == "__main__":
    unittest.main()
