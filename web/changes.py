"""Compare paired, verified validation predictions without loading another model."""

from html import escape

from web.inference import scores_from_logits
from web.presentation import render_scores

KINDS = ("错→对", "对→错", "一直答对", "一直答错")


def change_kind(row, before):
    return {(False, True): KINDS[0], (True, False): KINDS[1],
            (True, True): KINDS[2], (False, False): KINDS[3]}[(before["correct"], row["correct"])]


class ChangeAnalysis:
    def __init__(self, experiment):
        self.experiment = experiment
        self.kinds = {row["id"]: change_kind(row, experiment.before[row["id"]])
                      for row in experiment.rows}
        self.counts = {kind: sum(value == kind for value in self.kinds.values()) for kind in KINDS}

    def filter_rows(self, kind="全部", query=""):
        query = (query or "").strip().casefold()
        return [row for row in self.experiment.rows
                if (kind == "全部" or self.kinds[row["id"]] == kind)
                and (not query or query in row["question"].casefold() or query in row["id"].casefold())]


def render_change_summary(analysis):
    total = len(analysis.experiment.rows)
    descriptions = ("微调后新增答对", "微调后变为答错", "前后均选中参考答案", "前后均未选中参考答案")
    parts = ['<div class="change-stats">']
    for index, (kind, description) in enumerate(zip(KINDS, descriptions)):
        count = analysis.counts[kind]
        parts.append(f'<div class="change-stat tone-{index}"><span>{kind}</span>'
                     f'<strong>{count:,}<small>题</small></strong><p>{description}</p>'
                     f'<span class="muted">占全部验证题 {count / total:.1%}</span></div>')
    parts.append('</div>')
    net = analysis.counts[KINDS[0]] - analysis.counts[KINDS[1]]
    parts.append(f'<p class="change-net">同一批 {total:,} 道验证题，答对题数净变化 '
                 f'<strong>{net:+,} 题</strong> = 错→对 {analysis.counts[KINDS[0]]:,} '
                 f'− 对→错 {analysis.counts[KINDS[1]]:,}。</p>')
    return "".join(parts)


def render_change_case(row, before):
    if row is None:
        return '<div class="empty-case"><h3>没有匹配的变化案例</h3><p>请更换变化类型，或清空搜索内容。</p></div>'
    kind = change_kind(row, before)
    gold = row["answerKey"]
    answer = row["choices"]["text"][row["choices"]["label"].index(gold)]
    parts = [f'<div class="change-case"><span class="status">{kind}</span>'
             f'<h3 class="case-question" lang="en">{escape(row["question"])}</h3>'
             f'<p class="record-id">题目 ID：{escape(row["id"])}</p>'
             f'<p class="change-reference">参考答案 <strong>{gold}. {escape(answer)}</strong></p>'
             '<div class="change-comparison">']
    for title, record in (("微调前 · 补充复核", before), ("微调后 · 原运行记录", row)):
        outcome = "答对" if record["correct"] else "答错"
        parts.append(f'<section class="change-model"><h4>{title}</h4>'
                     f'<p>模型选择 <strong>{record["prediction"]}</strong> · {outcome}</p>'
                     + render_scores(row["choices"]["text"], scores_from_logits(record["logits"]),
                                     record["prediction"], gold) + '</section>')
    parts.append('</div><p class="provenance">微调前逐题结果来自事后重建初始模型的补充复核；'
                 '微调后来自原训练运行保存的记录。两者题目、选项与参考答案一致。'
                 '此处展示离线评估，得分是选项间的相对得分。</p></div>')
    return "".join(parts)
