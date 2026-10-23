"""HTML presentation of real model outputs and recorded experiment evidence."""

from html import escape

from web.inference import LABELS, scores_from_logits

EMPTY_RESULT = '''<div class="empty-result"><span class="empty-letter">?</span>
<h3>等待一次预测</h3><p>填写左侧题目与五个选项，<br>查看模型如何作答。</p></div>'''


def example_values(row):
    if row is None:
        return ("",) * 6
    return row["question"], *row["choices"]["text"]


def render_scores(choices, scores, prediction, gold=None):
    parts = ['<div class="score-list">']
    for label, text in zip(LABELS, choices):
        score = scores[label]
        tags = []
        if label == prediction:
            tags.append("模型选择")
        if label == gold:
            tags.append("参考答案")
        state = " gold" if label == gold else (" selected" if label == prediction else "")
        parts.append(
            f'<div class="score-row{state}"><div class="score-heading">'
            f'<span class="choice-letter">{label}</span><span class="choice-text" lang="en">'
            f'{escape(text)}</span><strong>{score:.1%}</strong></div>'
            f'<div class="score-track" role="meter" aria-label="选项 {label} 相对得分" '
            f'aria-valuenow="{score * 100:.2f}" aria-valuemin="0" aria-valuemax="100">'
            f'<div class="score-fill" style="width:{score * 100:.4f}%"></div></div>'
            f'<div class="score-tags">{" · ".join(tags)}</div></div>'
        )
    parts.append('</div>')
    return "".join(parts)


def render_prediction(prediction, choices):
    result = (
        f'<div class="prediction-title"><span class="answer-letter">{prediction.label}</span>'
        f'<div><span class="muted">模型选择</span><h3 lang="en">{escape(prediction.answer)}</h3></div></div>'
        + render_scores(choices, prediction.scores, prediction.label)
    )
    timing = f"本机 CPU 推理耗时 {prediction.inference_seconds:.3f} 秒（含分词，不含模型加载）"
    if prediction.loading_seconds:
        timing += f"；本次首次加载另耗时 {prediction.loading_seconds:.2f} 秒。"
    else:
        timing += "；复用已加载模型。"
    return result, timing


CASE_NOTES = {
    "1afa02df02c908a558b4036e80242fac": "题目包含 security measure 这一用途线索。参考答案为 bank；模型选择 mall，可能更依赖旋转门与购物场所的关联。这里是基于题意的案例解读，不能据此确定模型内部的推理过程。",
    "a7ab086045575bb497933726e4e6ad28": "问题询问工作时的主要目标，complete job 与目标直接对应。模型选中了参考答案；其余选项包含交谈等可能发生的行为，但不是题目所问的主要目标。",
}


def render_case(row, before):
    if row is None:
        return '<div class="empty-case"><h3>没有匹配的题目</h3><p>请更换筛选条件，或清空搜索内容。</p></div>'
    outcome = "答对" if row["correct"] else "答错"
    badge = "good" if row["correct"] else "bad"
    html = (
        f'<div class="case-top"><span class="status {badge}">{outcome}</span>'
        f'<span class="muted">验证集 · 已保存的离线评估</span></div>'
        f'<h3 class="case-question" lang="en">{escape(row["question"])}</h3>'
        f'<p class="record-id">题目 ID：{escape(row["id"])}</p>'
        + render_scores(row["choices"]["text"], scores_from_logits(row["logits"]),
                        row["prediction"], row["answerKey"])
        + f'<p class="case-comparison">参考答案 <strong>{row["answerKey"]}</strong> '
          f'&nbsp; · &nbsp; 微调后选择 <strong>{row["prediction"]}</strong> '
          f'&nbsp; · &nbsp; 初始模型补充复核选择 <strong>{before["prediction"]}</strong></p>'
    )
    if row["id"] in CASE_NOTES:
        html += f'<div class="case-note"><h4>案例解读</h4><p>{escape(CASE_NOTES[row["id"]])}</p></div>'
    return html


def render_overview(exp):
    total = len(exp.rows)
    improvement = (exp.accuracy - exp.before_accuracy) * 100
    return (
        '<div class="experiment-summary"><h2>一次完整的基线实验</h2>'
        f'<p>在同一组 <strong>{total:,}</strong> 道验证题上，微调后的准确率为 '
        f'<strong>{exp.accuracy:.2%}</strong>，较微调前提高 <strong>{improvement:.2f} 个百分点</strong>。</p>'
        '<table class="metrics-table"><thead><tr><th>评估阶段</th><th>答对 / 总题数</th><th>准确率</th></tr></thead>'
        f'<tbody><tr><td>微调前</td><td>{exp.before_correct} / {total}</td><td>{exp.before_accuracy:.2%}</td></tr>'
        f'<tr><td>微调 3 轮后</td><td>{exp.correct} / {total}</td><td>{exp.accuracy:.2%}</td></tr>'
        '<tr><td>五选一均匀随机猜测（理论期望）</td><td>—</td><td>20.00%</td></tr></tbody></table>'
        '<p class="muted">以上为开发集 / validation 成绩；公开 test 不提供答案，尚无隐藏测试集成绩。</p></div>'
    )


def experiment_details(exp):
    cfg = exp.report["config"]
    seconds = exp.report["training_wall_seconds"]
    splits = exp.data_info["splits"]
    return f'''### 数据与训练设置

| 项目 | 本次实验 |
| :--- | :--- |
| 模型 | RoBERTa-base → 五选一评分模型 |
| train / validation / test | {splits["train"]["rows"]:,} / {splits["validation"]["rows"]:,} / {splits["test"]["rows"]:,} |
| 训练轮数 / 参数更新 | {cfg["epochs"]} / {exp.report["optimizer_steps"]:,} |
| 最大输入长度 | {cfg["max_length"]} tokens / 问题与候选答案组合 |
| 每批题数 / 梯度累积 | {cfg["batch_size"]} / {cfg["gradient_accumulation_steps"]} |
| 学习率 / 权重衰减 | {cfg["learning_rate"]:g} / {cfg["weight_decay"]} |
| Warmup / 随机种子 | {cfg["warmup_fraction"]:.0%} / {cfg["seed"]} |
| 计算环境 | Windows · CPU FP32 · {cfg["threads"]} 线程 |
| 训练耗时 | {int(seconds // 3600)} 小时 {int(seconds % 3600 // 60)} 分钟（不含前后评估） |

### 如何理解结果

本次是一个随机种子的 RoBERTa-base 基线实验。训练损失下降、验证集准确率提高，说明微调在当前开发集上有效；仍有 {len(exp.rows) - exp.correct} 题答错。尚未进行多随机种子统计，也未逐轮评估验证集，因此不能认定第 3 轮是最优轮次。

原论文使用 BERT-LARGE 等模型；论文表 5 的 55.9% 是其 random split 的 **test** 成绩。本页 58.97% 是本组 **validation** 成绩，模型和评估设置有差异，不能直接据此声称超过原论文。
'''
