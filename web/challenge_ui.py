"""Challenge presentation and session-local Gradio controls."""

from html import escape
import logging

from web.challenge import next_question, start_round, submit_answer
from web.inference import LABELS
from web.presentation import render_prediction, render_scores

LOGGER = logging.getLogger(__name__)


def render_scoreboard(session):
    answered = len(session.attempts) if session else 0
    total = len(session.question_ids) if session else 10
    values = (("本轮进度", f"{answered} / {total}"),
              ("你答对", str(session.human_correct if session else 0)),
              ("模型答对", str(session.model_correct if session else 0)))
    return '<div class="challenge-scoreboard">' + "".join(
        f'<div><span>{label}</span><strong>{value}</strong></div>' for label, value in values) + '</div>'


def render_question(session, row):
    if session is None:
        return '<div class="empty-case"><h3>和本地模型一起答十道题</h3><p>点击开始新一轮，随机抽取十道不重复的验证题。</p></div>'
    return (f'<div class="challenge-question"><span class="muted">第 {session.position + 1} '
            f'/ {len(session.question_ids)} 题 · validation</span>'
            f'<h3 lang="en">{escape(row["question"])}</h3></div>')


def render_review(session, by_id):
    if session is None or not session.finished:
        return ""
    total = len(session.question_ids)
    if session.human_correct > session.model_correct:
        outcome = "本轮你领先模型"
    elif session.human_correct < session.model_correct:
        outcome = "本轮模型领先"
    else:
        outcome = "本轮双方平分"
    parts = [f'<div class="round-complete"><h3>本轮完成 · {outcome}</h3>'
             f'<p>你的正确率 <strong>{session.human_correct / total:.0%}</strong> · '
             f'模型正确率 <strong>{session.model_correct / total:.0%}</strong></p>'
             '<p class="muted">这是本轮十题抽样结果，不替代完整验证集的 58.97%。</p>'
             '<h4>逐题回顾</h4><div class="review-scroll"><table class="round-review">'
             '<thead><tr><th>题目</th><th>你选</th><th>模型</th><th>参考</th></tr></thead><tbody>']
    for index, attempt in enumerate(session.attempts, 1):
        row = by_id[attempt.question_id]
        human = "答对" if attempt.human_label == attempt.reference_label else "答错"
        model = "答对" if attempt.prediction.label == attempt.reference_label else "答错"
        parts.append(f'<tr><td lang="en">{index}. {escape(row["question"])}</td>'
                     f'<td>{attempt.human_label}<small>{human}</small></td>'
                     f'<td>{attempt.prediction.label}<small>{model}</small></td>'
                     f'<td>{attempt.reference_label}</td></tr>')
    return "".join(parts) + '</tbody></table></div></div>'


def render_feedback(session, by_id):
    if session is None or not session.answered:
        return '<div class="challenge-waiting">先选择你的答案，提交后再揭晓本题结果。</div>'
    attempt = session.attempts[session.position]
    row = by_id[attempt.question_id]
    answer = row["choices"]["text"][LABELS.index(attempt.reference_label)]
    human = "答对" if attempt.human_label == attempt.reference_label else "答错"
    model = "答对" if attempt.prediction.label == attempt.reference_label else "答错"
    html = (f'<div class="challenge-reveal"><h3>本题结果已揭晓</h3>'
            f'<p>你的选择 <strong>{attempt.human_label}</strong> · {human} '
            f'&nbsp; | &nbsp; 模型选择 <strong>{attempt.prediction.label}</strong> · {model}</p>'
            f'<p>参考答案 <strong>{attempt.reference_label}. {escape(answer)}</strong></p>'
            + render_scores(row["choices"]["text"], attempt.prediction.scores,
                            attempt.prediction.label, attempt.reference_label) + '</div>')
    return html + render_review(session, by_id)


def add_challenge_tab(experiment, runner):
    import gradio as gr

    by_id = {row["id"]: row for row in experiment.rows}
    with gr.Tab("人机答题挑战", id="challenge"):
        gr.Markdown("### 你和模型，谁更懂常识？\n每轮随机抽取 **10 道英文验证题**。你先作答，提交后本地模型再作答，一起揭晓结果。", elem_classes="intro")
        state = gr.State(None)
        scoreboard = gr.HTML(render_scoreboard(None), elem_id="challenge-scoreboard")
        question = gr.HTML(render_question(None, None), elem_id="challenge-question")
        selection = gr.Radio(choices=[], value=None, label="你的答案", interactive=False,
                             elem_id="challenge-options")
        with gr.Row():
            submit = gr.Button("提交并揭晓", variant="primary", interactive=False)
            advance = gr.Button("下一题", interactive=False)
            start = gr.Button("开始新一轮")
        feedback = gr.HTML(render_feedback(None, by_id), elem_id="challenge-feedback")
        timing = gr.Markdown("首次模型作答需要加载本地检查点，请稍候。")
        gr.Markdown("**本轮说明**：题目来自 validation，模型为本组已微调的 RoBERTa。每题实际运行本地模型；参考标签只用于提交后的核对。模型得分是选项间的相对得分。开始新一轮会清空当前成绩；记录仅保留在当前会话。", elem_classes="provenance")

        outputs = [state, scoreboard, question, selection, feedback, timing, submit, advance, start]

        def view(session, error=None, selected=None):
            row = by_id[session.question_ids[session.position]] if session else None
            answered = bool(session and session.answered)
            active = bool(session and not answered)
            choices = [(f"{label}. {text}", label) for label, text in zip(LABELS, row["choices"]["text"])] if row else []
            value = session.attempts[session.position].human_label if answered else selected
            result = render_feedback(session, by_id)
            if error:
                result = f'<div class="input-error" role="alert">{escape(error)}</div>' + result
            elapsed = "模型将在你提交后作答。"
            if answered:
                elapsed = render_prediction(session.attempts[session.position].prediction,
                                            row["choices"]["text"])[1]
            return (session, render_scoreboard(session), render_question(session, row),
                    gr.update(choices=choices, value=value, interactive=active), result, elapsed,
                    gr.update(interactive=active),
                    gr.update(interactive=answered and not session.finished), gr.update(interactive=True))

        def begin():
            return (gr.update(interactive=False), gr.update(interactive=False),
                    gr.update(interactive=False), gr.update(interactive=False),
                    '<div class="challenge-waiting">你的答案已提交，正在运行本地模型……</div>',
                    "正在作答，首次加载可能较慢。")

        def answer(session, label):
            try:
                updated = submit_answer(session, label, by_id, runner)
                return view(updated)
            except (ValueError, FileNotFoundError) as exc:
                return view(session, str(exc), label)
            except Exception:
                LOGGER.exception("Challenge model inference failed")
                return view(session, "本次模型作答未完成，尚未计分。请检查启动终端后重试。", label)

        def advance_question(session):
            try:
                return view(next_question(session))
            except ValueError as exc:
                return view(session, str(exc))

        def recover_submission(session, label):
            return view(session, "请求未能进入作答队列，尚未计分。请稍后重试。", label)

        start.click(lambda: view(start_round(experiment.rows)), outputs=outputs,
                    queue=False, api_name=False)
        advance.click(advance_question, inputs=state, outputs=outputs, queue=False, api_name=False)
        submission = submit.click(begin, outputs=[selection, submit, advance, start, feedback, timing],
                     queue=False, api_name=False).then(
            answer, inputs=[state, selection], outputs=outputs, api_name=False,
            concurrency_id="local-model", concurrency_limit=1,
        )
        submission.failure(recover_submission, inputs=[state, selection], outputs=outputs,
                           queue=False, api_name=False)
