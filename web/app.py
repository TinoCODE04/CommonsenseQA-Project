"""Launch with the project's Python executable and web/app.py."""

import argparse
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.project_env import configure_cache

configure_cache()
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TEMP_DIR"] = str(ROOT / ".cache/gradio-web")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import gradio as gr

from web.experiment import load_experiment
from web.inference import ModelRunner
from web.presentation import (
    EMPTY_RESULT, example_values, experiment_details, render_case, render_overview, render_prediction,
)

LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


def case_options(rows):
    return [(f"{index + 1}. {row['question'][:105]}", row["id"])
            for index, row in enumerate(rows)]


def build_app():
    exp = load_experiment()
    by_id = {row["id"]: row for row in exp.rows}
    sample_rows = [exp.rows[index] for index in (1, 0, 2, 3, 4)]
    initial = sample_rows[0]
    runner = ModelRunner()

    def predict(question, a, b, c, d, e):
        choices = [a, b, c, d, e]
        try:
            result = runner.predict(question, choices)
            return render_prediction(result, [text.strip() for text in choices])
        except (ValueError, FileNotFoundError) as exc:
            from html import escape
            return f'<div class="input-error" role="alert">{escape(str(exc))}</div>', "请检查输入或本地文件后重试。"
        except Exception:
            LOGGER.exception("Local checkpoint inference failed")
            return '<div class="input-error" role="alert">模型推理未完成。请查看启动终端的错误信息，检查依赖和检查点后重试。</div>', "本次没有生成预测结果。"

    def load_example(sample_id):
        row = by_id.get(sample_id)
        message = "示例已载入，点击开始预测运行本地模型。" if row else "示例选择已清空，请输入题目或重新选择示例。"
        return *example_values(row), EMPTY_RESULT, message

    def show_case(sample_id):
        row = by_id.get(sample_id)
        return render_case(row, exp.before[row["id"]] if row else None)

    def filter_cases(outcome, query):
        rows = exp.filter_rows(outcome, query)
        value = rows[0]["id"] if rows else None
        return (gr.update(choices=case_options(rows), value=value),
                show_case(value), f"找到 {len(rows):,} 道题 · 当前来源：完整验证集的离线评估记录")

    with gr.Blocks(title="常识问答实验台 · CommonsenseQA", analytics_enabled=False) as demo:
        gr.HTML('''<header class="page-header"><div><h1>常识问答实验台<span>CommonsenseQA</span></h1>
<p>给模型一道题，看看它如何选择。</p></div><div class="local-badge">本地运行 · RoBERTa-base</div></header>''')
        with gr.Tabs():
            with gr.Tab("模型体验", id="experience"):
                gr.Markdown("输入**英文问题与五个选项**，运行本组已微调的模型。也可以先载入一道验证集示例。", elem_classes="intro")
                with gr.Row(equal_height=False, elem_id="inference-layout"):
                    with gr.Column(scale=6, min_width=320, elem_id="question-panel"):
                        gr.Markdown("### 输入题目")
                        sample_picker = gr.Dropdown(
                            choices=case_options(sample_rows), value=initial["id"],
                            label="载入验证集示例", filterable=True,
                        )
                        question = gr.Textbox(
                            value=initial["question"], label="英文问题", lines=3,
                            placeholder="Type an English commonsense question…", max_lines=6,
                        )
                        choices = [gr.Textbox(value=text, label=f"选项 {label}", lines=1,
                                              placeholder=f"Answer option {label}", max_lines=3)
                                   for label, text in zip("ABCDE", initial["choices"]["text"])]
                        with gr.Row():
                            submit = gr.Button("开始预测", variant="primary", scale=3)
                            clear = gr.Button("清空输入", scale=1)
                    with gr.Column(scale=4, min_width=300, elem_id="prediction-panel"):
                        gr.Markdown("### 模型作答")
                        prediction = gr.HTML(EMPTY_RESULT, elem_id="live-result")
                        timing = gr.Markdown("首次预测会加载本地模型，请稍候。", elem_id="timing")
                        gr.Markdown("**读懂得分**\n\n条形长度表示五个选项间的 softmax 相对得分，不能当作答案正确的概率。模型每次组合输入最多保留 128 tokens。", elem_classes="score-explanation")
                inputs = [question, *choices]
                editable_controls = [*inputs, sample_picker, submit, clear]

                def begin_prediction():
                    pending = '<div class="empty-result"><h3>正在运行本地模型</h3><p>首次加载耗时较长。<br>完成后会自动恢复输入。</p></div>'
                    return pending, "正在预测，请稍候。", *[gr.update(interactive=False) for _ in editable_controls]

                def finish_prediction():
                    return [gr.update(interactive=True) for _ in editable_controls]

                submit.click(begin_prediction, outputs=[prediction, timing, *editable_controls], queue=False, api_name=False).then(
                    predict, inputs=inputs, outputs=[prediction, timing], api_name="predict", concurrency_limit=1, scroll_to_output=True,
                ).then(finish_prediction, outputs=editable_controls, queue=False, api_name=False)
                sample_picker.change(load_example, inputs=sample_picker, outputs=[*inputs, prediction, timing], queue=False, api_name=False)
                clear.click(lambda: (*([""] * 6), EMPTY_RESULT, "输入已清空，请填写题目与五个选项。"),
                            outputs=[*inputs, prediction, timing], queue=False, api_name=False)
                for textbox in inputs:
                    textbox.change(lambda: (EMPTY_RESULT, "输入已修改，请重新点击开始预测。"),
                                  outputs=[prediction, timing], queue=False, api_name=False)
            with gr.Tab("案例分析", id="cases"):
                gr.Markdown("### 从答对与答错中观察模型\n这里展示已保存的验证集预测，切换案例无需运行模型。", elem_classes="intro")
                with gr.Row():
                    outcome = gr.Radio(["全部", "答对", "答错"], value="全部", label="筛选结果", scale=1)
                    query = gr.Textbox(label="搜索问题或题目 ID", placeholder="输入英文关键词或完整 ID，按回车搜索", scale=2)
                    search = gr.Button("搜索案例", scale=0)
                case_count = gr.Markdown(f"找到 {len(exp.rows):,} 道题 · 当前来源：完整验证集的离线评估记录")
                case_picker = gr.Dropdown(choices=case_options(exp.rows), value=exp.rows[0]["id"], label="选择题目", filterable=True)
                case_display = gr.HTML(show_case(exp.rows[0]["id"]), elem_id="case-display")
                case_picker.change(show_case, inputs=case_picker, outputs=case_display, queue=False, api_name=False)
                for event in (outcome.change, query.submit, search.click):
                    event(filter_cases, inputs=[outcome, query], outputs=[case_picker, case_display, case_count], queue=False, api_name=False)
                gr.Markdown("**记录来源**：微调后数据来自原训练运行保存的逐题评估。初始模型的逐题选择来自事后重建、补充复核，其准确率与原始汇总指标一致；原运行未保存微调前逐题预测。", elem_classes="provenance")
            with gr.Tab("实验结果", id="results"):
                gr.HTML(render_overview(exp))
                chart = ROOT / "reports/first-baseline-training.png"
                if chart.is_file():
                    gr.Image(value=str(chart), label="真实训练日志与验证集评估", interactive=False,
                             buttons=["download", "fullscreen"], elem_id="training-chart")
                gr.Markdown(experiment_details(exp))
                with gr.Accordion("结果来源与复现资料", open=False):
                    gr.Markdown("""页面启动时从本地报告重新计数，并核对微调前后题目与准确率。

- 汇总报告：`reports/full-roberta-base.json`
- 微调后逐题记录：`reports/full-roberta-base-predictions.jsonl`
- 初始模型复核：`reports/initial-model-recheck.json`
- 微调前复核逐题记录：`reports/full-roberta-base-before-predictions-recheck.jsonl`
- 当前推理检查点：`outputs/full-roberta-base/checkpoint-1827`

[原项目代码](https://github.com/jonathanherzig/commonsenseqa) · [数据集](https://huggingface.co/datasets/tau/commonsense_qa) · [预训练模型](https://huggingface.co/FacebookAI/roberta-base)
""")
        gr.HTML('<footer class="page-footer"><span>自然语言处理课程 · 常识问答基线</span><span>英文五选一 · 本机推理</span></footer>')
    return demo


def main():
    parser = argparse.ArgumentParser(description="CommonsenseQA local web demo")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    theme = gr.themes.Base(primary_hue="blue", neutral_hue="slate",
                           font=["Segoe UI", "Microsoft YaHei", "sans-serif"],
                           font_mono=["Consolas", "monospace"])
    # Keep the classroom surface consistently light, including on a dark OS.
    theme.set(**{name: getattr(theme, name[:-5]) for name in vars(theme)
                 if name.endswith("_dark") and hasattr(theme, name[:-5])})
    build_app().queue(default_concurrency_limit=1, max_size=8).launch(
        server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False,
        theme=theme, css=(HERE / "style.css").read_text(encoding="utf-8"),
        footer_links=[], ssr_mode=False,
        allowed_paths=[str(ROOT / "reports/first-baseline-training.png")],
        blocked_paths=[str(ROOT / name) for name in ("outputs", "data", ".venv", ".git", ".cache/huggingface")],
    )


if __name__ == "__main__":
    main()
