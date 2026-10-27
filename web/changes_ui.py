"""The change explorer uses recorded evidence, with no extra model instance."""

from web.changes import ChangeAnalysis, KINDS, render_change_case, render_change_summary


def add_changes_tab(experiment):
    import gradio as gr

    analysis = ChangeAnalysis(experiment)
    by_id = {row["id"]: row for row in experiment.rows}

    def options(rows):
        return [(f"{index + 1}. {row['question'][:105]}", row["id"])
                for index, row in enumerate(rows)]

    def show(record_id):
        row = by_id.get(record_id)
        return render_change_case(row, experiment.before[row["id"]] if row else None)

    def count_text(rows, kind):
        return f"找到 {len(rows):,} 道题 · {kind} · 同一批验证题的离线记录"

    def filter_cases(kind, query):
        rows = analysis.filter_rows(kind, query)
        value = rows[0]["id"] if rows else None
        return gr.update(choices=options(rows), value=value), show(value), count_text(rows, kind)

    with gr.Tab("微调变化分析", id="changes"):
        gr.Markdown("### 微调让哪些题目发生了变化？\n把同一批验证题逐题对齐，区分改善、退步与未改变的结果。", elem_classes="intro")
        gr.HTML(render_change_summary(analysis), elem_id="change-summary")
        with gr.Row():
            kind = gr.Radio(["全部", *KINDS], value="错→对", label="变化类型", scale=2)
            query = gr.Textbox(label="搜索变化案例", placeholder="输入英文关键词或题目 ID", scale=2)
            search = gr.Button("搜索变化", scale=0)
        rows = analysis.filter_rows("错→对")
        value = rows[0]["id"] if rows else None
        count = gr.Markdown(count_text(rows, "错→对"))
        picker = gr.Dropdown(choices=options(rows), value=value, label="选择变化案例", filterable=True)
        display = gr.HTML(show(value), elem_id="change-case")
        picker.change(show, inputs=picker, outputs=display, queue=False, api_name=False)
        for event in (kind.change, query.submit, search.click):
            event(filter_cases, inputs=[kind, query], outputs=[picker, display, count], queue=False, api_name=False)
        gr.Markdown("**来源说明**：微调前是事后重建初始模型的补充复核；微调后是原训练运行的逐题评估。此页读取已有记录，无需加载第二个模型。改善与退步都发生在 validation 上，不能代替隐藏 test 成绩。", elem_classes="provenance")
