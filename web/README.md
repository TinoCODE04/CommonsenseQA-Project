# 常识问答实验台

在本机浏览器体验已微调的 RoBERTa-base 五选一模型，浏览 CommonsenseQA 验证集案例与真实实验结果。

## 启动

在 PowerShell 中执行：

```powershell
cd D:\CommonsenseQA
.\.venv\Scripts\python.exe .\web\app.py
```

打开 <http://127.0.0.1:7860>。终端保持运行；停止时按 `Ctrl+C`。

也可以执行 `.\web\start.ps1`。如果系统阻止运行 PowerShell 脚本，直接使用上面的 Python 命令，无需更改系统执行策略。

端口占用时：

```powershell
.\.venv\Scripts\python.exe .\web\app.py --port 7861
```

页面只监听本机地址，未开启公网分享。第一次点击预测会加载约 0.5 GB 的模型权重，后续预测复用模型。实际内存与耗时以当前电脑运行状况为准。

## 安装网页依赖

本项目现有训练环境上增加 Gradio，保留根目录 `requirements-lock.txt`。从项目根目录执行：

```powershell
uv pip install --python .venv\Scripts\python.exe --cache-dir .cache\uv -c requirements-lock.txt -r web\requirements.txt
```

如果没有 uv，也可以使用已有 Python 环境中的 pip 执行对应的 `install -c requirements-lock.txt -r web\requirements.txt`。网页安装后的完整版本记录在 `web/requirements-lock.txt`；这是环境记录，不替代原训练锁文件。

## 如何演示

1. **模型体验**：选择内置示例，点击“开始预测”；也可以编辑英文问题和五个选项。这里执行真实本地模型推理，不读取案例缓存作为回答。预测期间暂停修改输入，完成后自动恢复；编辑题目会清空旧预测。
2. **案例分析**：切换“答对 / 答错”，按英文问题关键词或 ID 搜索。这里是已保存的验证集离线评估，包含全部 1,221 题：720 对、501 错。选择题目不消耗模型推理。
3. **实验结果**：查看微调前后验证集准确率、已有训练曲线、配置与结果来源。

条形图是五个候选答案间的 softmax 相对得分，不是经过校准的正确率。模型不会生成自由文本答案或解释。两条精选案例解读是展示用分析文字，不代表模型内部的推理过程。

## 文件依赖与来源

网页源码在本目录；模型和实验材料继续读取项目原位置，不复制检查点：

- `outputs/full-roberta-base/checkpoint-1827/`：推理只加载模型与分词器，不读取优化器。
- `reports/full-roberta-base.json`、`reports/data-info.json`：训练报告、配置与数据划分。
- `reports/full-roberta-base-predictions.jsonl`：原训练后保存的逐题预测。
- `reports/full-roberta-base-before-predictions-recheck.jsonl`：事后重建初始模型的补充评估；不是原运行当时的逐题文件。
- `reports/first-baseline-training.png`：已有训练曲线。

启动时核对题目、选项、参考答案、logits、计数与汇总准确率。缺少必须文件或不一致会停止启动并提示。单独复制 web 目录到其他电脑，仍需要一起提供相应模型、报告和 Python 环境。

推理过程不需要网络、不会重新训练。使用 CPU FP32 / 4 线程 / 最大长度 128，输入为五组问题与选项；参考答案不传给模型。Gradio 的临时文件位于项目 `.cache/gradio-web/`，未开放整个模型目录的文件下载。

## 检查

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s web\tests -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`DESIGN.md` 和 `PLAN.md` 记录实现范围与检查步骤。主入口 `app.py` 负责组件和事件；`inference.py` 负责模型；`experiment.py` 负责实验记录；`presentation.py` 和 `style.css` 负责显示。
