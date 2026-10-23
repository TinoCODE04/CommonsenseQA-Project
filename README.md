# CommonsenseQA-Project

前端运行
cd D:\CommonsenseQA
.\.venv\Scripts\python.exe .\web\app.py

停止：
Get-CimInstance Win32_Process |
  Where-Object {
    $_.Name -eq 'python.exe' -and
    $_.CommandLine -match 'D:\\CommonsenseQA\\.*web[\\/]app\.py'
  } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

使用 RoBERTa-Base，在 Windows CPU 环境下开展 CommonsenseQA 五选一监督微调实验。

环境配置记录见 `environment-info.json`，依赖版本见 `requirements-lock.txt`。

在项目根目录运行以下命令，下载、检查并保存数据：

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_data.py
```

脚本会先通过 `scripts/project_env.py` 将 Hugging Face 缓存设置到
`.cache/huggingface/`，再导入相关库。该设置仅作用于当前 Python 进程。
已保存的数据会直接复用，并检查内容哈希；不会自动覆盖为上游的新版本。

- `data/commonsense_qa/`：可直接离线读取的数据，保持官方划分和原始内容。
- `reports/data-info.json`：来源版本、题数、标签分布、重复选项检查和内容哈希。
- `reports/data-samples.json`：各划分示例、标签索引示例和重复选项案例。

在项目根目录，可以这样读取已保存的数据：

```python
from datasets import load_from_disk

dataset = load_from_disk("data/commonsense_qa")
example = dataset["train"][0]
question = example["question"]
answers = example["choices"]["text"]
label_index = example["choices"]["label"].index(example["answerKey"])
```

公开测试集不提供正确答案。使用验证集选模型时，准确率应标为开发集准确率。
原始数据中存在相同选项文本，本项目记录这些情况，暂不删除、改写或重排数据。

数据准备完成后，下载并检查 RoBERTa-Base：

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_model.py
```

该脚本固定模型来源版本，只下载 Safetensors 权重、配置和分词器文件，
保存到项目的 Hugging Face 缓存。`reports/model-info.json` 记录本地快照位置、
文件校验和、权重加载情况，以及一道真实训练题目的前向和反向计算结果。
后续运行会复用记录的本地快照。

模型通过 `AutoModelForMultipleChoice` 加载。新增的多项选择评分层及缺失的
池化层参数会随机初始化，需要后续微调。准备脚本只检查梯度，不执行参数更新，
也不保存微调检查点；其输出不能作为正式问答准确率。

执行小规模训练试运行：

```powershell
.\.venv\Scripts\python.exe .\scripts\train.py --config configs/trial.json
```

试运行固定抽取 256 道训练题和 64 道验证题，采用 CPU FP32、每批 1 题、
梯度累积 16 次、1 个 epoch、学习率 `2e-5` 和动态填充（最大长度 128）。
每次实验从记录的原始预训练快照开始。非空实验目录会拒绝覆盖。

- `outputs/trial-roberta-base/checkpoint-16/`：模型、分词器、优化器及调度器状态。
- `reports/trial-roberta-base.json`：配置、抽样 ID、试运行开发集指标、速度、内存和重新加载验证。
- `reports/trial-roberta-base-predictions.jsonl`：64 道验证题的预测，供案例分析。
- `reports/token-lengths.json`：完整训练集和验证集的长度检查结果。

试运行报告中的准确率只对应指定的小样本，不能称为全量数据实验或测试集成绩。
耗时外推基于本机采样的训练速度，不包含完整验证和检查点保存开销。
已验证保存模型及分词器可重新加载；恢复优化器继续训练尚未在本阶段测试。

本机已完成上述试运行：训练约 179 秒，进程峰值私有内存约 3.41 GiB。
同一组 64 道验证题的准确率从 15/64（23.44%）变为 11/64（17.19%）；
这次仅训练 256 题、更新 16 步，结果用于检查流程，尚不足以评估全量微调效果。
按实测速率外推，完整 9741 题训练 3 个 epoch 的训练计算量约 5.65 小时，另需验证及保存时间。

`scripts/mcqa_trainer.py` 显式指定损失需要按梯度累积次数归一化，修正当前
Transformers 版本对 RoBERTa 通用 kwargs 接口的推断。背景和诊断记录见
`reports/training-compatibility.json`。运行测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

全量训练使用 `configs/full.json`。其中 `train_size` 和 `validation_size` 为 `0`，
表示使用官方完整划分：9741 道训练题、1221 道验证题。配置为 3 个 epoch、
CPU FP32、4 个计算线程、每批 1 题、梯度累积 16 次，预计共 1827 次参数更新。
模型从原始预训练快照开始；每轮结束保存检查点，并保留最近一个检查点。
开发集在训练开始前与三轮训练结束后各评估一次，最终指标对应固定训练三轮后的模型。
这是一项 RoBERTa-base 监督微调基线实验，模型及训练设置与原论文的基线存在差异。

首次启动命令如下；如果本轮已有训练进程在运行，请直接查看日志：

```powershell
.\.venv\Scripts\python.exe -u .\scripts\train.py --config configs/full.json
```

本轮由助手在后台启动时，进程信息及启动时的状态保存在
`reports/full-roberta-base-launch.json`。查看训练日志：

```powershell
Get-Content .\reports\full-roberta-base-training.log -Tail 15 -Wait
```

按 `Ctrl+C` 只会退出上述日志查看命令。训练完成后生成：

- `outputs/full-roberta-base/checkpoint-1827/`：最终模型、分词器及训练状态。
- `reports/full-roberta-base.json`：完整开发集准确率、参数、样本 ID、耗时和内存。
- `reports/full-roberta-base-predictions.jsonl`：全部 1221 道验证题的逐题预测。

运行期间请保持电脑接通电源并避免睡眠。标准错误日志位于
`reports/full-roberta-base-training.err.log`；进度条和权重加载提示也会写入该文件，
不能仅凭该文件非空判断训练失败。

本轮还启动了一个临时辅助进程，在训练工作进程存活期间请求 Windows 阻止闲置自动睡眠，
训练进程退出后释放请求。状态记录在 `reports/full-roberta-base-power.json`，
仅适用于本次启动；它不会阻止手动睡眠或合盖引发的睡眠。

`reports/resume-probe.json` 记录了底层 Trainer 的恢复检查：一个随机初始化的小模型
在第一轮保存、恢复至第三轮后，全部参数与连续训练完全一致；其中也包含不足一组的梯度累积。
该检查不等同于全尺寸检查点的恢复验证。当前训练入口不会自动恢复非空目录，
如训练中断，请保留输出目录，先检查现有检查点再处理恢复。

首次全量训练已完成：官方开发集 1221 题答对 720 题，准确率 58.97%；
本任务微调前为 17.94%，提高 41.03 个百分点。CPU 训练约 5 小时 16 分钟。
结果、训练曲线、实验边界和可用于课程汇报的文字见
`reports/first-baseline-summary.md`；完整预测已独立核对，记录见
`reports/full-result-verification.json`。这些数字是单次运行的开发集结果。
