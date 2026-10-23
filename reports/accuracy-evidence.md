# 两个准确率数字从哪里来

准确率是模型对全部 1221 道官方开发题进行预测后，预测选项与标准答案相等的题数除以总题数。它不是根据训练 loss 估算出来的。

| 评估 | 正确 / 总数 | 准确率 |
|---|---:|---:|
| 本任务微调前的初始模型 | 219 / 1221 | 17.9361179%，四舍五入为 17.94% |
| 全量微调三轮后的模型 | 720 / 1221 | 58.9680589%，四舍五入为 58.97% |

**原始评估记录。** 打开 `reports/full-roberta-base.json`，搜索：

```text
baseline_validation_metrics.dev_before_accuracy = 0.17936117936117937
after_training_validation_metrics.dev_after_accuracy = 0.5896805896805897
```

原脚本 `scripts/train.py` 第 124 行在训练前评估，第 127 行开始训练，第 135 行在训练后评估。前后使用同一组开发题。`scripts/mcqa.py` 的 `compute_accuracy` 选择每题得分最高的选项，与正确标签比较后取平均。

**微调前记录的范围。** 原运行保存了微调前汇总准确率，没有保存微调前逐题预测；最初报告的 219 题由完整精度的准确率乘以 1221 还原。随后已用相同版本初始权重、随机种子 42、相同分词与评估设置，重建初始模型并单独评估；未创建优化器，也没有训练。补充评估重新答对 219 题，准确率和损失均与原始微调前记录一致。

这次补充评估的汇总记录为 `reports/initial-model-recheck.json`，其中：

```text
development_questions = 1221
development_correct = 219
development_accuracy = 0.17936117936117937
matches_original_before_accuracy = true
```

微调前逐题复核文件为 `reports/full-roberta-base-before-predictions-recheck.jsonl`。这是训练后补充生成的初始模型复核记录，不是原运行当时留存的逐题文件；也无法证明当时每一道题的预测都与现在完全相同，能够复现的是原先记录的整体指标。

**微调后逐题记录。** `reports/full-roberta-base-predictions.jsonl` 是原训练结束时保存的 1221 道逐题预测。重新逐行比较预测选项与标准答案，可直接统计得到 720 道正确。

两个逐题文件均为每行一题，关键字段为：

| 字段 | 含义 |
|---|---|
| `id` | 官方题目 ID |
| `question`、`choices` | 题目与五个选项 |
| `answerKey` | 官方标准答案选项 |
| `prediction` | 模型预测选项 |
| `correct` | 预测是否等于标准答案 |
| `logits` | 模型对五个选项输出的原始分数 |

在 `D:\CommonsenseQA` 的 PowerShell 中，可使用下面的只读命令重新统计两个文件。统计直接比较 `prediction` 和 `answerKey`，不依赖文件中已有的 `correct` 字段：

```powershell
$taskFiles = @(
  'reports/full-roberta-base-before-predictions-recheck.jsonl',
  'reports/full-roberta-base-predictions.jsonl'
)
foreach ($taskFile in $taskFiles) {
  $taskRows = @(Get-Content -LiteralPath $taskFile -Encoding UTF8 | ForEach-Object { $_ | ConvertFrom-Json })
  $taskCorrect = @($taskRows | Where-Object { $_.prediction -eq $_.answerKey }).Count
  '{0}: {1}/{2} = {3:P2}' -f $taskFile, $taskCorrect, $taskRows.Count, ($taskCorrect / $taskRows.Count)
}
```

两行结果应分别对应 `219/1221`、`720/1221`。这些都是开发集评估数据；开发题不参与模型参数更新。
