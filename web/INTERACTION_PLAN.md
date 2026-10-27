# Challenge and Change Analysis Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, task by task. Steps use checkbox syntax for tracking.

**Goal:** 加入真实模型参与的十题挑战和已有验证记录的前后变化分析。
**Architecture:** 不可变会话负责答题状态；既有 Experiment 负责已核验记录；两个独立 UI 模块接入 app。
**Tech Stack:** Python / existing Gradio / local CPU RoBERTa。
**Spec:** web/INTERACTION_DESIGN.md。

## Global Constraints

所有交付文件位于 web；不新增依赖；不训练、不接外部 API；原检查点、数据和报告只读。按用户指定的本地项目就地执行，不创建分支或提交。自动设计上下文加载不可用，直接以 web/DESIGN.md 和现有源码为界面依据。

## Review Focus

重复提交不重复推理或计分；漏选与推理失败不能吞掉题目；独立会话不得共享成绩；原题重复选项按字母计分；空筛选不能保留旧案例。

## Task 1: Challenge

Files: challenge.py, tests/test_challenge.py。
Interfaces: start_round(rows, size=10, rng=None) -> Round; submit_answer(round, label, by_id, runner) -> Round; next_question(round) -> Round。

- [x] 测试先失败：十题唯一、提交前零成绩、标签计分、重复提交不重复调用、漏选/越过未答题无推理、失败后原状态可重试、完整一轮及新一轮隔离。
- [x] 最小实现不可变 Round/Attempt；题目和选项才是 runner 输入。
- [x] 对应测试通过。

## Task 2: Change analysis

Files: changes.py, tests/test_changes.py。
Interfaces: ChangeAnalysis(exp).counts / filter_rows(kind, query); render_change_summary(analysis), render_change_case(row, before)。

- [x] 测试先失败：手算四类、真实 1,221 题分区、净增加 501、筛选搜索、空结果和 HTML 转义。
- [x] 实现分类及并排得分展示，保留来源说明。
- [x] 对应测试通过。

## Task 3: UI, docs, verification

Files: challenge_ui.py, changes_ui.py, app.py, style.css, README.md, DESIGN.md, tests/test_web.py。

- [x] 加入两个标签页，挑战实际使用共享 runner；提交前隐藏答案，忙碌锁定与错误恢复，满十题总结。
- [x] 加入四类总览和筛选搜索、空结果；原三个标签页继续工作。
- [x] 运行 web 全套和原 tests；用本地真实模型完成一轮，测试桌面/窄屏、原页面与控制台。
- [x] 一次独立整体复查、修复重要问题；结束测试服务并记录验证。

## 执行记录

Ruling: 用户已明确选择并要求实现，直接执行已讨论方案；不再增加审批步骤。当前工作区干净；Git 元数据只读，且交付目标是当前 web，故不创建 worktree 或提交。
Pre-flight: Task3 消费 Task1/Task2 的公开接口，并与原 predict 共享 local-model 队列；不改 ModelRunner 编码或现有记录格式。

## 验证记录（2026-10-27）

- 先运行缺失功能测试失败，再实现通过；独立复查后的队列拒绝问题也经过 RED→GREEN。最终 web 27 项、原 tests 6 项通过。
- 四类完整记录：599 / 98 / 121 / 403，合计 1,221；净增加 501，与 219→720 一致。
- Edge 浏览器完成真实模型十题挑战，测试选择固定为 A（用于验证流程，不是人类表现测量）：用户 1/10、模型 7/10，最终回顾逐题一致；页面注明本轮抽样不替代完整验证集成绩。
- 验证提交前隐藏结果、漏选不计分、下一题清空选择、新一轮清零、独立浏览器上下文不共享成绩；受控延迟请求时选择/提交/下一题/新一轮均禁用，完成后恢复。
- 独立复查发现：先禁用控件再入队，队列满会使原回答回调不执行。新增非排队 failure 恢复事件。浏览器拦截实际 queue/join? 请求模拟 HTTP503，确认保留选择、恢复提交、未增加分数，随后真实重试成功。
- 变化分析验证全部四类及全部题目计数、ID 搜索、一条命中、空搜索结果清空旧得分。微调前后来源明确。
- 正常流程原模型返回 complete job，旧结果随编辑清空、空输入提示正确，案例答错 501，实验页 720/1221 和 58.97% 正常；正常流程监听 console error/warning 与 pageerror 均为零。模拟503及测试重启产生的预期网络日志不当作正常流程错误。
- 1440×1000、390×844 均无页面横向溢出，手机对照得分纵向排列；手机 More tabs 能访问两新页；深色偏好下维持原浅色设计。
- Browser plugin not available，使用已缓存 Playwright CLI 与已安装 Edge。截图保存在系统临时目录 csqa-interaction-qa-20261027，未加入项目。自动设计上下文工具缺少可运行引擎，未安装新工具，直接按既有 web/DESIGN.md 与界面实现。
- 已关闭本次专用浏览器与 7862 测试服务，socket 确认 7862 无监听。7860 检测到有服务运行，未停止该现有服务；需要用户在原终端重启以重新加载新页面。Git 变更均位于 web，原训练产物未写入。
