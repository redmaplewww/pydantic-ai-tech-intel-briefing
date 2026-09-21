# 学习闭环与评测闭环咬合（路径 B）

本文档描述"经验沉淀 → 统计验证 → 门控合入"的自进化链路：Bad Case 修复后
自动沉淀为带版本的技能文件，技能版本变更自动触发 eval-replay 回归，由
McNemar 判定改进是否统计显著，显著且无回归才允许合入。

## 1. 设计目标

- 把"修一个坏例"沉淀为**可复用、带版本、可回滚**的程序性技能，而不是只留在
  日志或记忆里。
- 技能变更不能静默生效：必须先过统计回归门（复用现有 Delta + McNemar），
  p < 0.05 **且**无回归才合入；否则拒绝并归档。
- 每次"学习 → 回归 → 判定"全链路落盘（JSON 台账 + 人类可读 REPORT.md），
  可追溯到触发它的 Bad Case。

## 2. 技能库与版本约定

目录约定（默认根为 `skills/`，可用 `--data-dir` 隔离运行）：

```
skills/<slug>/SKILL.md             active 技能（生效，带 frontmatter）
skills/staging/<slug>/SKILL.md     草稿（staging，不生效）
skills/.archive/<slug>/SKILL.md    废弃/被拒技能（归档，永不删除）
skills/audits/<cycle_id>/ledger.json   一次学习循环的 JSON 台账
skills/audits/<cycle_id>/REPORT.md     一次学习循环的人类可读报告
```

SKILL.md 使用 frontmatter：

```markdown
---
name: review-rejection-recovery
version: 3
description: Recover from a review-rejected answer.
related:
  - search-discipline
---
```

- `version` 每次修改递增（草稿重写 +1；合入覆盖 active 时在 active 版本上 +1）。
- frontmatter 解析是手写的（无 PyYAML 依赖），支持标量与 `- item` 列表。
- 废弃技能移入 `.archive/` **而非删除**；归档时写入 `archived` 与
  `archive_reason` 字段便于审计。

## 3. 学习触发器

`search_assistant.skills.learning.SkillLearningTrigger`：Bad Case（带
`id/question/quality_flags/root_causes/attribution_layer`）修复后，按内置
`DRAFT_INSTRUCTIONS` 模板调用 LLM（注入式 `llm_runner(instructions, payload)`，
与 `_build_quality_judge` 的 runner 约定一致，测试可 mock）生成 SKILL.md，
解析 frontmatter 后写入 `skills/staging/<slug>/SKILL.md`，并记录
`source_bad_case` 便于追溯。草稿只入 staging，不直接生效。

## 4. Replay 钩子与回归

`search_assistant.skills.regression.SkillRegressionHook`：监听技能版本变更
（staging → active），对评测报告中的题目逐题重跑，逐维比较新旧答案的
Rubric 判定（复用 `cli._judge_verdicts` / `_verdict_rank`），统计
improved/regressed/stable 维度与题目数，并用 `cli._mcnemar_p_value`（双边精确
McNemar）给出 p 值与显著性。`answer_runner` 可注入（生产接 workflow，测试接
fake）。

## 5. 发布门

`search_assistant.skills.release.SkillReleaseGate` + `decide_regression`：

- **合入条件**：`p < 0.05` 且 `regressed_questions == 0` 且 `regressed == 0`。
- 合入：staging → active（版本在现有 active 版本上 +1），gate 记录 `passed`，
  ledger 状态 `completed`。
- 拒绝：staging → `.archive/`（`archived: rejected`），gate 记录 `failed`，
  ledger 状态 `rejected`，reason 记录 p 值与判定依据。
- 每次判定写 `agentops_gate_records`（gate_type=`skill_release`）与
  `project_ledger_entries`（entry_type=`skill_learning`），metadata 含
  p 值、显著性、improved/regressed 计数与 bad_case_id。

McNemar 边界（与 `eval-replay` 一致）：无 discordant 对 → p=1.0；5/0 →
p=0.0625（不显著）；6/0 → p=0.03125（显著）。

## 6. 审计

`search_assistant.skills.audit.SkillLearningAudit`：每个 cycle 写
`skills/audits/<cycle_id>/ledger.json`（完整 JSON，含 bad_case、draft、
regression、release、audit 路径）与 `REPORT.md`（人类可读，含触发 Bad Case
id、技能 slug/version、回归统计、判定理由、gate/ledger id）。

## 7. CLI

```
python -m search_assistant.cli skill-library-list
python -m search_assistant.cli skill-library-archive <slug> --reason "..."
python -m search_assistant.cli skill-learning-loop \
    --bad-case <question_id> --fix "<fix note>" [--judge rule|llm]
```

`skill-learning-loop` 完整执行 learn → regress → decide → audit；退出码
0 表示合入，1 表示拒绝或失败。`--bad-case-json` 可直接传入 JSON bad case，
`--report` 可指定评测报告路径（默认 `--data-dir/evaluations/evaluation-report.json`）。

## 8. 与现有能力的关系（不重构）

- 复用 `cli._judge_verdicts` / `_verdict_rank` / `_mcnemar_p_value`（懒加载，
  保持模块图无环）。
- 复用 `store.add_gate_record` / `add_project_ledger_entry`。
- 复用 `RuleQualityJudge` / `LLMRubricJudge` 作为可插拔 judge。
- 不修改 `eval-suite` / `eval-replay` / Judge / 现有 `SkillDraftService`。
