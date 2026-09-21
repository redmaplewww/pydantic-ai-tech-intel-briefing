# 知识图谱与自进化系统改进计划

> Historical planning note: this document records the branch and evolution
> planning state that led into PR #10. Current runtime architecture is governed
> by ADR 0001: Pydantic AI is the primary agent runtime; GLM and DeepSeek are
> model providers, not separate agent frameworks.
>
> 基于《AI Agents in Depth》教程第 3 章和第 8 章，结合当前系统诊断结果。
> **状态：已标注所有分支与改进计划的冲突与重叠。**
> **更新时间：2026-08-17**

---

## 全分支冲突分析

### 分支总览

| 分支名 | 基于版本 | 提交数 | 与改进计划的关系 |
|--------|---------|-------|----------------|
| `main` | d64d99e | 0 | 无知识图谱/自进化代码，计划全部待做 |
| `agent/agent-reach-briefing-runtime` | main + 3 | 3 | ❌ 无知识图谱/自进化，只改搜索 provider |
| `agent/domain-knowledge-graphs` | main + 1 | 1 | ✅ 知识图谱初始版本（PR #3），不含 embedding |
| `agent/evolution-knowledge-graph-compat` | main + 2 | 2 | ✅ 自进化初始版本（PR #4），不含 embedding |
| `agent/filter-marketing-content` | main + 1 | 1 | ❌ 只改 briefing 过滤，与计划无关 |
| `agent/p0-p1-source-observability-memory` | main + 1 | 1 | ⚠️ 可观测性+分层记忆，已合入 pr10-work |
| `agent/phase-one-verifiable-evolution` | main + 3 | 3 | ⚠️ 自进化初版（无知识图谱），已合入 pr10-work |
| `agent/agentops-infrastructure-readiness` | main + 14 | 14 | ✅ 知识图谱+自进化+agentops（PR #6），不含 embedding |
| `agent/dynamic-fallback-headings` | main + 15 | 15 | ✅ 同上 + 动态标题（PR #9），不含 embedding |
| `agent/search-budget-timeout-fix` | main + 2 | 2 | ⚠️ P0/P1 + 搜索预算修复，已合入 pr10-work |
| **`agent/tiered-search-budget-observability`** | main + 31 | 31 | **✅ 最完整分支（= pr10-work + 3 个提交），含 embedding** |

### 关键发现：`agent/tiered-search-budget-observability` 是当前最完整分支

这个分支就是 pr10-work 的完整版，包含了所有 PR 合入 + 3 个之前本地丢失的提交：

1. **3a6bfc4** — P1-1 minimal fixes：layered memory read path, candidate query isolation, bilingual graph aliases
2. **37baf5f** — Add semantic similarity matching to knowledge graph queries
3. **c26d6e7** — Allow embedding provider to fall back to DeepSeek credentials

### 逐项冲突对比

| 改进计划项 | 所有分支覆盖情况 | 冲突/重叠 |
|-----------|----------------|----------|
| **P0: Embedding Provider 接线修复** | ⚠️ **部分覆盖** | `embedding.py` + `knowledge_graph/service.py` 中的 embedding 逻辑已存在（37baf5f, c26d6e7），但 `briefing/service.py:1237` 仍创建 `DomainKnowledgeGraphService(self.store)` 不传 `embedding_provider`，`evolution/service.py:66` 同样不传。**核心 bug 未修复**。 |
| **P0: _candidate_matches_topic 过宽** | ⚠️ **部分覆盖** | 3a6bfc4 修复了 `_knowledge_guided_queries()` 中候选查询隔离（不再拼接 claim terms 到搜索查询），但 `_candidate_matches_topic()` 仍然拼接 topic+claim+applies_when 做宽泛匹配。**核心 bug 未修复**。 |
| **P0: 验证基础功能** | ❌ 未覆盖 | DeepSeek embedding API 是否可用未验证 |
| **P1: 混合检索** | ❌ 未覆盖 | 所有分支只有 dense embedding，无 BM25/RRF/Reranking |
| **P1: 图谱动态扩展** | ❌ 未覆盖 | 所有分支只有 `seeds.py` 硬编码种子图谱 |
| **P1: 多类型记忆查询路由** | ⚠️ **部分覆盖** | `layered_memory_items` 表 + `add_layered_memory_item` 写入已有（7e01cbf），`workflow/service.py` 中的 `_memory_context()` 和 `_experience_context()` 已改用 layered_memory_items（3a6bfc4），但 `_knowledge_context()` 仍不按 memory_type 分路检索 |
| **P2: 在线/离线双循环** | ⚠️ **部分覆盖** | 在线记录已有（ProviderTraceEvent, SourceCandidate, BriefingCollectionTrace, TopicFeedbackSignal），离线进化循环无 |
| **P2: 轨迹验证三层评估** | ⚠️ **部分覆盖** | `EvolutionDiagnosisService.diagnose()` 已有 result/process/quality 三层 flag 分类（phase-one-verifiable-evolution 分支），但这是**诊断**不是**验证**。验证需要 LLM 评估 claim-source 一致性、query 覆盖度、质量 Rubric，目前无。 |
| **P2: 跨轨迹知识蒸馏** | ❌ 未覆盖 | `agentops_gate_records` 的 `domain_knowledge_candidate_validation` gate 只做单次验证，无跨轨迹支持阈值 |
| **P2: 发布/回滚/灰度自动化** | ❌ 未覆盖 | 无 |

### 具体代码冲突分析

#### 1. `briefing/service.py` — 最可能冲突的文件

当前 `agent/tiered-search-budget-observability` 分支的 briefing service 有 2641 行，包含：
- 搜索可观测性（7e01cbf）
- 候选查询隔离（3a6bfc4）
- 知识图谱/自进化上下文（PR #6, #9）

我的计划需要修改：
- `_knowledge_context()` 传 `embedding_provider`（行 1237）— 与现有代码无冲突，只改一行
- `_candidate_matches_topic()` 改用 embedding 相似度（行 1319）— 与现有代码无冲突，改方法实现
- 新增 `_knowledge_query_route()` — 新增方法，无冲突

#### 2. `knowledge_graph/service.py` — 无冲突

37baf5f 已经添加了 `_query_embedding()` 和 `_semantic_similarity()` 方法。我的计划只需要：
- 确保调用方传入 `embedding_provider`（上面已覆盖）
- 新增 BM25 索引方法 — 新增方法，无冲突

#### 3. `knowledge_graph/embedding.py` — 无冲突

37baf5f 和 c26d6e7 已完整实现。不需要修改。

#### 4. `evolution/service.py` — 无冲突

只需修改 `__init__` 传 `embedding_provider`，和新增离线进化循环方法。

#### 5. `memory/store.py` — 需注意

37baf5f 已添加 `entity_embeddings` 表和 `get/upsert` 方法。我的计划需要：
- 新增 BM25 索引表 — 新增，无冲突
- 确认现有表结构兼容

### 3 个"丢失提交"的详细内容

以下提交已在 `agent/tiered-search-budget-observability` 分支中找回：

#### 3a6bfc4 — P1-1 minimal fixes

| 文件 | 改动 |
|------|------|
| `briefing/service.py` | 候选查询隔离：不再拼接 claim terms 到搜索查询 |
| `knowledge_graph/seeds.py` | 双语别名：给 seeded entities 加英文别名 |
| `workflow/service.py` | `_memory_context()` 和 `_experience_context()` 改用 `layered_memory_items` |
| `.gitignore` | 忽略 pytest temp dirs, tmp/, .workbuddy/ |
| `tests/test_briefing_service.py` | 断言不生成"自进化知识补充"搜索方向 |

**与计划的关系**：部分覆盖了 P0 `_candidate_matches_topic` 的症状（不再拼接 claim 到搜索），但**没有修复根本原因**（`_candidate_matches_topic` 仍然过宽匹配）。

#### 37baf5f — Semantic similarity matching

| 文件 | 改动 |
|------|------|
| `knowledge_graph/embedding.py` | 新增 102 行：EmbeddingProvider, NullEmbeddingProvider, OpenAIEmbeddingProvider, build_embedding_provider, cosine_similarity |
| `knowledge_graph/service.py` | +77 行：`_query_embedding()`, `_semantic_similarity()`, semantic boost (cosine >= 0.72 → +3.0) |
| `memory/store.py` | +46 行：`entity_embeddings` 表, `get_entity_embedding()`, `upsert_entity_embedding()` |
| `config.py` | +10 行：`SEARCH_ASSISTANT_EMBEDDING_ENABLED`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_EMBEDDING_MODEL` |
| `.env.example` | +7 行：embedding 相关环境变量 |
| `tests/test_knowledge_graph_embedding.py` | 新增 135 行：cosine similarity, provider builder, semantic cross-lingual hits |

**与计划的关系**：**覆盖了 P0 embedding 基础设施的 80%**。但关键遗漏：`build_embedding_provider()` 从 `os.environ` 读取 key，而 `.env.local` 的值不在 `os.environ` 中（`Settings.from_env()` 加载 `.env.local` 到 Settings 但不写入 `os.environ`），导致 `NullEmbeddingProvider` 始终返回。

#### c26d6e7 — DeepSeek fallback

| 文件 | 改动 |
|------|------|
| `knowledge_graph/embedding.py` | `build_embedding_provider()` 增加 `DEEPSEEK_API_KEY`/`DEEPSEEK_BASE_URL` 作为 fallback |
| `config.py` | `Settings.openai_api_key` fallback 到 `DEEPSEEK_API_KEY`，`openai_base_url` fallback 到 `DEEPSEEK_BASE_URL` |
| `.env.example` | +6 行：DeepSeek fallback 说明 |
| `docs/configuration.md` | +11 行：DeepSeek fallback 文档 |
| `tests/test_knowledge_graph_embedding.py` | +17 行：DeepSeek fallback 测试 |

**与计划的关系**：**解决了 DeepSeek API key 可用性问题**，但 `build_embedding_provider()` 仍然从 `os.environ` 读取，而不是从 `Settings` 读取。`.env.local` 中的 `DEEPSEEK_API_KEY` 不会被 `os.environ` 读取。

---

## 更新后的改进计划（标注所有分支覆盖情况）

> **2026-08-17 更新：P0 全部完成、P1 全部完成，已在分支 `agent/kg-evolution-p0-p1`（commit d20e9cf）实现并推送，PR #1 已创建。**
> P0/P1 实现细节：`embedding_provider_from_settings()` 接线 + 熔断器；`_candidate_matches_topic` 语义匹配 + 收紧字面匹配；BM25 混合检索；轨迹实体/关系自动提取（extractor）；episodic/procedural 记忆路由。

### 阶段 1 — 修复基础（P0）

#### 1.1 修复 Embedding Provider 接线 ✅ 已完成（d20e9cf）

**现状**：
- `embedding.py` + `knowledge_graph/service.py` 中的 embedding 逻辑已完整实现（37baf5f, c26d6e7）
- `build_embedding_provider()` 已支持 DeepSeek fallback（c26d6e7）
- **但**：`briefing/service.py:1237` 仍创建 `DomainKnowledgeGraphService(self.store)` 不传 `embedding_provider`
- **但**：`evolution/service.py:66` 仍创建 `DomainKnowledgeGraphService(store)` 不传 `embedding_provider`
- **但**：`build_embedding_provider()` 从 `os.environ` 读取 key，而 `.env.local` 的值不在 `os.environ` 中

**修复方案**：
1. `_knowledge_context()` 和 `DomainKnowledgeCandidateService.__init__` 传入 `embedding_provider`
2. `build_embedding_provider()` 接受 `Settings` 对象，或从 `Settings` 读取 key 传入

**冲突风险**：🟢 无冲突。只改调用方，不改 `embedding.py` 或 `knowledge_graph/service.py`。

**✅ 实施结果（d20e9cf）**：新增 `embedding_provider_from_settings()`，CLI `_briefing_service`/`_candidate_service`/`_graph_service` 从 Settings 构建 provider 并传入；`OpenAIEmbeddingProvider` 加熔断器（DeepSeek 无 embedding 端点，首次失败后短路）。验证：`build_embedding_provider()` 从 Settings 解析正确（不再是 Null），DeepSeek 404 时 0.81s→0.0000s 降级。

#### 1.2 修复 _candidate_matches_topic 过宽 ✅ 已完成（d20e9cf）

**现状**：
- 3a6bfc4 修复了**症状**：`_knowledge_guided_queries()` 不再拼接 claim terms 到搜索查询
- **但**：`_candidate_matches_topic()` 仍然拼接 topic+claim+applies_when 做宽泛匹配
- 结果：`_knowledge_context()` 仍会把所有 39 个候选都放进 `validated_candidates`/`weak_signals`

**修复方案**：
- 方案 A：用 embedding 相似度做主题匹配（cosine >= 0.78）
- 方案 B：限制只匹配 topic 字段，且要求至少 2 个 term 重叠

**冲突风险**：🟢 无冲突。只改 `_candidate_matches_topic()` 方法实现。

**✅ 实施结果（d20e9cf）**：`_candidate_matches_topic` 改为实例方法，语义相似度（cosine≥0.78）优先，字面回退只检查 topic 字段且 ≥2 term 重叠。

#### 1.3 验证基础功能 ✅ 已完成（d20e9cf）

**修复方案**：确认 DeepSeek embedding API 可用，`entity_embeddings` 表写入正常

**冲突风险**：🟢 无冲突。纯验证步骤。

**✅ 实施结果（d20e9cf）**：确认 DeepSeek API **只有 chat 模型**（deepseek-v4-flash/pro），无 embedding 端点（404）→ 熔断降级为字面匹配，graph query 0.77s 正常返回。`entity_embeddings` 缓存方法正常。

### 阶段 2 — 检索增强（P1）

#### 4.1 混合检索 ✅ 已完成（d20e9cf）

**现状**：所有分支只有 dense embedding（37baf5f），无 BM25/RRF/Reranking

**修复方案**：
- 新增 `knowledge_graph/bm25.py`：BM25 索引和查询
- 新增 `knowledge_graph/retriever.py`：混合检索 + RRF fusion + reranking
- 修改 `knowledge_graph/service.py`：`query()` 改用混合检索

**冲突风险**：🟡 低风险。`knowledge_graph/service.py` 的 `query()` 方法需要修改，但 37baf5f 的 embedding 逻辑是独立方法，不影响。

**✅ 实施结果（d20e9cf）**：新增 `knowledge_graph/bm25.py`（BM25Index + rrf_rank，纯标准库），`query()` 在字面/语义评分之上叠加稀疏加分（cap 2.0），图谱级懒加载索引缓存。5 个测试通过。

#### 4.2 图谱动态扩展 ✅ 已完成（d20e9cf）

**现状**：所有分支只有 `seeds.py` 硬编码种子图谱

**修复方案**：
- 新增 `knowledge_graph/extractor.py`：从日报轨迹自动提取实体/关系
- 修改 `evolution/service.py`：`capture_briefing()` 后自动提取实体

**冲突风险**：🟢 无冲突。新增文件和新增方法。

**✅ 实施结果（d20e9cf）**：新增 `knowledge_graph/extractor.py`（单词 + 英文 bigram 词频提取实体、共现关系），写入 `auto-<slug>` 图谱（不动种子图谱），briefing 完成后自动调用（`extend_graph_from_briefing`）。5 个测试通过。

**⚠️ 后续修复（跨主题合并 bug）**：原实现合并时只找第一个 `auto-` 图谱（`graph_id` 参数被忽略），导致不同主题的自动提取实体混入同一张图（实测 133 实体三题混杂）。已修复：目标图谱改为按 `graph_id`/主题 slug 定位（`auto-<slug>`），同主题多轮合入同一张图、不同主题各自成图；存量污染数据已按实体 evidence 的 briefing 归属拆分还原（50/45/38）。

**⚠️ 后续修复（实体质量）**：自动提取实体大量是 CJK 切词碎片（`智能/能体`、`大模 模型`、`端视 视觉`）与泛化词（`ai`、`模型`、`部署`），经 `query_relevant` 反向匹配造成跨主题误命中并生成低质量"知识图谱补充"查询。已修复三层：① `extract_entities` 改为整段 CJK 词组（2-10 字 run）计数、丢弃窗口 bigram 碎片、丢弃纯 CJK-CJK 对；② 新增 `is_meaningful_entity_name`（拒绝单字/泛化词/2 字碎片/双侧 2 字 CJK 对），`query()` 检索时按同图兄弟实体过滤；③ `_knowledge_guided_queries` 只消费通过该过滤的实体名。存量 auto 图谱已清理（两轮共移除 228 个碎片/泛化实体），检索命中从 `能体/大模 模型` 变为 `a2a 协议/mes/受控工单编排/graphrag` 级别。

#### 4.3 多类型记忆查询路由 ✅ 已完成（d20e9cf）

**现状**：
- ✅ `layered_memory_items` 表 + `add_layered_memory_item` 写入（7e01cbf）
- ✅ `workflow/service.py` 的 `_memory_context()` 和 `_experience_context()` 改用 layered_memory_items（3a6bfc4）
- ❌ `_knowledge_context()` 仍不按 memory_type 分路检索

**修复方案**：
- 修改 `_knowledge_context()`：按 memory_type 分路检索（episodic → trajectory_logs, semantic → graph entities, procedural → evolution rules）

**冲突风险**：🟢 无冲突。只改 `_knowledge_context()` 方法。

**✅ 实施结果（d20e9cf）**：`_knowledge_context()` 新增 `episodic_trajectories`（提及主题的最近轨迹日志）和 `procedural_rules`（active 的 run_experience/domain_knowledge 分层记忆项），metadata 增加对应计数。4 个测试通过。

### 阶段 3 — 进化闭环（P2）

#### 5.1 在线/离线双循环 ✅ 已完成

**现状**：
- ✅ 在线记录：ProviderTraceEvent, SourceCandidate, BriefingCollectionTrace, TopicFeedbackSignal
- ✅ 离线进化循环：新增 `evolution/offline_loop.py`（聚合-三层验证-诊断-候选蒸馏-发布/回滚监控），CLI 命令 `evolution-offline-run`（`--judge rule|llm`、`--max-trajectories`）

**实施结果**：每次运行聚合尚无评估的不可变轨迹 → 三层验证 → 诊断 → 持久化 trajectory_evaluation → 失败写 experience item → 重跑候选验证门（蒸馏）→ 检查 eval 报告发布门 → 输出 `evolution/offline-evolution-report.json` + gate 记录 + ledger 条目。幂等：已评估轨迹不重复验证。

#### 5.2 轨迹验证三层评估 ✅ 已完成

**现状**：
- ✅ `EvolutionDiagnosisService.diagnose()` 有 result/process/quality 三层 flag 分类
- ✅ 新增 `evolution/verification.py`：结果层与过程层为纯代码检查（答案存在、搜索执行、来源返回、必检搜索、校准/审查纪律、引用溯源、不确定性披露），质量层为 Rubric 判定。默认 `RuleQualityJudge`（确定性、离线），可选 `LLMRubricJudge`（LLM 评分，含结构化 JSON 解析、失败/解析失败时整体弃权为 uncertain，不编造结论）；`calibrate_quality_judge()` 提供按维度的精确率/召回率校准报告。

**实施结果**：证据位置化（answer 摘录、来源 URL、review 状态），维度含事实可靠性、引用忠实度、承诺-行动一致性、表达质量，事实可靠性与引用忠实度为一票否决维度。质量层 uncertain 仅上报不判失败。

#### 5.3 跨轨迹知识蒸馏 ✅ 已完成

**现状**：
- ✅ `capture_briefing` 改用 `store.upsert_domain_knowledge_candidate`：同指纹候选按 URL 合并 evidence、合并 source_ids/contradictions、置信度只升不降，并记录 merge 事件（原来 `INSERT OR IGNORE` 会静默丢弃第二次出现的证据）
- ✅ **验证门门槛调整为 ≥1 条轨迹**（`_MIN_INDEPENDENT_TRAJECTORIES = 1`）：单条轨迹 + 双独立域 + 权威度达标 + 非低置信即可通过自动门
- ✅ **语义质量门**（`evolution/semantics.py`）：LLM 主判（注入式 `llm_runner`，与 judge 同一约定）+ 规则回退，区分 technical / marketing / news / vague；营销话术与模糊空话判为 `marketing_or_vague_content` 失败 → `rejected_noise`，即使轨迹门槛放宽也不会升级
- ✅ **用户确认门**：`distill_candidates()` 通过全部验证门的候选进入 `pending_user_confirm` 状态（不再自动 `validated`）；用户在聊天中发送"查看待确认候选"查看、"确认知识候选：<id>"确认后才置为 `validated`（写 human_review gate + knowledge_release ledger）。`validate()`/`approve()` 人工命令路径保持直接提升不变
- ✅ **topic 语义合并**：`_find_similar_candidate` 的字面 `_topic_similar` 之外增加 embedding 语义通道（`_topic_semantically_similar`，cosine ≥ 0.75），"AI Agent Harness 上下文工程 工具调用可靠性" 与 "上下文工程" 这类措辞漂移可合并；embedding 不可用/熔断时自动回退字面匹配

#### 5.4 发布/回滚/灰度自动化 ⚠️ 监控完成（发布仍人工把关）

**实施结果**：离线循环增加发布监控（读取最新 eval 报告，存在阻断项则报告失败）与回滚监控（发现已 validated 但独立轨迹数 < 1 的候选，仅列出建议、不自动废弃）。**候选提升已自动化但入库需用户确认**：`distill_candidates()` 对通过全部验证门（≥1 独立轨迹、≥2 独立域、权威度达标、无矛盾、非低置信、非陈旧、非营销/模糊内容）的候选置为 `pending_user_confirm`，用户在聊天确认后才成为 `validated_knowledge`。自动发布仍不启用——`validated_knowledge` 进入正式知识上下文前须用户确认；回滚仍走 `knowledge-candidate-deprecate`。

---

## 最终待做清单

| 优先级 | 项目 | 状态 | 分支覆盖情况 | 冲突风险 |
|--------|------|------|-------------|---------|
| P0 | Embedding Provider 接线修复 | ✅ 已完成（d20e9cf） | ⚠️ 基础设施已存在，接线+熔断已补齐 | 🟢 无 |
| P0 | _candidate_matches_topic 过宽修复 | ✅ 已完成（d20e9cf） | ⚠️ 语义匹配 + 收紧字面匹配 | 🟢 无 |
| P0 | 验证基础功能 | ✅ 已完成（d20e9cf） | 确认 DeepSeek 无 embedding 端点 → 熔断降级 | 🟢 无 |
| P1 | 混合检索 (Dense+Sparse+RRF+Reranking) | ✅ 已完成（d20e9cf） | BM25 + 稀疏加分（cap 2.0） | 🟡 低 |
| P1 | 图谱动态扩展 | ✅ 已完成（d20e9cf） | extractor + auto-<slug> 图谱 | 🟢 无 |
| P1 | 多类型记忆查询路由 | ✅ 已完成（d20e9cf） | episodic_trajectories + procedural_rules | 🟢 无 |
| P2 | 离线进化循环 | ✅ 已完成（evolution/offline_loop.py + evolution-offline-run） | ⚠️ 在线记录已有 | 🟢 无 |
| P2 | 轨迹验证三层评估 | ✅ 已完成（evolution/verification.py：结果/过程纯代码 + Rubric 质量层，规则/LLM 双 judge + 校准） | ⚠️ 诊断 flag 分类已有，LLM 验证未做 | 🟡 低 |
| P2 | 跨轨迹知识蒸馏 | ✅ 已完成（候选合并 + ≥2 独立轨迹门 + distill_candidates） | ✅ | 🟢 无 |
| P2 | 发布/回滚/灰度自动化 | ⚠️ 监控完成（发布/回滚监控 + eval 门），自动发布/回滚按设计不启用 | ✅ | 🟢 无 |

---

## 风险与注意事项

1. **本地仓库已修复**：之前的 git 损坏已通过从 fork bare 仓库恢复解决。当前 `pr10-work` 分支指向 `c26d6e7`（agent/tiered-search-budget-observability），工作区干净。

2. **3 个"丢失提交"已找回**：3a6bfc4, 37baf5f, c26d6e7 在 `agent/tiered-search-budget-observability` 分支中完整存在，已恢复到 pr10-work。

3. **Embedding Provider 接线已修复**：`embedding_provider_from_settings()` 从 Settings 读取 key 并显式传入（P0-1，d20e9cf）。

4. **DeepSeek API 无 embedding 端点**：只有 chat 模型（deepseek-v4-flash/pro）。`OpenAIEmbeddingProvider` 熔断器会在首次失败后短路（P0-1），语义匹配优雅降级为字面匹配。若需语义检索，需配置支持 embedding 的 OpenAI 兼容端点（如硅基流动等）。

5. **向后兼容**：新增字段应通过 metadata 扩展而非修改顶层结构。

6. **所有分支的改动都是互补的**：没有任何分支与改进计划有实质性代码冲突。所有已完成项都是基础设施，改进计划在其上构建。

7. **P0/P1 已完成并提交 PR**：分支 `agent/kg-evolution-p0-p1`（commit d20e9cf），PR #1：https://github.com/xiangkaiyi815-pixel/pydantic-ai-tech-intel-briefing/pull/1

8. **P2 待做**：离线进化循环（聚合-诊断-验证-发布）、轨迹验证三层评估（LLM claim-source 一致性 + query 覆盖度 + 质量 Rubric）、跨轨迹知识蒸馏（≥2 条独立轨迹支持）、发布/回滚/灰度自动化。
