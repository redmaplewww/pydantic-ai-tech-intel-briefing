import json
from pathlib import Path
import time

import pytest

from search_assistant.contracts import AnswerPackage, IncomingMessage
from search_assistant.memory.store import MemoryStore
from search_assistant.search.provider import SearchResult
from search_assistant.skills.service import SkillDraftService
from search_assistant.runtime import FakeAgentRuntime
from search_assistant.workflow.grounding import (
    build_question_profile,
    classify_question_profile,
    decide_grounding_policy,
)
from search_assistant.workflow.service import SearchAssistantWorkflow


@pytest.mark.parametrize(
    ("question", "expected_category", "expected_classification", "expected_claim_type"),
    [
        ("能否解释向量数据库", "foundational", "research", "concept"),
        ("Can you explain cache coherence?", "foundational", "research", "concept"),
        ("Redis 最新版本是什么", "evidence_required", "research", "current_fact"),
        ("某型号具体带宽是多少", "evidence_required", "hard", "precise_number"),
        ("4 张某加速卡能否部署某模型", "evidence_required", "hard", "deployment_feasibility"),
    ],
)
def test_question_profile_drives_classification_and_grounding_policy(
    question,
    expected_category,
    expected_classification,
    expected_claim_type,
):
    profile = build_question_profile(question)
    classification = classify_question_profile(profile)
    decision = decide_grounding_policy(profile, classification)

    assert classification == expected_classification
    assert decision.category == expected_category
    assert expected_claim_type in profile.requested_claim_types
    assert decision.allow_stable_model_knowledge is (expected_category == "foundational")
    assert decision.require_retrieval_sources is (expected_category != "foundational")
    assert decision.precise_numbers_require_sources is True
    assert decision.vendor_version_claims_require_sources is True


def test_broad_can_or_nengfou_does_not_make_explanation_a_deployment_question():
    chinese_profile = build_question_profile("能否解释向量数据库")
    english_profile = build_question_profile("Can you explain cache coherence?")

    assert "deployment_feasibility" not in chinese_profile.requested_claim_types
    assert "deployment_feasibility" not in english_profile.requested_claim_types


def test_hard_question_runs_calibration_and_persists_answer(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(answer_text="The answer depends on verified API behavior in 2026.")
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime)
    message = IncomingMessage(
        message_id="m-1",
        event_id="e-1",
        user_id="u-1",
        chat_id="c-1",
        text="Compare the newest Agent Framework workflow API with Semantic Kernel and give migration risks.",
        source="cli",
    )

    package = workflow.answer(message)

    assert package.classification == "hard"
    assert package.calibration["ran"] is True
    assert runtime.calibration_calls == 1
    assert store.latest_answer_for_dedupe_key("e-1").question_id == package.question_id


def test_duplicate_event_returns_stored_answer(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(answer_text="First answer.")
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime)
    message = IncomingMessage(
        message_id="m-1",
        event_id="e-1",
        user_id="u-1",
        chat_id="c-1",
        text="simple note",
        source="cli",
    )

    first = workflow.answer(message)
    second = workflow.answer(message)

    assert second.question_id == first.question_id
    assert runtime.answer_calls == 1


def test_chinese_learning_direction_question_is_hard_and_updates_topics(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(answer_text="需要围绕飞书 bot 和 Microsoft Agent Framework 建立学习路线。")
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime)
    message = IncomingMessage(
        message_id="m-cn-1",
        event_id="e-cn-1",
        user_id="u-1",
        chat_id="c-1",
        text="我想系统学习 Microsoft Agent Framework 和飞书 bot 集成，请根据我的问题判断学习方向。",
        source="cli",
    )

    package = workflow.answer(message)
    topics = [item["content"] if isinstance(item, dict) else item.content for item in package.memory_updates]

    assert package.classification == "hard"
    assert package.calibration["ran"] is True
    assert "Feishu integration" in topics
    assert "Microsoft Agent Framework" in topics
    assert "Learning direction planning" in topics


def test_chinese_api_compare_question_is_hard(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    workflow = SearchAssistantWorkflow(store=store, runtime=FakeAgentRuntime())
    message = IncomingMessage(
        message_id="m-cn-2",
        event_id="e-cn-2",
        user_id="u-1",
        chat_id="c-1",
        text="请比较飞书机器人事件订阅和消息回复 API 的风险，并告诉我下一步该学什么。",
        source="cli",
    )

    package = workflow.answer(message)

    assert package.classification == "hard"
    assert package.calibration["ran"] is True


def test_hard_question_searches_before_answer_and_passes_sources_to_calibration(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = RecordingRuntime()
    search = RecordingSearchClient(
        [
            SearchResult(
                title="Feishu Event API",
                url="https://open.feishu.cn/document/event",
                snippet="Receive message events and reply with bot APIs.",
                provider="unit",
                checked_at="2026-06-27T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-search-1",
        event_id="e-search-1",
        user_id="u-1",
        chat_id="c-1",
        text="Compare current Feishu bot event APIs and Agent Framework workflow risks.",
        source="cli",
    )

    package = workflow.answer(message)

    assert search.queries == ["Compare current Feishu bot event APIs and Agent Framework workflow risks."]
    assert runtime.answer_contexts[0]["search_results"][0]["title"] == "Feishu Event API"
    assert runtime.calibration_contexts[0]["search_results"][0]["url"] == "https://open.feishu.cn/document/event"
    assert package.sources[0].url == "https://open.feishu.cn/document/event"
    assert package.answer_text.startswith("calibrated answer with citations")
    assert "搜索记录:" in package.answer_text


def test_review_agent_runs_after_calibration_and_can_revise_final_answer(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = ReviewingRuntime(review_revision="reviewed final answer with source discipline")
    search = RecordingSearchClient(
        [
            SearchResult(
                title="Feishu Event API",
                url="https://open.feishu.cn/document/event",
                snippet="Receive message events and reply with bot APIs.",
                provider="unit",
                checked_at="2026-06-27T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-review-1",
        event_id="e-review-1",
        user_id="u-1",
        chat_id="c-1",
        text="Compare current Feishu bot event APIs and Agent Framework workflow risks.",
        source="cli",
    )

    package = workflow.answer(message)

    assert package.calibration["ran"] is True
    assert package.review["ran"] is True
    assert package.review["approved"] is True
    assert runtime.review_contexts[0]["answer"] == "calibrated answer with citations"
    assert runtime.review_contexts[0]["search_results"][0]["url"] == "https://open.feishu.cn/document/event"
    assert package.answer_text.startswith("reviewed final answer with source discipline")
    assert store.latest_answer_for_dedupe_key("e-review-1").review["ran"] is True


def test_review_rejection_blocks_unsafe_final_answer(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = RejectedReviewRuntime()
    search = RecordingSearchClient(
        [
            SearchResult(
                title="AI physical world models overview",
                url="https://example.com/world-models",
                snippet="World models and embodied physical AI are discussed at a high level.",
                provider="unit",
                checked_at="2026-06-29T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-review-reject-1",
        event_id="e-review-reject-1",
        user_id="u-1",
        chat_id="c-1",
        text="As of 2026, where are AI physical world models?",
        source="cli",
    )

    package = workflow.answer(message)

    assert package.review["approved"] is False
    assert "最终审查未通过" in package.answer_text
    assert "unsupported model-memory claims" in package.answer_text
    assert "unsafe unsupported original answer" not in package.answer_text
    assert package.confidence == "low"


def test_simple_question_still_searches_before_answer(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = RecordingRuntime()
    search = RecordingSearchClient(
        [
            SearchResult(
                title="Greeting Reference",
                url="https://example.com/greeting",
                snippet="A greeting is a short friendly opening.",
                provider="unit",
                checked_at="2026-06-28T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-simple-search",
        event_id="e-simple-search",
        user_id="u-1",
        chat_id="c-1",
        text="hello",
        source="cli",
    )

    package = workflow.answer(message)

    assert package.classification == "simple"
    assert search.queries == ["hello"]
    assert runtime.answer_contexts[0]["search_results"][0]["url"] == "https://example.com/greeting"
    assert package.sources[0].url == "https://example.com/greeting"


def test_answer_text_appends_search_record_with_queries_and_sources(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(answer_text="Use the searched source.")
    search = RecordingSearchClient(
        [
            SearchResult(
                title="Source One",
                url="https://example.com/source-one",
                snippet="First source snippet.",
                provider="unit",
                checked_at="2026-06-28T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-search-record-1",
        event_id="e-search-record-1",
        user_id="u-1",
        chat_id="c-1",
        text="What is this source?",
        source="cli",
    )

    package = workflow.answer(message)

    assert "Use the searched source." in package.answer_text
    assert "搜索记录:" in package.answer_text
    assert "联网搜索: 已执行" in package.answer_text
    assert "this source" in package.answer_text
    assert "Source One" in package.answer_text
    assert "https://example.com/source-one" in package.answer_text
    assert package.search_record is not None
    assert package.search_record.executed is True
    assert package.search_record.queries == search.queries
    assert package.search_record.sources[0].url == "https://example.com/source-one"
    stored = store.latest_answer_for_dedupe_key("e-search-record-1")
    assert stored is not None
    assert stored.search_record is not None
    assert stored.search_record.sources[0].provider == "unit"
    assert "摘要:" not in package.answer_text
    assert "First source snippet." not in package.answer_text


def test_answer_text_appends_search_engine_names_when_available(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(answer_text="Use the searched source.")
    search = RecordingSearchClient([])
    search.engine_names = ["bing", "baidu", "google"]
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-search-record-engines",
        event_id="e-search-record-engines",
        user_id="u-1",
        chat_id="c-1",
        text="What is this source?",
        source="cli",
    )

    package = workflow.answer(message)

    assert "搜索引擎: bing, baidu, google" in package.answer_text


def test_answer_text_appends_search_record_when_no_results(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(answer_text="No public result was useful.")
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-search-record-empty",
        event_id="e-search-record-empty",
        user_id="u-1",
        chat_id="c-1",
        text="hello",
        source="cli",
    )

    package = workflow.answer(message)

    assert "搜索记录:" in package.answer_text
    assert "联网搜索: 已执行" in package.answer_text
    assert "搜索结果: 未返回可用结果" in package.answer_text
    assert "hello" in package.answer_text


def test_irrelevant_search_results_block_model_memory_answer(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(answer_text="unsupported detailed world-model industry report from memory")
    search = RecordingSearchClient(
        [
            SearchResult(
                title="2026 calendar",
                url="https://example.com/calendar",
                snippet="A calendar page for the year 2026.",
                provider="unit",
                checked_at="2026-06-29T00:00:00Z",
            ),
            SearchResult(
                title="physical dictionary",
                url="https://example.com/dictionary",
                snippet="Dictionary definition for the word physical.",
                provider="unit",
                checked_at="2026-06-29T00:00:00Z",
            ),
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-irrelevant-results-1",
        event_id="e-irrelevant-results-1",
        user_id="u-1",
        chat_id="c-1",
        text="As of 2026, where are AI physical world models and embodied physical AI?",
        source="cli",
    )

    package = workflow.answer(message)

    assert "搜索结果相关性不足" in package.answer_text
    assert "unsupported detailed world-model industry report" not in package.answer_text
    assert runtime.answer_calls == 0
    assert package.review["approved"] is False
    assert package.confidence == "low"


def test_foundational_distributed_model_question_answers_with_caveat_when_search_is_weak(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = RecordingRuntime()
    search = RecordingSearchClient(
        [
            SearchResult(
                title="2026 conference travel calendar",
                url="https://example.com/calendar",
                snippet="A calendar page unrelated to distributed inference.",
                provider="unit",
                checked_at="2026-07-02T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-foundational-distributed-llm-1",
        event_id="e-foundational-distributed-llm-1",
        user_id="u-1",
        chat_id="c-1",
        text="分布式大模型是否可以理解为多个相对独立的节点，每个节点负责一部分推理工作，并且不同节点间互联互通",
        source="cli",
    )

    package = workflow.answer(message)

    assert runtime.answer_contexts
    assert package.classification == "hard"
    assert package.review["approved"] is True
    assert package.confidence == "low"
    assert "搜索结果相关性不足" not in package.answer_text
    assert "calibrated answer with citations" in package.answer_text
    assert runtime.answer_contexts[0]["source_relevance_issue"].startswith("搜索结果相关性不足")
    assert runtime.answer_contexts[0]["allow_foundational_fallback"] is True
    assert runtime.calibration_contexts[0]["allow_foundational_fallback"] is True
    assert runtime.review_contexts[0]["allow_foundational_fallback"] is True


def test_foundational_distributed_model_question_uses_fallback_when_search_returns_no_results(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = RecordingRuntime()
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-foundational-distributed-llm-empty-search-1",
        event_id="e-foundational-distributed-llm-empty-search-1",
        user_id="u-1",
        chat_id="c-1",
        text="分布式大模型是否可以理解为多个相对独立的节点，每个节点负责一部分推理工作，并且不同节点间互联互通",
        source="cli",
    )

    package = workflow.answer(message)

    assert runtime.answer_contexts
    assert package.classification == "hard"
    assert package.review["approved"] is True
    assert package.confidence == "low"
    assert "我已阻断本轮生成" not in package.answer_text
    assert runtime.answer_contexts[0]["source_relevance_issue"].startswith("搜索未返回可用结果")
    assert runtime.answer_contexts[0]["allow_foundational_fallback"] is True
    assert runtime.calibration_contexts[0]["allow_foundational_fallback"] is True
    assert runtime.review_contexts[0]["allow_foundational_fallback"] is True


def test_foundational_distributed_model_question_accepts_english_parallelism_sources(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = RecordingRuntime()
    search = RecordingSearchClient(
        [
            SearchResult(
                title="NVIDIA Megatron Bridge official parallelisms documentation",
                url="https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html",
                snippet=(
                    "Tensor parallelism, pipeline parallelism, and expert parallelism "
                    "are distributed strategies for large model inference and training. "
                    "Expert parallelism uses all-to-all token dispatching for MoE models."
                ),
                provider="direct-official",
                checked_at="2026-07-03T00:00:00Z",
            ),
            SearchResult(
                title="vLLM official parallelism and scaling documentation",
                url="https://docs.vllm.ai/en/stable/serving/parallelism_scaling/",
                snippet=(
                    "Distributed inference can use tensor parallelism on a node, or combine "
                    "tensor parallelism and pipeline parallelism across multiple nodes."
                ),
                provider="direct-official",
                checked_at="2026-07-03T00:00:00Z",
            ),
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-foundational-distributed-llm-english-sources-1",
        event_id="e-foundational-distributed-llm-english-sources-1",
        user_id="u-1",
        chat_id="c-1",
        text="分布式大模型是否可以理解为多个相对独立的节点，每个节点负责一部分推理工作，并且不同节点间互联互通",
        source="cli",
    )

    package = workflow.answer(message)

    assert runtime.answer_contexts
    assert "source_relevance_issue" not in runtime.answer_contexts[0]
    assert "allow_foundational_fallback" not in runtime.answer_contexts[0]
    assert "搜索结果相关性不足" not in package.unverified_claims
    assert "搜索结果相关性不足" not in package.answer_text


def test_foundational_distributed_model_question_still_answers_when_review_rejects_weak_sources(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = RejectedReviewRuntime()
    search = RecordingSearchClient(
        [
            SearchResult(
                title="2026 conference travel calendar",
                url="https://example.com/calendar",
                snippet="A calendar page unrelated to distributed inference.",
                provider="unit",
                checked_at="2026-07-02T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-foundational-distributed-llm-review-reject-1",
        event_id="e-foundational-distributed-llm-review-reject-1",
        user_id="u-1",
        chat_id="c-1",
        text="分布式大模型是否可以理解为多个相对独立的节点，每个节点负责一部分推理工作，并且不同节点间互联互通",
        source="cli",
    )

    package = workflow.answer(message)

    assert runtime.answer_contexts
    assert package.review["approved"] is False
    assert package.confidence == "low"
    assert "最终审查未通过" not in package.answer_text
    assert "unsafe unsupported original answer" not in package.answer_text
    assert "可以这么理解" in package.answer_text
    assert "节点" in package.answer_text
    assert "推理" in package.answer_text
    assert "互联" in package.answer_text
    assert "搜索记录:" in package.answer_text


def test_foundational_distributed_model_question_uses_policy_fallback_for_refusal_revision(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = ReviewingRuntime(review_revision="搜索结果相关性不足：当前来源不足，不能回答。")
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-foundational-distributed-llm-refusal-revision-1",
        event_id="e-foundational-distributed-llm-refusal-revision-1",
        user_id="u-1",
        chat_id="c-1",
        text="分布式大模型是否可以理解为多个相对独立的节点，每个节点负责一部分推理工作，并且不同节点间互联互通",
        source="cli",
    )

    package = workflow.answer(message)

    assert package.review["approved"] is True
    assert package.confidence == "low"
    assert "当前来源不足，不能回答" not in package.answer_text
    assert "可以这么理解" in package.answer_text
    assert "节点" in package.answer_text
    assert "互联" in package.answer_text


def test_foundational_distributed_model_question_uses_policy_fallback_even_with_relevant_sources(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = ReviewingRuntime(review_revision="搜索内容不足：当前来源不足，不能回答。")
    search = RecordingSearchClient(
        [
            SearchResult(
                title="分布式大模型推理并行概述",
                url="https://example.com/distributed-llm-inference",
                snippet=(
                    "分布式大模型推理可以由多个节点分担模型计算，"
                    "节点之间通过互联通信同步中间结果。"
                ),
                provider="unit",
                checked_at="2026-07-02T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-foundational-distributed-llm-relevant-source-refusal-1",
        event_id="e-foundational-distributed-llm-relevant-source-refusal-1",
        user_id="u-1",
        chat_id="c-1",
        text="分布式大模型是否可以理解为多个相对独立的节点，每个节点负责一部分推理工作，并且不同节点间互联互通",
        source="cli",
    )

    package = workflow.answer(message)

    assert "source_relevance_issue" not in runtime.answer_contexts[0]
    assert "stable concept" in runtime.answer_contexts[0]["foundational_answer_policy"]
    assert runtime.calibration_contexts[0]["foundational_answer_policy"] == runtime.answer_contexts[0]["foundational_answer_policy"]
    assert runtime.review_contexts[0]["foundational_answer_policy"] == runtime.answer_contexts[0]["foundational_answer_policy"]
    assert package.review["approved"] is True
    assert package.confidence == "low"
    assert "当前来源不足，不能回答" not in package.answer_text
    assert "可以这么理解" in package.answer_text
    assert "稳定概念" in package.answer_text
    assert "搜索记录:" in package.answer_text


def test_foundational_question_uses_policy_fallback_for_retrieval_insufficiency_refusal(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = ReviewingRuntime(review_revision="根据当前检索结果，无法确认这个理解是否成立，建议补充更相关来源后再判断。")
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-foundational-distributed-llm-retrieval-refusal-1",
        event_id="e-foundational-distributed-llm-retrieval-refusal-1",
        user_id="u-1",
        chat_id="c-1",
        text="分布式大模型是否可以理解为多个相对独立的节点，每个节点负责一部分推理工作，并且不同节点间互联互通",
        source="cli",
    )

    package = workflow.answer(message)

    assert package.confidence == "low"
    assert "无法确认这个理解是否成立" not in package.answer_text
    assert "可以这么理解" in package.answer_text
    assert "节点" in package.answer_text
    assert "推理" in package.answer_text
    assert "互联" in package.answer_text
    assert "搜索记录:" in package.answer_text


def test_foundational_question_uses_policy_fallback_for_source_stall_without_explicit_refusal(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = ReviewingRuntime(review_revision="搜索内容不足：建议补充更相关来源后再判断。")
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-foundational-distributed-llm-source-stall-1",
        event_id="e-foundational-distributed-llm-source-stall-1",
        user_id="u-1",
        chat_id="c-1",
        text="分布式大模型是否可以理解为多个相对独立的节点，每个节点负责一部分推理工作，并且不同节点间互联互通",
        source="cli",
    )

    package = workflow.answer(message)

    assert package.confidence == "low"
    assert "搜索内容不足：建议补充更相关来源后再判断" not in package.answer_text
    assert "可以这么理解" in package.answer_text
    assert "节点" in package.answer_text
    assert "推理" in package.answer_text
    assert "互联" in package.answer_text
    assert "搜索记录:" in package.answer_text


def test_foundational_question_uses_policy_fallback_even_when_revision_mentions_concepts(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = ReviewingRuntime(
        review_revision=(
            "搜索内容不足，所以不建议直接判断。"
            "分布式大模型涉及多个节点、推理分工和互联通信，但建议补充更相关来源后再回答。"
        )
    )
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-foundational-distributed-llm-soft-stall-1",
        event_id="e-foundational-distributed-llm-soft-stall-1",
        user_id="u-1",
        chat_id="c-1",
        text="分布式大模型是否可以理解为多个相对独立的节点，每个节点负责一部分推理工作，并且不同节点间互联互通",
        source="cli",
    )

    package = workflow.answer(message)

    assert package.confidence == "low"
    assert "不建议直接判断" not in package.answer_text
    assert "建议补充更相关来源后再回答" not in package.answer_text
    assert "可以这么理解" in package.answer_text
    assert "稳定概念" in package.answer_text
    assert "低置信说明" in package.answer_text
    assert "搜索记录:" in package.answer_text


def test_foundational_question_uses_policy_fallback_for_soft_evidence_deferral_wording(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = ReviewingRuntime(review_revision="现有证据不够充分，暂时不能判断。")
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-foundational-distributed-llm-soft-evidence-deferral-1",
        event_id="e-foundational-distributed-llm-soft-evidence-deferral-1",
        user_id="u-1",
        chat_id="c-1",
        text=(
            "分布式大模型是否可以理解为多个相对独立的节点，"
            "每个节点负责一部分推理工作，并且不同节点间互联互通"
        ),
        source="cli",
    )

    package = workflow.answer(message)

    assert "现有证据不够充分，暂时不能判断" not in package.answer_text
    assert "可以这么理解" in package.answer_text
    assert "稳定概念" in package.answer_text
    assert "低置信说明" in package.answer_text


def test_foundational_concept_question_can_answer_when_search_has_no_sources(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = ReviewingRuntime(review_revision="搜索内容不足：当前来源不足，不能回答。")
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-grounding-foundational-empty-search",
        event_id="e-grounding-foundational-empty-search",
        user_id="u-1",
        chat_id="c-1",
        text="什么是缓存一致性互连？",
        source="cli",
    )

    package = workflow.answer(message)

    assert runtime.answer_contexts
    policy = runtime.answer_contexts[0]["grounding_policy"]
    assert policy["category"] == "foundational"
    assert policy["allow_stable_model_knowledge"] is True
    assert policy["require_retrieval_sources"] is False
    assert policy["allow_precise_numbers"] is False
    assert policy["allow_vendor_version_claims"] is False
    assert package.confidence == "low"
    assert "当前来源不足，不能回答" not in package.answer_text
    assert "基础解释" in package.answer_text
    assert "低置信说明" in package.answer_text


def test_current_fact_question_refuses_definite_answer_without_sources(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(answer_text="FastAPI 9.9.9 is the latest release today.")
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-grounding-current-empty-search",
        event_id="e-grounding-current-empty-search",
        user_id="u-1",
        chat_id="c-1",
        text="What is the latest FastAPI version today?",
        source="cli",
    )

    package = workflow.answer(message)

    assert runtime.answer_calls == 0
    policy = package.trajectory_context["grounding_policy"]
    assert policy["category"] == "evidence_required"
    assert policy["allow_stable_model_knowledge"] is False
    assert policy["require_retrieval_sources"] is True
    assert policy["allow_precise_numbers"] is True
    assert policy["allow_vendor_version_claims"] is True
    assert "9.9.9" not in package.answer_text
    assert "搜索未返回可用结果" in package.answer_text
    assert package.review["approved"] is False
    assert package.confidence == "low"


def test_exact_spec_question_removes_numbers_when_no_source_supports_them(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(answer_text="The accelerator has 512 GB memory and 900 GB/s bandwidth.")
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-grounding-exact-spec-empty-search",
        event_id="e-grounding-exact-spec-empty-search",
        user_id="u-1",
        chat_id="c-1",
        text="What are the exact memory capacity and bandwidth specs of ACME X9000?",
        source="cli",
    )

    package = workflow.answer(message)

    assert runtime.answer_calls == 0
    assert "512 GB" not in package.answer_text
    assert "900 GB/s" not in package.answer_text
    assert "搜索未返回可用结果" in package.answer_text
    assert package.confidence == "low"


def test_deployment_reasoning_separates_known_sources_from_missing_assumptions(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(
        answer_text=(
            "Eight ACME X900 systems can deploy Model-Z9. "
            "The combined memory is enough and expected throughput is 15 tok/s."
        )
    )
    search = RecordingSearchClient(
        [
            SearchResult(
                title="ACME X900 official system specifications",
                url="https://example.com/acme-x900-specs",
                snippet="ACME X900 official specifications list 256 GB unified memory per system.",
                provider="direct-official",
                checked_at="2026-08-22T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-grounding-deployment-missing-model-evidence",
        event_id="e-grounding-deployment-missing-model-evidence",
        user_id="u-1",
        chat_id="c-1",
        text="Can 8 ACME X900 systems deploy Model-Z9? Include calculation assumptions and limits.",
        source="cli",
    )

    package = workflow.answer(message)

    assert "can deploy Model-Z9" not in package.answer_text
    assert "15 tok/s" not in package.answer_text
    assert "cannot confirm" in package.answer_text
    assert "ACME X900 official system specifications" in package.answer_text
    assert "model parameters" in package.answer_text
    assert "benchmark" in package.answer_text


def test_empty_search_treats_foundational_and_evidence_required_questions_differently(tmp_path):
    foundational_store = MemoryStore(tmp_path / "foundational.sqlite3")
    foundational_store.initialize()
    foundational_runtime = ReviewingRuntime(review_revision="检索证据不足，不能回答。")
    foundational_workflow = SearchAssistantWorkflow(
        store=foundational_store,
        runtime=foundational_runtime,
        search_client=RecordingSearchClient([]),
    )

    foundational = foundational_workflow.answer(
        IncomingMessage(
            message_id="m-grounding-empty-foundational",
            event_id="e-grounding-empty-foundational",
            user_id="u-1",
            chat_id="c-1",
            text="解释一下向量数据库的基本概念。",
            source="cli",
        )
    )

    evidence_store = MemoryStore(tmp_path / "evidence.sqlite3")
    evidence_store.initialize()
    evidence_runtime = FakeAgentRuntime(answer_text="ACME X900 was released on August 1, 2026.")
    evidence_workflow = SearchAssistantWorkflow(
        store=evidence_store,
        runtime=evidence_runtime,
        search_client=RecordingSearchClient([]),
    )

    evidence_required = evidence_workflow.answer(
        IncomingMessage(
            message_id="m-grounding-empty-evidence-required",
            event_id="e-grounding-empty-evidence-required",
            user_id="u-1",
            chat_id="c-1",
            text="What is the current release date of ACME X900 in 2026?",
            source="cli",
        )
    )

    assert foundational_runtime.answer_contexts
    assert "基础解释" in foundational.answer_text
    assert "不能回答" not in foundational.answer_text
    assert evidence_runtime.answer_calls == 0
    assert "August 1, 2026" not in evidence_required.answer_text
    assert "搜索未返回可用结果" in evidence_required.answer_text


def test_relevance_gate_allows_strong_acronym_source_match(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = RecordingRuntime()
    search = RecordingSearchClient(
        [
            SearchResult(
                title="Compute Express Link CXL Consortium",
                url="https://www.computeexpresslink.org/",
                snippet="CXL is an open interconnect standard for CPU-to-device and CPU-to-memory connectivity.",
                provider="direct-official",
                checked_at="2026-06-29T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-cxl-relevance-1",
        event_id="e-cxl-relevance-1",
        user_id="u-1",
        chat_id="c-1",
        text="What is CXL?",
        source="cli",
    )

    package = workflow.answer(message)

    assert runtime.answer_contexts
    assert package.review["approved"] is True
    assert "搜索结果相关性不足" not in package.answer_text


def test_relevance_gate_allows_h100_inference_bandwidth_source(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = RecordingRuntime()
    search = RecordingSearchClient(
        [
            SearchResult(
                title="NVIDIA H100 Tensor Core GPU specifications",
                url="https://www.nvidia.com/en-us/data-center/h100/",
                snippet=(
                    "H100 uses HBM3 memory bandwidth, Tensor Cores, and Transformer Engine "
                    "for large language model inference throughput."
                ),
                provider="direct-official",
                checked_at="2026-06-29T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-h100-bandwidth-relevance-1",
        event_id="e-h100-bandwidth-relevance-1",
        user_id="u-1",
        chat_id="c-1",
        text="为什么H100的tok/s就很高，是哪里的带宽让他可以输出的这么快？",
        source="cli",
    )

    package = workflow.answer(message)

    assert runtime.answer_contexts
    assert package.classification == "hard"
    assert package.review["approved"] is True
    assert "搜索结果相关性不足" not in package.answer_text


def test_workflow_filters_irrelevant_search_tail_when_relevant_sources_exist(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = RecordingRuntime()
    search = RecordingSearchClient(
        [
            SearchResult(
                title="NVIDIA H100 Tensor Core GPU specifications",
                url="https://www.nvidia.com/en-us/data-center/h100/",
                snippet="H100 NVL uses Transformer Engine, NVLink, and 188GB HBM3 memory for LLM inference.",
                provider="direct-official",
                checked_at="2026-06-29T00:00:00Z",
            ),
            SearchResult(
                title="室内 照明测量方法 GB 5700-85.doc",
                url="https://example.com/lighting-measurement.pdf",
                snippet="A PDF about indoor lighting measurement methods and building standards.",
                provider="browser-bing",
                checked_at="2026-06-29T00:00:00Z",
            ),
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-h100-filter-irrelevant-1",
        event_id="e-h100-filter-irrelevant-1",
        user_id="u-1",
        chat_id="c-1",
        text="为什么H100的tok/s就很高，是哪里的带宽让他可以输出的这么快？",
        source="cli",
    )

    package = workflow.answer(message)

    assert [source.url for source in package.sources] == ["https://www.nvidia.com/en-us/data-center/h100/"]
    assert "lighting-measurement" not in package.answer_text
    assert runtime.answer_contexts[0]["search_results"][0]["url"] == "https://www.nvidia.com/en-us/data-center/h100/"
    assert len(runtime.answer_contexts[0]["search_results"]) == 1


def test_workflow_filters_irrelevant_browser_result_with_weak_tracking_token_overlap(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = RecordingRuntime()
    search = RecordingSearchClient(
        [
            SearchResult(
                title="NVIDIA H100 Tensor Core GPU specifications",
                url="https://www.nvidia.com/en-us/data-center/h100/",
                snippet="H100 supports LLM inference with Transformer Engine, HBM, and NVLink Switch System.",
                provider="direct-official",
                checked_at="2026-07-02T00:00:00Z",
            ),
            SearchResult(
                title="Explainer-Why are the Houthis threatening to attack Red Sea shipping ...",
                url=(
                    "https://www.al-monitor.com/originals/2026/06/"
                    "explainer-why-are-houthis-threatening-attack-red-sea-shipping-and-what-does-it"
                ),
                snippet=(
                    "The Houthi attacks in the Red Sea disrupted global shipping. "
                    "Page excerpt: tok=YOUR_TRACKING_CODE; hotjar PageView script."
                ),
                provider="browser-bing",
                checked_at="2026-07-02T00:00:00Z",
            ),
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-h100-filter-tracking-token-1",
        event_id="e-h100-filter-tracking-token-1",
        user_id="u-1",
        chat_id="c-1",
        text="为什么H100的tok/s就很高，是哪里的带宽让它可以输出这么快？",
        source="cli",
    )

    package = workflow.answer(message)

    assert [source.url for source in package.sources] == ["https://www.nvidia.com/en-us/data-center/h100/"]
    assert "al-monitor" not in package.answer_text
    assert len(runtime.answer_contexts[0]["search_results"]) == 1


def test_relevance_gate_allows_physical_ai_source_family_for_chinese_question(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = RecordingRuntime()
    search = RecordingSearchClient(
        [
            SearchResult(
                title="NVIDIA Cosmos official world foundation models page",
                url="https://www.nvidia.com/en-us/ai-data-science/cosmos/",
                snippet="NVIDIA Cosmos provides world foundation models for physical AI.",
                provider="direct-official",
                checked_at="2026-06-29T00:00:00Z",
            ),
            SearchResult(
                title="Google DeepMind Genie 2 official world model article",
                url="https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/",
                snippet="Genie 2 is a foundation world model for interactive environments.",
                provider="direct-official",
                checked_at="2026-06-29T00:00:00Z",
            ),
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-physical-ai-cn-relevance-1",
        event_id="e-physical-ai-cn-relevance-1",
        user_id="u-1",
        chat_id="c-1",
        text="现在AI物理大模型有哪些可靠进展？",
        source="cli",
    )

    package = workflow.answer(message)

    assert runtime.answer_contexts
    assert package.review["approved"] is True
    assert "搜索结果相关性不足" not in package.answer_text


def test_h100_tokens_per_second_question_expands_fallback_search_query(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = RecordingRuntime()
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-h100-bandwidth-search-1",
        event_id="e-h100-bandwidth-search-1",
        user_id="u-1",
        chat_id="c-1",
        text="为什么H100的tok/s就很高，是哪里的带宽让他可以输出的这么快？",
        source="cli",
    )

    package = workflow.answer(message)

    assert package.classification == "hard"
    assert search.queries == [
        "H100 tok/s LLM inference decode throughput HBM3 memory bandwidth KV cache Tensor Core"
    ]


def test_english_deployment_calculation_question_triggers_search(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    planned_queries = [
        "DGX Spark official specifications ConnectX-7 memory",
        "ModelScope DeepSeek-V3 671B 37B parameters",
        "Hugging Face DeepSeek-V3 model card active parameters",
    ]
    runtime = PlanningRuntime(planned_queries)
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-deploy-calc-1",
        event_id="e-deploy-calc-1",
        user_id="u-1",
        chat_id="c-1",
        text=(
            "Help me calculate whether 10 parallel NVIDIA GB10 systems can deploy "
            "a full DeepSeek-V4-class model. Include memory assumptions and limits."
        ),
        source="cli",
    )

    package = workflow.answer(message)

    assert package.classification in {"research", "hard"}
    assert search.queries == planned_queries
    assert runtime.search_plan_contexts[0]["classification"] == "hard"


def test_hardware_deployment_gap_replaces_unsupported_positive_feasibility_answer(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = UnsafeDeploymentRuntime()
    search = RecordingSearchClient(
        [
            SearchResult(
                title="NVIDIA DGX Spark official specifications",
                url="https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
                snippet="DGX Spark includes 128 GB unified memory, 273 GB/s memory bandwidth, and ConnectX-7 networking.",
                provider="direct-official",
                checked_at="2026-07-03T00:00:00Z",
            ),
            SearchResult(
                title="DeepSeek-V4 on ModelScope",
                url="https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4",
                snippet="DeepSeek-V4 model page exists, but this page excerpt does not disclose total parameters, active parameters, experts, or benchmark data.",
                provider="direct-model-hub",
                checked_at="2026-07-03T00:00:00Z",
            ),
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-deploy-gap-1",
        event_id="e-deploy-gap-1",
        user_id="u-1",
        chat_id="c-1",
        text=(
            "Help me calculate whether 10 parallel NVIDIA GB10 systems can deploy "
            "a full DeepSeek-V4-class model. Include memory assumptions and limits."
        ),
        source="cli",
    )

    package = workflow.answer(message)

    assert "can probably fit" not in package.answer_text
    assert "few tok/s" not in package.answer_text
    assert "cannot confirm" in package.answer_text
    assert "requested model parameters" in package.answer_text
    assert "NVIDIA DGX Spark official specifications" in package.answer_text
    assert any("Unsupported evidence-required answer replaced" in claim for claim in package.unverified_claims)


def test_hardware_deployment_gap_replaces_uncertain_answer_with_conditional_fit_claim(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(
        answer_text=(
            "根据搜索证据，无法判断 10 台 DGX Spark 是否能部署 DeepSeek-V4。"
            "但如果假设 DeepSeek-V4 与 V3 类似并使用 FP4，权重分摊到 10 个节点后确实小于 128 GB，"
            "因此内存上可能装下，只是缺少公开参数和 benchmark。"
        )
    )
    search = RecordingSearchClient(
        [
            SearchResult(
                title="NVIDIA DGX Spark official specifications",
                url="https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
                snippet="DGX Spark includes 128 GB unified memory, 273 GB/s memory bandwidth, and ConnectX-7 networking.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            ),
            SearchResult(
                title="DeepSeek-V4 on ModelScope",
                url="https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4",
                snippet="DeepSeek-V4 model page exists, but this excerpt does not disclose total parameters or benchmark data.",
                provider="direct-model-hub",
                checked_at="2026-07-04T00:00:00Z",
            ),
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-deploy-gap-uncertain-claim-1",
        event_id="e-deploy-gap-uncertain-claim-1",
        user_id="u-1",
        chat_id="c-1",
        text=(
            "Help me calculate whether 10 parallel NVIDIA GB10 systems can deploy "
            "a full DeepSeek-V4-class model. Include memory assumptions and limits."
        ),
        source="cli",
    )

    package = workflow.answer(message)

    assert "确实小于 128 GB" not in package.answer_text
    assert "可能装下" not in package.answer_text
    assert "cannot confirm" in package.answer_text
    assert "requested model parameters" in package.answer_text
    assert any("Unsupported evidence-required answer replaced" in claim for claim in package.unverified_claims)


def test_hardware_deployment_ignores_parameter_numbers_echoed_from_search_query(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(
        answer_text=(
            "Conditional answer: memory capacity is likely sufficient under FP8 or FP4 quantization, "
            "and 10 nodes can probably fit the model, but throughput will be single-digit tok/s."
        )
    )
    search = RecordingSearchClient(
        [
            SearchResult(
                title="NVIDIA DGX Spark official specifications",
                url="https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
                snippet="DGX Spark includes 128 GB unified memory, 273 GB/s memory bandwidth, and ConnectX-7 networking.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            ),
            SearchResult(
                title="DeepSeek-V4 on Hugging Face",
                url="https://huggingface.co/deepseek-ai/DeepSeek-V4",
                snippet=(
                    "Direct source selected for query: DeepSeek-V4 model parameters total parameters "
                    "MoE architecture 671B 37B activated layers experts. Page content fetch was unavailable."
                ),
                provider="direct-model-hub",
                checked_at="2026-07-04T00:00:00Z",
            ),
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-deploy-gap-query-echo-1",
        event_id="e-deploy-gap-query-echo-1",
        user_id="u-1",
        chat_id="c-1",
        text=(
            "Help me calculate whether 10 parallel NVIDIA GB10 systems can deploy "
            "a full DeepSeek-V4-class model. Include memory assumptions and limits."
        ),
        source="cli",
    )

    package = workflow.answer(message)

    assert "can probably fit" not in package.answer_text
    assert "single-digit tok/s" not in package.answer_text
    assert "cannot confirm" in package.answer_text
    assert "requested model parameters" in package.answer_text
    assert any("Unsupported evidence-required answer replaced" in claim for claim in package.unverified_claims)


def test_hardware_deployment_does_not_use_v3_parameters_as_v4_evidence(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(
        answer_text=(
            "10台DGX Spark（GB10）在FP4或FP8量化下有可能在内存层面装下一个"
            "类DeepSeek-V3规模的MoE模型。DeepSeek-V3为671B总参数、37B激活参数。"
            "结论：FP8或FP4下权重肯定放得下。"
        )
    )
    search = RecordingSearchClient(
        [
            SearchResult(
                title="NVIDIA DGX Spark official specifications",
                url="https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
                snippet="DGX Spark includes 128 GB unified memory, 273 GB/s memory bandwidth, and ConnectX-7 networking.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            ),
            SearchResult(
                title="DeepSeek-V3 on ModelScope",
                url="https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3",
                snippet=(
                    "DeepSeek-V3 is a MoE model with 671B total parameters and 37B activated "
                    "for each token."
                ),
                provider="direct-model-hub",
                checked_at="2026-07-04T00:00:00Z",
            ),
            SearchResult(
                title="DeepSeek-V4 on Hugging Face",
                url="https://huggingface.co/deepseek-ai/DeepSeek-V4",
                snippet="Direct source selected for query: DeepSeek-V4 model card parameters. Page content fetch was unavailable.",
                provider="direct-model-hub",
                checked_at="2026-07-04T00:00:00Z",
            ),
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-deploy-gap-v3-not-v4-1",
        event_id="e-deploy-gap-v3-not-v4-1",
        user_id="u-1",
        chat_id="c-1",
        text=(
            "Help me calculate whether 10 parallel NVIDIA GB10 systems can deploy "
            "a full DeepSeek-V4-class model. Include memory assumptions and limits."
        ),
        source="cli",
    )

    package = workflow.answer(message)

    assert "有可能在内存层面装下" not in package.answer_text
    assert "权重肯定放得下" not in package.answer_text
    assert "cannot confirm" in package.answer_text
    assert "requested model parameters" in package.answer_text
    assert any("Unsupported evidence-required answer replaced" in claim for claim in package.unverified_claims)


def test_workflow_uses_runtime_search_plan_instead_of_topic_hardcoding(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = PlanningRuntime(
        [
            "runtime planned official model card query",
            "runtime planned interconnect specification query",
        ]
    )
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-runtime-plan-1",
        event_id="e-runtime-plan-1",
        user_id="u-1",
        chat_id="c-1",
        text="10台GB10并联跑满血DeepSeek-V4，需要怎么查参数和互联限制？",
        source="cli",
    )

    workflow.answer(message)

    assert search.queries == [
        "runtime planned official model card query",
        "runtime planned interconnect specification query",
    ]
    assert runtime.search_plan_contexts[0]["classification"] == "hard"


def test_user_correction_is_recorded_as_experience_for_future_planning(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = PlanningRuntime(["answer correction search query"])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=RecordingSearchClient([]))
    message = IncomingMessage(
        message_id="m-feedback-1",
        event_id="e-feedback-1",
        user_id="u-1",
        chat_id="c-1",
        text="你这个回答不对，漏掉了魔搭社区和CX7网卡，以后搜索要先理解问题再规划关键词。",
        source="cli",
    )

    package = workflow.answer(message)

    experiences = store.list_experience_items()
    feedback_items = [item for item in experiences if item["title"] == "User feedback: search and answer correction"]
    assert len(feedback_items) == 1
    assert "魔搭社区" in feedback_items[0]["body"]
    assert "CX7" in feedback_items[0]["body"]
    assert package.question_id in feedback_items[0]["source_ids_json"]


def test_user_correction_command_is_handled_as_feedback_not_question(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = PlanningRuntime(["ordinary question search should not run"])
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-feedback-command-1",
        event_id="e-feedback-command-1",
        user_id="u-1",
        chat_id="c-1",
        text=(
            "你为什么没有去魔塔这些开源模型社区去查找deepseek的模型参数，这里都是有的。"
            "这种明确的修正类命令不应该识别为问题对话。"
        ),
        source="cli",
    )

    package = workflow.answer(message)

    assert runtime.search_plan_contexts == []
    assert runtime.answer_contexts == []
    assert search.queries == []
    assert package.answer_text.startswith("已记录这条修正")
    assert "ModelScope" in package.answer_text
    assert "魔塔" in package.answer_text
    assert "联网搜索: 未执行" in package.answer_text
    feedback_items = [item for item in store.list_experience_items() if item["title"] == "User feedback: search and answer correction"]
    assert len(feedback_items) == 1
    assert "魔塔" in feedback_items[0]["body"]
    assert "ModelScope" in feedback_items[0]["body"]
    assert "source_family=deepseek_model_communities" in feedback_items[0]["body"]


def test_answer_style_correction_is_feedback_even_without_direct_you(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = RecordingRuntime()
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-feedback-style-1",
        event_id="e-feedback-style-1",
        user_id="u-1",
        chat_id="c-1",
        text="这个搜索助手回答有点太死板，没有体现出 LLM 的思考，以后要像技术伙伴一样说明判断过程。",
        source="cli",
    )

    package = workflow.answer(message)

    assert runtime.answer_contexts == []
    assert search.queries == []
    assert "已记录这条修正" in package.answer_text
    feedback_items = [item for item in store.list_experience_items() if item["title"] == "User feedback: search and answer correction"]
    assert len(feedback_items) == 1
    assert "answer_style=natural_technical_partner" in feedback_items[0]["body"]
    assert "avoid_rigid_template" in feedback_items[0]["body"]


def test_source_insufficiency_correction_is_feedback_and_records_foundational_fallback_rule(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = RecordingRuntime()
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-feedback-source-relevance-1",
        event_id="e-feedback-source-relevance-1",
        user_id="u-1",
        chat_id="c-1",
        text="面对这个问题他又摆烂了，搜索内容不足不应该作为阻断理由，起码要提供一些基础的信息。",
        source="feishu-long-connection",
    )

    package = workflow.answer(message)

    assert runtime.answer_contexts == []
    assert search.queries == []
    assert "已记录这条修正" in package.answer_text
    assert "基础概念" in package.answer_text
    feedback_items = [item for item in store.list_experience_items() if item["title"] == "User feedback: search and answer correction"]
    assert len(feedback_items) == 1
    assert "source_relevance_policy=foundational_fallback" in feedback_items[0]["body"]
    assert "avoid_blocking_foundational_answers=true" in feedback_items[0]["body"]


def test_user_skill_generation_command_creates_review_only_draft_without_search_or_model(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    profile_id = store.add_profile_snapshot(
        {
            "recurring_topics": ["H100 throughput reasoning"],
            "preferred_answer_style": "mechanism-first technical answers",
        }
    )
    runtime = PlanningRuntime(["ordinary search should not run"])
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=runtime,
        search_client=search,
        skill_drafts_dir=tmp_path / "skills" / "drafts",
    )
    message = IncomingMessage(
        message_id="m-skill-command-1",
        event_id="e-skill-command-1",
        user_id="u-1",
        chat_id="c-1",
        text="请生成skill：H100 Throughput Reasoning",
        source="feishu-long-connection",
    )

    package = workflow.answer(message)

    assert runtime.search_plan_contexts == []
    assert runtime.answer_contexts == []
    assert search.queries == []
    drafts = store.list_skill_drafts()
    assert len(drafts) == 1
    assert drafts[0]["name"] == "H100 Throughput Reasoning"
    assert drafts[0]["review_status"] == "draft"
    assert json.loads(drafts[0]["source_ids_json"]) == [profile_id]
    assert Path(drafts[0]["path"]).read_text(encoding="utf-8").startswith("---\nname: h100-throughput-reasoning")
    assert "已创建 skill 草稿" in package.answer_text
    assert "SKILL.md" in package.answer_text
    assert "联网搜索: 未执行" in package.answer_text


def test_user_skill_generation_command_falls_back_to_latest_answer_source(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    previous = AnswerPackage(
        question_id="q-previous-answer",
        answer_text="Feishu reply checklist answer",
        classification="research",
        confidence="medium",
    )
    store.record_answer(previous)
    runtime = PlanningRuntime(["ordinary search should not run"])
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=runtime,
        search_client=RecordingSearchClient([]),
        skill_drafts_dir=tmp_path / "skills" / "drafts",
    )

    workflow.answer(
        IncomingMessage(
            message_id="m-skill-command-2",
            event_id="e-skill-command-2",
            user_id="u-1",
            chat_id="c-1",
            text="把刚才的回答沉淀成skill：Feishu Reply Practice",
            source="cli",
        )
    )

    drafts = store.list_skill_drafts()
    assert len(drafts) == 1
    assert drafts[0]["name"] == "Feishu Reply Practice"
    assert json.loads(drafts[0]["source_ids_json"]) == ["q-previous-answer"]


def test_user_skill_list_command_reports_review_status_without_search_or_model(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    skill_service = SkillDraftService(
        store,
        drafts_dir=tmp_path / "skills" / "drafts",
        active_dir=tmp_path / "skills" / "active",
    )
    skill_service.create_from_experience("Draft Only Guidance", source_ids=["manual"])
    skill_service.create_from_experience("Feishu Reply Practice", source_ids=["manual"])
    skill_service.promote("feishu-reply-practice")
    runtime = PlanningRuntime(["ordinary search should not run"])
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=runtime,
        search_client=search,
        skill_drafts_dir=tmp_path / "skills" / "drafts",
    )

    package = workflow.answer(
        IncomingMessage(
            message_id="m-skill-list-command",
            event_id="e-skill-list-command",
            user_id="u-1",
            chat_id="c-1",
            text="查看skill列表",
            source="feishu-long-connection",
        )
    )

    assert runtime.search_plan_contexts == []
    assert runtime.answer_contexts == []
    assert search.queries == []
    assert "skill 列表" in package.answer_text
    assert "total=2" in package.answer_text
    assert "draft-only-guidance" in package.answer_text
    assert "feishu-reply-practice" in package.answer_text
    assert "promoted" in package.answer_text
    assert "联网搜索: 未执行" in package.answer_text


def test_user_skill_promote_command_activates_draft_for_later_questions(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    SkillDraftService(
        store,
        drafts_dir=tmp_path / "skills" / "drafts",
        active_dir=tmp_path / "skills" / "active",
    ).create_from_experience("Feishu Reply Practice", source_ids=["manual"])
    runtime = PlanningRuntime(["Feishu reply API official docs"])
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=runtime,
        search_client=search,
        skill_drafts_dir=tmp_path / "skills" / "drafts",
    )

    promote_package = workflow.answer(
        IncomingMessage(
            message_id="m-skill-promote-command",
            event_id="e-skill-promote-command",
            user_id="u-1",
            chat_id="c-1",
            text="启用skill：Feishu Reply Practice",
            source="feishu-long-connection",
        )
    )
    answer_package = workflow.answer(
        IncomingMessage(
            message_id="m-after-skill-promote",
            event_id="e-after-skill-promote",
            user_id="u-1",
            chat_id="c-1",
            text="How should I verify Feishu replies?",
            source="cli",
        )
    )

    drafts = store.list_skill_drafts()
    assert drafts[0]["review_status"] == "promoted"
    assert Path(drafts[0]["path"]).exists()
    assert "已启用 skill" in promote_package.answer_text
    assert "feishu-reply-practice" in promote_package.answer_text
    assert "联网搜索: 未执行" in promote_package.answer_text
    assert runtime.answer_contexts[0]["active_skills"][0]["name"] == "Feishu Reply Practice"
    assert answer_package.answer_text


def test_user_learning_report_command_generates_report_without_search_or_model(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.record_answer(
        AnswerPackage(
            question_id="q-existing-answer",
            answer_text="Existing Feishu learning answer.",
            classification="research",
            confidence="medium",
        )
    )
    store.add_memory_item("topic", "Feishu integration", "q-existing-answer")
    store.add_experience_item("Verification habit", "Search before answering.", ["q-existing-answer"])
    runtime = PlanningRuntime(["ordinary search should not run"])
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=runtime,
        search_client=search,
        report_output_dir=tmp_path / "reports",
    )
    message = IncomingMessage(
        message_id="m-report-command-1",
        event_id="e-report-command-1",
        user_id="u-1",
        chat_id="c-1",
        text="请生成学习报告",
        source="feishu-long-connection",
    )

    package = workflow.answer(message)

    assert runtime.search_plan_contexts == []
    assert runtime.answer_contexts == []
    assert search.queries == []
    report_path = tmp_path / "reports" / "learning-report.md"
    assert report_path.exists()
    assert "Feishu integration" in report_path.read_text(encoding="utf-8")
    reports = store.list_learning_reports()
    assert len(reports) == 1
    assert reports[0]["path"] == str(report_path)
    assert package.classification == "simple"
    assert package.confidence == "high"
    assert str(report_path) in package.answer_text
    assert "Existing Feishu learning answer." in package.answer_text
    assert "learning report command" in package.review["reason"]


def test_english_learning_report_command_uses_same_report_path(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = PlanningRuntime(["ordinary search should not run"])
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=runtime,
        search_client=RecordingSearchClient([]),
        report_output_dir=tmp_path / "reports",
    )

    package = workflow.answer(
        IncomingMessage(
            message_id="m-report-command-2",
            event_id="e-report-command-2",
            user_id="u-1",
            chat_id="c-1",
            text="learning report",
            source="cli",
        )
    )

    assert runtime.search_plan_contexts == []
    assert (tmp_path / "reports" / "learning-report.md").exists()
    assert "learning-report.md" in package.answer_text


def test_feishu_evolution_command_generates_report_and_review_only_skill_draft(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    profile_id = store.add_profile_snapshot(
        {
            "recurring_topics": ["H100 throughput reasoning"],
            "preferred_answer_style": "natural technical partner",
        }
    )
    runtime = PlanningRuntime(["ordinary search should not run"])
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=runtime,
        search_client=search,
        skill_drafts_dir=tmp_path / "skills" / "drafts",
        report_output_dir=tmp_path / "reports",
    )
    message = IncomingMessage(
        message_id="m-evolve-command-1",
        event_id="e-evolve-command-1",
        user_id="u-1",
        chat_id="c-1",
        text="开始自进化",
        source="feishu-long-connection",
    )

    package = workflow.answer(message)

    assert runtime.search_plan_contexts == []
    assert runtime.answer_contexts == []
    assert search.queries == []
    report_path = tmp_path / "reports" / "learning-report.md"
    assert report_path.exists()
    drafts = store.list_skill_drafts()
    assert len(drafts) == 1
    assert drafts[0]["review_status"] == "draft"
    assert drafts[0]["source_ids_json"] == json.dumps([profile_id])
    assert Path(drafts[0]["path"]).exists()
    assert "已运行自进化" in package.answer_text
    assert str(report_path) in package.answer_text
    assert "H100 Throughput Reasoning Practice" in package.answer_text
    assert "review-only draft" in package.answer_text
    assert "联网搜索: 未执行" in package.answer_text
    assert package.review["reason"] == "self-evolution command generated report and review-only skill drafts"
    experiences = [item for item in store.list_experience_items() if item["title"] == "User requested self-evolution"]
    assert len(experiences) == 1
    assert "draft_count=1" in experiences[0]["body"]


def test_evolution_command_reports_refreshed_review_only_skill_drafts(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.add_profile_snapshot(
        {
            "recurring_topics": ["AI infrastructure and model deployment"],
            "preferred_answer_style": "natural technical partner",
        }
    )
    skill_service = SkillDraftService(store, drafts_dir=tmp_path / "skills" / "drafts")
    paths = skill_service.auto_create_from_experience(max_drafts=1)
    draft_path = Path(paths[0])
    draft_path.write_text(
        (
            "---\n"
            "name: ai-infrastructure-and-model-deployment-practice\n"
            "description: old generic draft\n"
            "---\n\n"
            "## Instructions\n\n"
            "Answer similar future questions with explicit assumptions, verification notes, and a short final recommendation.\n"
        ),
        encoding="utf-8",
    )
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=PlanningRuntime(["ordinary search should not run"]),
        search_client=RecordingSearchClient([]),
        skill_drafts_dir=tmp_path / "skills" / "drafts",
        report_output_dir=tmp_path / "reports",
    )

    package = workflow.answer(
        IncomingMessage(
            message_id="m-evolve-refresh-1",
            event_id="e-evolve-refresh-1",
            user_id="u-1",
            chat_id="c-1",
            text="运行自进化",
            source="feishu-long-connection",
        )
    )

    assert "新增 skill 草稿: 0" in package.answer_text
    assert "刷新 skill 草稿: 1" in package.answer_text
    assert str(draft_path) in package.answer_text
    assert "ModelScope/魔搭/魔塔" in draft_path.read_text(encoding="utf-8")


def test_english_evolve_command_skips_ordinary_qa_even_when_no_new_skill_created(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = PlanningRuntime(["ordinary search should not run"])
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=runtime,
        search_client=RecordingSearchClient([]),
        skill_drafts_dir=tmp_path / "skills" / "drafts",
        report_output_dir=tmp_path / "reports",
    )

    package = workflow.answer(
        IncomingMessage(
            message_id="m-evolve-command-2",
            event_id="e-evolve-command-2",
            user_id="u-1",
            chat_id="c-1",
            text="evolve",
            source="cli",
        )
    )

    assert runtime.search_plan_contexts == []
    assert (tmp_path / "reports" / "learning-report.md").exists()
    assert "新增 skill 草稿: 0" in package.answer_text
    assert "搜索记录:" in package.answer_text


def test_user_feedback_entities_are_added_as_search_constraints(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.add_experience_item(
        title="User feedback: search and answer correction",
        body=(
            "user_feedback=以后回答DeepSeek部署问题，不要漏掉魔搭社区和CX7网卡。\n"
            "future_rule=cover missed entities/source families."
        ),
        source_ids=["q-previous"],
    )
    runtime = PlanningRuntime(["NVIDIA GB10 DGX Spark memory"])
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-feedback-constraints-1",
        event_id="e-feedback-constraints-1",
        user_id="u-1",
        chat_id="c-1",
        text="Can 10 GB10 systems deploy DeepSeek-V4?",
        source="cli",
    )

    workflow.answer(message)

    assert search.queries[0] == "NVIDIA GB10 DGX Spark memory"
    assert any("魔搭" in query for query in search.queries)
    assert any("CX7" in query for query in search.queries)
    assert len(search.queries) <= 6


def test_feedback_about_model_communities_adds_deepseek_parameter_search_constraints(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.add_experience_item(
        title="User feedback: search and answer correction",
        body=(
            "user_feedback=你为什么没有去魔塔这些开源模型社区去查找deepseek的模型参数，这里都是有的。\n"
            "source_family=deepseek_model_communities\n"
            "future_rule=For DeepSeek model parameter questions, search ModelScope/魔搭/魔塔, Hugging Face, GitHub model cards first."
        ),
        source_ids=["q-previous"],
    )
    runtime = PlanningRuntime(["DeepSeek model parameters"])
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-feedback-model-communities-1",
        event_id="e-feedback-model-communities-1",
        user_id="u-1",
        chat_id="c-1",
        text="DeepSeek模型参数是多少？",
        source="cli",
    )

    workflow.answer(message)

    assert any("ModelScope" in query and "魔塔" in query for query in search.queries)
    assert any("Hugging Face" in query for query in search.queries)
    assert any("GitHub" in query for query in search.queries)


def test_workflow_stops_starting_extra_search_queries_after_budget(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = PlanningRuntime(
        [
            "first planned query",
            "second planned query",
            "third planned query",
        ]
    )
    search = SleepingRecordingSearchClient(delay_seconds=0.03)
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=runtime,
        search_client=search,
        search_budget_seconds=0.01,
    )
    message = IncomingMessage(
        message_id="m-search-budget-1",
        event_id="e-search-budget-1",
        user_id="u-1",
        chat_id="c-1",
        text=(
            "Help me calculate whether 10 parallel NVIDIA GB10 systems can deploy "
            "a full DeepSeek-V4-class model. Include memory assumptions and limits."
        ),
        source="cli",
    )

    package = workflow.answer(message)

    assert search.queries == ["first planned query"]
    assert "联网搜索: 已执行" in package.answer_text


def test_workflow_reuses_existing_interaction_when_previous_answer_failed(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    message = IncomingMessage(
        message_id="m-retry-after-failure",
        event_id="e-retry-after-failure",
        user_id="u-1",
        chat_id="c-1",
        text="hello retry",
        source="cli",
    )
    failing_workflow = SearchAssistantWorkflow(
        store=store,
        runtime=FailingRuntime(),
        search_client=RecordingSearchClient([]),
    )

    with pytest.raises(RuntimeError, match="runtime blocked"):
        failing_workflow.answer(message)

    retry_workflow = SearchAssistantWorkflow(
        store=store,
        runtime=FakeAgentRuntime(answer_text="retry succeeded"),
        search_client=RecordingSearchClient([]),
    )
    package = retry_workflow.answer(message)

    interactions = store.list_interactions()
    assert len(interactions) == 1
    assert package.question_id == interactions[0]["id"]
    assert package.answer_text.startswith("retry succeeded")


def test_english_cxl_question_uses_compact_search_query(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = RecordingRuntime()
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-cxl-en-1",
        event_id="e-cxl-en-1",
        user_id="u-1",
        chat_id="c-1",
        text="What is CXL, and how does it relate to AI servers, memory pooling, GPUs, and accelerators?",
        source="cli",
    )

    package = workflow.answer(message)

    assert package.classification == "research"
    assert search.queries == ["CXL AI servers memory pooling GPUs accelerators"]


def test_english_ai_physics_model_question_uses_compact_search_query(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    planned_queries = [
        "GR00T N1 physical AI NVIDIA",
        "Genie 3 DeepMind world model",
        "AI physics foundation models world models embodied physical AI 2026",
    ]
    runtime = PlanningRuntime(planned_queries)
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-ai-physics-en-1",
        event_id="e-ai-physics-en-1",
        user_id="u-1",
        chat_id="c-1",
        text="As of 2026, where are AI physics foundation models, world models, and embodied physical AI?",
        source="cli",
    )

    package = workflow.answer(message)

    assert package.classification == "research"
    assert search.queries == planned_queries


def test_workflow_persists_source_verified_claims(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(
        answer_text="Feishu message reply API supports replying to received messages in 2026."
    )
    search = RecordingSearchClient(
        [
            SearchResult(
                title="Feishu message reply API 2026",
                url="https://open.feishu.cn/document/server-docs/im-v1/message/reply",
                snippet="The Feishu message reply API supports replying to received messages from a bot.",
                provider="unit",
                checked_at="2026-06-27T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-verify-1",
        event_id="e-verify-1",
        user_id="u-1",
        chat_id="c-1",
        text="What is the current Feishu message reply API behavior?",
        source="cli",
    )

    package = workflow.answer(message)

    assert [claim.claim for claim in package.verified_claims] == [
        "Feishu message reply API supports replying to received messages in 2026"
    ]
    assert package.unverified_claims == []
    assert store.list_evidence()[0]["source"] == "https://open.feishu.cn/document/server-docs/im-v1/message/reply"


def test_workflow_persists_chinese_markdown_verified_claims_without_search_record_text(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(
        answer_text=(
            "### 证据核查\n"
            "- NVIDIA DGX Spark 官方规格确认 128GB 统一内存和 ConnectX-7 网卡。\n"
            "\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [direct-official] NVIDIA DGX Spark official specifications - https://www.nvidia.com/en-us/products/workstations/dgx-spark/"
        )
    )
    search = RecordingSearchClient(
        [
            SearchResult(
                title="NVIDIA DGX Spark official specifications",
                url="https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
                snippet="NVIDIA DGX Spark 官方规格确认 128GB 统一内存和 ConnectX-7 网卡。",
                provider="direct-official",
                checked_at="2026-07-03T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-verify-cn-markdown-1",
        event_id="e-verify-cn-markdown-1",
        user_id="u-1",
        chat_id="c-1",
        text="请验证 NVIDIA DGX Spark 官方规格中的内存和网卡信息。",
        source="cli",
    )

    package = workflow.answer(message)
    evidence = store.list_evidence()

    assert [claim.claim for claim in package.verified_claims] == [
        "NVIDIA DGX Spark 官方规格确认 128GB 统一内存和 ConnectX-7 网卡"
    ]
    assert package.unverified_claims == []
    assert evidence[0]["claim"] == "NVIDIA DGX Spark 官方规格确认 128GB 统一内存和 ConnectX-7 网卡"
    assert evidence[0]["source"] == "https://www.nvidia.com/en-us/products/workstations/dgx-spark/"
    assert all("搜索记录" not in claim.claim for claim in package.verified_claims)


def test_workflow_persists_verified_claims_from_markdown_evidence_table(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(
        answer_text=(
            "### 证据核查\n"
            "\n"
            "| 来源 | 确认了什么 | 缺少什么 |\n"
            "|------|-----------|----------|\n"
            "| NVIDIA DGX Spark 官方规格 | 128GB 统一内存、273GB/s 带宽、ConnectX-7 网卡 | 无 DeepSeek-V4 性能实测 |\n"
            "\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
        )
    )
    search = RecordingSearchClient(
        [
            SearchResult(
                title="NVIDIA DGX Spark 官方规格",
                url="https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
                snippet="NVIDIA DGX Spark 官方规格确认 128GB 统一内存、273GB/s 带宽、ConnectX-7 网卡。",
                provider="direct-official",
                checked_at="2026-07-03T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-verify-cn-table-1",
        event_id="e-verify-cn-table-1",
        user_id="u-1",
        chat_id="c-1",
        text="请验证 NVIDIA DGX Spark 官方规格表格里的内存、带宽和网卡信息。",
        source="cli",
    )

    package = workflow.answer(message)
    evidence = store.list_evidence()

    assert [claim.claim for claim in package.verified_claims] == [
        "NVIDIA DGX Spark 官方规格 确认 128GB 统一内存、273GB/s 带宽、ConnectX-7 网卡"
    ]
    assert package.unverified_claims == []
    assert evidence[0]["claim"] == "NVIDIA DGX Spark 官方规格 确认 128GB 统一内存、273GB/s 带宽、ConnectX-7 网卡"
    assert evidence[0]["source"] == "https://www.nvidia.com/en-us/products/workstations/dgx-spark/"


def test_workflow_records_experience_item_for_answer(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(answer_text="Current API answers need search verification in 2026.")
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=RecordingSearchClient([]))
    message = IncomingMessage(
        message_id="m-exp-1",
        event_id="e-exp-1",
        user_id="u-1",
        chat_id="c-1",
        text="How should I verify current API answers before replying?",
        source="cli",
    )

    package = workflow.answer(message)

    experiences = store.list_experience_items()
    assert len(experiences) == 1
    assert experiences[0]["title"] == "Answer pattern: research"
    assert package.question_id in experiences[0]["source_ids_json"]
    assert "confidence=" in experiences[0]["body"]
    assert "unverified_claims=" in experiences[0]["body"]


def test_source_insufficiency_correction_is_recorded_instead_of_answered_as_qa(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FailingRuntime()
    search = RecordingSearchClient(
        [
            SearchResult(
                title="Should Not Search",
                url="https://example.com/should-not-search",
                snippet="This result should not be requested for correction feedback.",
                provider="unit",
                checked_at="2026-07-03T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-source-insufficiency-feedback-1",
        event_id="e-source-insufficiency-feedback-1",
        user_id="u-1",
        chat_id="c-1",
        text=(
            "面对这个问题他又摆烂了，搜索内容不足不应该作为阻断理由，"
            "他起码要提供一些基础的信息，并且这个问题他是应该能回答的"
        ),
        source="cli",
    )

    package = workflow.answer(message)
    experiences = store.list_experience_items()

    assert search.queries == []
    assert package.answer_text.startswith("已记录这条修正")
    assert "基础概念/理解校正类问题不能因为搜索证据弱就直接阻断" in package.answer_text
    assert experiences[0]["title"] == "User feedback: search and answer correction"
    assert "source_relevance_policy=foundational_fallback" in experiences[0]["body"]


def test_mixed_question_and_source_insufficiency_feedback_answers_question_and_records_feedback(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = ReviewingRuntime(review_revision="搜索内容不足：当前来源不足，不能回答。")
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-mixed-foundational-feedback-1",
        event_id="e-mixed-foundational-feedback-1",
        user_id="u-1",
        chat_id="c-1",
        text=(
            "分布式大模型是否可以理解为多个相对独立的节点，每个节点负责一部分推理工作，并且不同节点间互联互通\n"
            "面对这个问题他又摆烂了，搜索内容不足不应该作为阻断理由，"
            "他起码要提供一些基础的信息，并且这个问题他是应该能回答的"
        ),
        source="cli",
    )

    package = workflow.answer(message)
    experiences = store.list_experience_items()
    feedback_items = [item for item in experiences if item["title"] == "User feedback: search and answer correction"]

    assert search.queries
    assert runtime.answer_contexts
    assert runtime.answer_contexts[0]["question"].startswith("分布式大模型是否可以理解为")
    assert "摆烂" not in runtime.answer_contexts[0]["question"]
    assert not package.answer_text.startswith("已记录这条修正")
    assert "可以这么理解" in package.answer_text
    assert "节点" in package.answer_text
    assert "推理" in package.answer_text
    assert "互联" in package.answer_text
    assert "搜索记录:" in package.answer_text
    assert len(feedback_items) == 1
    assert "source_relevance_policy=foundational_fallback" in feedback_items[0]["body"]
    assert "avoid_blocking_foundational_answers=true" in feedback_items[0]["body"]


def test_workflow_removes_unsupported_hardware_spec_numbers_from_visible_answer(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(
        answer_text=(
            "可以这么理解。单个 GPU 的内存（即使 80GB HBM）也可能放不下超大模型，"
            "所以分布式推理会把张量并行、流水线并行或专家并行分到多个节点。"
        )
    )
    search = RecordingSearchClient(
        [
            SearchResult(
                title="vLLM distributed inference parallelism",
                url="https://docs.vllm.ai/en/stable/serving/parallelism_scaling/",
                snippet=(
                    "Distributed inference supports tensor parallelism, pipeline parallelism, "
                    "expert parallelism, and communication across nodes."
                ),
                provider="direct-official",
                checked_at="2026-07-03T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-unsupported-spec-number-1",
        event_id="e-unsupported-spec-number-1",
        user_id="u-1",
        chat_id="c-1",
        text="分布式大模型是否可以理解为多个节点分别负责一部分推理工作？",
        source="cli",
    )

    package = workflow.answer(message)

    assert "80GB" not in package.answer_text
    assert "HBM）" not in package.answer_text
    assert "张量并行" in package.answer_text
    assert any("80GB HBM" in claim for claim in package.unverified_claims)


def test_workflow_removes_unsupported_h100_example_calculation(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(
        answer_text=(
            "H100 的 tok/s 高主要来自 HBM3 带宽和 Tensor Core 算力。\n\n"
            "机制追溯:\n"
            "- H100 SXM 的 HBM3 带宽是 3.35 TB/s，FP8 Tensor Core 算力是 3,958 TFLOPS。\n"
            "- 以 70B FP8 模型为例，H100 的 3.35 TB/s 带宽下，读取时间约 21 毫秒。"
        )
    )
    search = RecordingSearchClient(
        [
            SearchResult(
                title="NVIDIA H100 Tensor Core GPU specifications",
                url="https://www.nvidia.com/en-us/data-center/h100/",
                snippet="H100 SXM has 3.35 TB/s HBM3 memory bandwidth and 3,958 TFLOPS FP8 Tensor Core performance.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-unsupported-h100-example-calculation-1",
        event_id="e-unsupported-h100-example-calculation-1",
        user_id="u-1",
        chat_id="c-1",
        text="为什么H100的tok/s就很高，是哪里的带宽让它可以输出这么快？",
        source="cli",
    )

    package = workflow.answer(message)

    assert "3.35 TB/s" in package.answer_text
    assert "3,958 TFLOPS" in package.answer_text
    assert "70B" not in package.answer_text
    assert "21 毫秒" not in package.answer_text
    assert any("Removed unsupported hardware example estimate" in claim for claim in package.unverified_claims)


def test_workflow_removes_external_knowledge_sentences_from_visible_answer(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(
        answer_text=(
            "CXL 是缓存一致性互连，支持内存扩展和加速器资源共享。"
            "搜索中未获取具体产品列表，但 CXL 内存扩展设备已有厂商公开（如 Samsung、Micron）——"
            "此信息未在本次搜索结果中出现，来自外部知识，需自行核实。"
            "因此 CXL 更适合作为容量扩展层而非主性能路径——但这一结论是基于行业常识的逻辑推断，搜索结果中没有实际部署案例。"
        )
    )
    search = RecordingSearchClient(
        [
            SearchResult(
                title="Compute Express Link Consortium About CXL official page",
                url="https://www.computeexpresslink.org/about-cxl/",
                snippet=(
                    "CXL is a cache-coherent interconnect for processors, memory expansion, "
                    "and accelerators, allowing resource sharing."
                ),
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-external-knowledge-cleanup-1",
        event_id="e-external-knowledge-cleanup-1",
        user_id="u-1",
        chat_id="c-1",
        text="What is CXL, and how does it relate to AI servers?",
        source="cli",
    )

    package = workflow.answer(message)

    assert "CXL 是缓存一致性互连" in package.answer_text
    assert "Samsung" not in package.answer_text
    assert "Micron" not in package.answer_text
    assert "来自外部知识" not in package.answer_text
    assert "行业常识" not in package.answer_text


def test_workflow_removes_external_knowledge_marker_lines(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(
        answer_text=(
            "CXL 是缓存一致性互连。\n\n"
            "### 关键假设（此处为模型记忆推测，未获搜索直接验证）\n"
            "- CXL 作为容量扩展层这一角色是基于带宽差距的定性推断，未获得部署验证。\n\n"
            "下一步验证: 查找厂商白皮书。"
        )
    )
    search = RecordingSearchClient(
        [
            SearchResult(
                title="Compute Express Link Consortium About CXL official page",
                url="https://www.computeexpresslink.org/about-cxl/",
                snippet="CXL is a cache-coherent interconnect for processors, memory expansion, and accelerators.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-external-knowledge-marker-line-1",
        event_id="e-external-knowledge-marker-line-1",
        user_id="u-1",
        chat_id="c-1",
        text="What is CXL, and how does it relate to AI servers?",
        source="cli",
    )

    package = workflow.answer(message)

    assert "模型记忆" not in package.answer_text
    assert "外部知识" not in package.answer_text
    assert "下一步验证" in package.answer_text


def test_workflow_removes_malformed_partial_lines_from_visible_answer(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(
        answer_text=(
            "CXL 是缓存一致性互连，官方页面确认它面向处理器、内存扩展和加速器。\n"
            "- 假设CXL的带宽（128 GT/s，CXL 4.\n"
            "3.\n"
            "- 官方页面确认 CXL 4.0 带宽从 64 GT/s 提升至 128 GT/s。"
        )
    )
    search = RecordingSearchClient(
        [
            SearchResult(
                title="Compute Express Link Consortium About CXL official page",
                url="https://www.computeexpresslink.org/about-cxl/",
                snippet=(
                    "CXL is a cache-coherent interconnect for processors, memory expansion, "
                    "and accelerators. CXL 4.0 doubles bandwidth from 64GTs to 128GTs."
                ),
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-malformed-partial-cleanup-1",
        event_id="e-malformed-partial-cleanup-1",
        user_id="u-1",
        chat_id="c-1",
        text="What is CXL, and how does it relate to AI servers?",
        source="cli",
    )

    package = workflow.answer(message)

    assert "CXL 4.\n" not in package.answer_text
    assert "\n3.\n" not in package.answer_text
    assert "官方页面确认 CXL 4.0" in package.answer_text
    assert "搜索记录:" in package.answer_text


def test_ai_infrastructure_questions_update_learning_topics(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(answer_text="AI infrastructure answer with verification.")
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=RecordingSearchClient([]))
    message = IncomingMessage(
        message_id="m-ai-infra-1",
        event_id="e-ai-infra-1",
        user_id="u-1",
        chat_id="c-1",
        text="Can 10 NVIDIA GB10 systems deploy a DeepSeek-V4 model, and how should I verify the memory assumptions?",
        source="cli",
    )

    package = workflow.answer(message)
    topics = [item["content"] if isinstance(item, dict) else item.content for item in package.memory_updates]

    assert "AI infrastructure and model deployment" in topics
    assert "Verification practice" in topics


def test_cxl_and_physical_ai_questions_update_learning_topics(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(answer_text="Research answer.")
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=RecordingSearchClient([]))

    cxl_package = workflow.answer(
        IncomingMessage(
            message_id="m-cxl-topic",
            event_id="e-cxl-topic",
            user_id="u-1",
            chat_id="c-1",
            text="What is CXL and how does memory pooling help AI servers and GPUs?",
            source="cli",
        )
    )
    physical_ai_package = workflow.answer(
        IncomingMessage(
            message_id="m-physical-ai-topic",
            event_id="e-physical-ai-topic",
            user_id="u-1",
            chat_id="c-1",
            text="Where are world models and embodied physical AI going in 2026?",
            source="cli",
        )
    )

    cxl_topics = [item["content"] if isinstance(item, dict) else item.content for item in cxl_package.memory_updates]
    physical_ai_topics = [
        item["content"] if isinstance(item, dict) else item.content for item in physical_ai_package.memory_updates
    ]

    assert "AI memory architecture" in cxl_topics
    assert "Physical AI and world models" in physical_ai_topics


def test_chinese_explicit_search_compute_question_triggers_search_with_compact_query(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    planned_queries = [
        "DGX Spark official specifications ConnectX-7 memory",
        "魔搭 ModelScope DeepSeek-V3 671B 37B",
        "NVIDIA GB10 DGX Spark DeepSeek-V4-Pro deployment memory estimate",
    ]
    runtime = PlanningRuntime(planned_queries)
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-search-cn-1",
        event_id="e-search-cn-1",
        user_id="u-1",
        chat_id="c-1",
        text="搜索并计算：10台并联 NVIDIA GB10（DGX Spark / Project DIGITS）能否部署满血 DeepSeek-V4-Pro？请给出显存/统一内存估算、权重量化假设和结论。",
        source="cli",
    )

    package = workflow.answer(message)

    assert package.classification in {"research", "hard"}
    assert search.queries == planned_queries


def test_chinese_cxl_search_question_triggers_search_with_compact_query(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = RecordingRuntime()
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-search-cn-2",
        event_id="e-search-cn-2",
        user_id="u-1",
        chat_id="c-1",
        text="搜索并解释：CXL 是什么？它和 AI 服务器、内存池化、GPU/加速器有什么关系？",
        source="cli",
    )

    package = workflow.answer(message)

    assert package.classification == "research"
    assert search.queries == ["CXL 是什么"]


def test_chinese_what_is_technical_question_triggers_search(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = RecordingRuntime()
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-search-cn-plain-cxl",
        event_id="e-search-cn-plain-cxl",
        user_id="u-1",
        chat_id="c-1",
        text="CXL是什么？",
        source="cli",
    )

    package = workflow.answer(message)

    assert package.classification == "research"
    assert search.queries == ["CXL是什么"]


def test_chinese_ai_physics_model_question_triggers_search_with_compact_query(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    planned_queries = [
        "GR00T N1 physical AI NVIDIA",
        "Genie 3 DeepMind world model",
        "AI 物理大模型 world model physics foundation model embodied physical AI 2026",
    ]
    runtime = PlanningRuntime(planned_queries)
    search = RecordingSearchClient([])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-search-cn-3",
        event_id="e-search-cn-3",
        user_id="u-1",
        chat_id="c-1",
        text="搜索并总结：截至 2026 年，AI 物理大模型（world model / physics foundation model / embodied physical AI）发展到哪里了？请说典型公司、技术路线和局限。",
        source="cli",
    )

    package = workflow.answer(message)

    assert package.classification == "research"
    assert search.queries == planned_queries


def test_technical_question_passes_visible_reasoning_strategy_to_runtime(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = PlanningRuntime(["H100 tok/s HBM3 memory bandwidth decode throughput"])
    search = RecordingSearchClient(
        [
            SearchResult(
                title="NVIDIA H100 Tensor Core GPU specifications",
                url="https://www.nvidia.com/en-us/data-center/h100/",
                snippet=(
                    "H100 inference uses HBM3 memory bandwidth, Tensor Cores, "
                    "Transformer Engine, KV cache, and decode throughput."
                ),
                provider="direct-official",
                checked_at="2026-07-02T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-answer-strategy-tech",
        event_id="e-answer-strategy-tech",
        user_id="u-1",
        chat_id="c-1",
        text="为什么H100的tok/s就很高，是哪里的带宽让它可以输出这么快？",
        source="cli",
    )

    package = workflow.answer(message)

    strategy = runtime.search_plan_contexts[0].get("answer_strategy")
    assert strategy is not None
    assert strategy["mode"] == "technical_explanation"
    assert strategy["style"] == "technical_partner"
    assert strategy["hidden_reasoning_policy"] == "do_not_reveal_chain_of_thought"
    assert strategy["max_clarifying_questions"] == 1
    assert strategy["reasoning_contract"] == "must_show_compact_user_visible_reasoning"
    assert strategy["presentation_contract"] == "natural_technical_partner_not_mechanical_checklist"
    assert strategy["section_policy"] == "required_sections_are_reasoning_moves_not_mandatory_headings"
    assert strategy["preferred_response_shape"] == "answer_first_then_reasoning_snapshot_then_next_check"
    assert strategy["required_sections"] == [
        "direct_judgment",
        "key_assumptions",
        "evidence_check",
        "reasoning_path",
        "counterpoints_or_risks",
        "next_verification",
    ]
    assert strategy["minimum_reasoning_detail"] == "explain_mechanism_or_calculation_before_final_recommendation"
    assert "direct_judgment" in strategy["visible_reasoning"]
    assert "reasoning_snapshot" in strategy["visible_reasoning"]
    assert "mechanistic_trace" in strategy["visible_reasoning"]
    assert "key_assumptions" in strategy["visible_reasoning"]
    assert "reasoning_path" in strategy["visible_reasoning"]
    assert "counterpoints_or_risks" in strategy["visible_reasoning"]
    assert "next_verification" in strategy["visible_reasoning"]
    assert runtime.answer_contexts[0]["answer_strategy"] == strategy
    assert runtime.calibration_contexts[0]["answer_strategy"] == strategy
    assert runtime.review_contexts[0]["answer_strategy"] == strategy
    assert "搜索记录:" in package.answer_text


def test_brainstorming_question_uses_discussion_strategy(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = PlanningRuntime(["H100 inference bottleneck HBM bandwidth Tensor Core KV cache"])
    search = RecordingSearchClient(
        [
            SearchResult(
                title="NVIDIA H100 Tensor Core GPU specifications",
                url="https://www.nvidia.com/en-us/data-center/h100/",
                snippet="H100 has HBM3 memory, Tensor Cores, Transformer Engine, and LLM inference features.",
                provider="direct-official",
                checked_at="2026-07-02T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-answer-strategy-brainstorm",
        event_id="e-answer-strategy-brainstorm",
        user_id="u-1",
        chat_id="c-1",
        text="我们头脑风暴一下：H100 tok/s 高到底主要受 HBM 带宽、Tensor Core 还是 KV cache 影响？可以反驳我的假设。",
        source="cli",
    )

    workflow.answer(message)

    strategy = runtime.answer_contexts[0].get("answer_strategy")
    assert strategy is not None
    assert strategy["mode"] == "brainstorming_discussion"
    assert strategy["style"] == "technical_partner"
    assert strategy["reasoning_contract"] == "must_show_compact_user_visible_reasoning"
    assert strategy["presentation_contract"] == "natural_technical_partner_not_mechanical_checklist"
    assert strategy["section_policy"] == "required_sections_are_reasoning_moves_not_mandatory_headings"
    assert strategy["preferred_response_shape"] == "judgment_then_hypotheses_then_collaborative_pushback"
    assert strategy["required_sections"] == [
        "direct_judgment",
        "competing_hypotheses",
        "evidence_check",
        "pushback",
        "decision_criteria",
        "next_verification",
    ]
    assert "competing_hypotheses" in strategy["visible_reasoning"]
    assert "reasoning_snapshot" in strategy["visible_reasoning"]
    assert "pushback" in strategy["visible_reasoning"]
    assert "decision_criteria" in strategy["visible_reasoning"]
    assert strategy["stance"] == "exploratory_but_evidence_grounded"


def test_chinese_partner_discussion_phrase_uses_discussion_strategy(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = PlanningRuntime(["H100 decode throughput HBM bandwidth KV cache"])
    search = RecordingSearchClient(
        [
            SearchResult(
                title="NVIDIA H100 Tensor Core GPU specifications",
                url="https://www.nvidia.com/en-us/data-center/h100/",
                snippet="H100 supports LLM inference with HBM3 memory, Tensor Cores, and Transformer Engine.",
                provider="direct-official",
                checked_at="2026-07-02T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)
    message = IncomingMessage(
        message_id="m-answer-strategy-partner-discussion",
        event_id="e-answer-strategy-partner-discussion",
        user_id="u-1",
        chat_id="c-1",
        text="你怎么看：H100 tok/s 高有没有可能主要不是算力，而是 HBM 带宽和 KV cache？",
        source="cli",
    )

    workflow.answer(message)

    strategy = runtime.answer_contexts[0].get("answer_strategy")
    assert strategy["mode"] == "brainstorming_discussion"
    assert strategy["preferred_response_shape"] == "judgment_then_hypotheses_then_collaborative_pushback"


def test_promoted_active_skill_is_passed_to_runtime_contexts(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    skill_service = SkillDraftService(
        store,
        drafts_dir=tmp_path / "skills" / "drafts",
        active_dir=tmp_path / "skills" / "active",
    )
    skill_service.create_from_experience("H100 Throughput Reasoning", source_ids=["manual"])
    active_path = Path(skill_service.promote("h100-throughput-reasoning"))
    active_path.write_text(
        active_path.read_text(encoding="utf-8")
        + "\n## Reviewed Guidance\n\nAlways explain HBM bandwidth, KV cache, prefill/decode split, and interconnect before the conclusion.\n",
        encoding="utf-8",
    )
    runtime = PlanningRuntime(["H100 inference HBM bandwidth KV cache"])
    search = RecordingSearchClient(
        [
            SearchResult(
                title="NVIDIA H100 Tensor Core GPU specifications",
                url="https://www.nvidia.com/en-us/data-center/h100/",
                snippet="H100 uses HBM3, Tensor Cores, Transformer Engine, and LLM inference features.",
                provider="direct-official",
                checked_at="2026-07-02T00:00:00Z",
            )
        ]
    )
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=search)

    workflow.answer(
        IncomingMessage(
            message_id="m-active-skill",
            event_id="e-active-skill",
            user_id="u-1",
            chat_id="c-1",
            text="Why is H100 tok/s high?",
            source="cli",
        )
    )

    active_skills = runtime.search_plan_contexts[0].get("active_skills")
    assert active_skills == runtime.answer_contexts[0].get("active_skills")
    assert active_skills == runtime.calibration_contexts[0].get("active_skills")
    assert active_skills == runtime.review_contexts[0].get("active_skills")
    assert active_skills[0]["name"] == "H100 Throughput Reasoning"
    assert active_skills[0]["path"] == str(active_path)
    assert "HBM bandwidth, KV cache" in active_skills[0]["content"]


def test_draft_skill_is_not_passed_to_runtime_contexts(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    SkillDraftService(store, drafts_dir=tmp_path / "skills" / "drafts").create_from_experience(
        "Draft Only Guidance",
        source_ids=["manual"],
    )
    runtime = PlanningRuntime(["Feishu bot reply verification"])
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime, search_client=RecordingSearchClient([]))

    workflow.answer(
        IncomingMessage(
            message_id="m-draft-skill",
            event_id="e-draft-skill",
            user_id="u-1",
            chat_id="c-1",
            text="How should I verify Feishu replies?",
            source="cli",
        )
    )

    assert runtime.answer_contexts[0].get("active_skills") == []


class RecordingRuntime:
    def __init__(self):
        self.answer_contexts = []
        self.calibration_contexts = []
        self.review_contexts = []

    def generate_answer(self, question, context):
        self.answer_contexts.append(context)
        return "draft answer mentions current API behavior."

    def calibrate(self, draft, context):
        self.calibration_contexts.append(context)
        return {
            "ran": True,
            "critique": "checked against sources",
            "revision": "calibrated answer with citations",
        }

    def review_answer(self, answer, context):
        self.review_contexts.append(context)
        return {
            "ran": True,
            "approved": True,
            "issues": [],
            "revision": answer,
        }


class ReviewingRuntime(RecordingRuntime):
    def __init__(self, review_revision):
        super().__init__()
        self.review_revision = review_revision

    def review_answer(self, answer, context):
        self.review_contexts.append(context)
        return {
            "ran": True,
            "approved": True,
            "issues": [],
            "revision": self.review_revision,
        }


class RejectedReviewRuntime(RecordingRuntime):
    def generate_answer(self, question, context):
        self.answer_contexts.append(context)
        return "unsafe unsupported original answer"

    def review_answer(self, answer, context):
        self.review_contexts.append(context)
        return {
            "ran": True,
            "approved": False,
            "issues": ["unsupported model-memory claims"],
            "revision": "unsafe unsupported original answer",
        }


class UnsafeDeploymentRuntime(RecordingRuntime):
    def __init__(self):
        super().__init__()
        self.unsafe_answer = (
            "10 DGX Spark systems can probably fit DeepSeek-V4 with FP4, "
            "but single-stream decode will only be a few tok/s."
        )

    def generate_answer(self, question, context):
        self.answer_contexts.append(context)
        return self.unsafe_answer

    def calibrate(self, draft, context):
        self.calibration_contexts.append(context)
        return {
            "ran": True,
            "critique": "left the estimate in place",
            "revision": self.unsafe_answer,
        }

    def review_answer(self, answer, context):
        self.review_contexts.append(context)
        return {
            "ran": True,
            "approved": True,
            "issues": [],
            "revision": self.unsafe_answer,
        }


class PlanningRuntime(RecordingRuntime):
    def __init__(self, planned_queries):
        super().__init__()
        self.planned_queries = planned_queries
        self.search_plan_contexts = []

    def plan_search_queries(self, question, context):
        self.search_plan_contexts.append(context)
        return self.planned_queries


class FailingRuntime:
    def generate_answer(self, question, context):
        raise RuntimeError("runtime blocked")

    def calibrate(self, draft, context):
        raise AssertionError("calibration should not run")


class RecordingSearchClient:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, query, limit=5):
        self.queries.append(query)
        return self.results[:limit]


class SleepingRecordingSearchClient:
    def __init__(self, delay_seconds):
        self.delay_seconds = delay_seconds
        self.queries = []

    def search(self, query, limit=5):
        self.queries.append(query)
        time.sleep(self.delay_seconds)
        return []
