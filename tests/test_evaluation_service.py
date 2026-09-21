from pathlib import Path

from search_assistant.contracts import AnswerPackage, SearchRecord, SourceEvidence
from search_assistant.evaluation.service import DEFAULT_EVALUATION_QUESTIONS, EvaluationService, _evaluation_item
from search_assistant.memory.store import MemoryStore
from search_assistant.search.provider import SearchResult
from search_assistant.runtime import FakeAgentRuntime
from search_assistant.workflow.service import SearchAssistantWorkflow


def test_evaluation_suite_runs_multiple_questions_and_writes_report(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=FakeAgentRuntime(
            answer_text="Feishu message reply API supports replying to received messages in 2026."
        ),
        search_client=StaticSearchClient(),
    )
    service = EvaluationService(
        store=store,
        workflow=workflow,
        output_dir=tmp_path / "evaluations",
        report_output_dir=tmp_path / "reports",
    )

    result = service.run(
        [
            "What is the current Feishu message reply API behavior?",
            "What is CXL and why does it matter for AI servers?",
            "Give me a learning direction for Feishu bot verification.",
        ]
    )

    assert result["total_questions"] == 3
    assert len(result["items"]) == 3
    assert result["summary"]["answers_recorded"] == 3
    assert result["summary"]["profile_snapshots"] == 3
    assert result["summary"]["learning_report_written"] is True
    assert Path(result["summary"]["evaluation_report_path"]).exists()
    assert Path(result["summary"]["learning_report_path"]).exists()
    assert result["items"][0]["source_count"] == 1
    assert result["items"][0]["verified_claims"] == 1
    assert "answer_excerpt" in result["items"][0]
    assert "Feishu message reply API" in result["items"][0]["answer_excerpt"]
    assert "搜索记录:" in result["items"][0]["search_record"]
    assert result["items"][0]["source_urls"] == [
        "https://open.feishu.cn/document/server-docs/im-v1/message/reply"
    ]
    assert result["items"][0]["review_approved"] is True
    assert result["items"][0]["review_issues"] == []
    assert result["items"][0]["quality_flags"] == []
    assert result["items"][0]["result_verification"]["passed"] is True
    assert result["items"][0]["process_verification"]["passed"] is True
    assert result["items"][0]["quality_verification"]["passed"] is True
    assert result["items"][0]["diagnosis"]["recommended_update_carrier"] == "none"
    assert result["summary"]["trajectory_evaluations"] == 3
    assert len(store.list_trajectory_logs()) == 3
    assert len(store.list_trajectory_evaluations()) == 3
    report = Path(result["summary"]["evaluation_report_path"]).read_text(encoding="utf-8")
    assert "answer_excerpt" in report
    assert "search_record" in report
    assert store.list_profile_snapshots()
    assert store.list_learning_reports()


def test_evaluation_suite_flags_review_rejection_and_blocked_answer(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=RejectedRuntime(),
        search_client=StaticSearchClient(),
    )
    service = EvaluationService(
        store=store,
        workflow=workflow,
        output_dir=tmp_path / "evaluations",
        report_output_dir=tmp_path / "reports",
    )

    result = service.run(["What is the current Feishu message reply API behavior?"])

    item = result["items"][0]
    assert item["review_approved"] is False
    assert "unsupported model-memory claims" in item["review_issues"]
    assert "review_rejected" in item["quality_flags"]
    assert "blocked_answer" in item["quality_flags"]
    assert item["result_verification"]["flags"] == ["task_blocked"]
    assert item["quality_verification"]["passed"] is False
    assert item["diagnosis"]["recommended_update_carrier"] == "program_harness"
    assert item["diagnosis"]["candidate_only"] is True
    assert result["summary"]["flagged_answers"] == 1
    assert result["summary"]["review_rejected_answers"] == 1
    experiences = store.list_experience_items()
    quality_experiences = [experience for experience in experiences if experience["title"] == "Evaluation quality issue"]
    assert len(quality_experiences) == 1
    assert item["question_id"] in quality_experiences[0]["source_ids_json"]
    assert "quality_flags=" in quality_experiences[0]["body"]
    assert "review_rejected" in quality_experiences[0]["body"]
    assert "blocked_answer" in quality_experiences[0]["body"]
    assert "review_issues=unsupported model-memory claims" in quality_experiences[0]["body"]
    assert "future_rule=Use evaluation failures as self-evolution input" in quality_experiences[0]["body"]


def test_evaluation_item_uses_structured_search_record_when_visible_text_missing():
    source = SourceEvidence(
        title="Official docs",
        url="https://example.com/docs",
        snippet="Official source snippet.",
        provider="unit",
        checked_at="2026-07-03T00:00:00Z",
    )
    package = AnswerPackage(
        question_id="q-structured-record",
        answer_text="Answer body only.",
        classification="research",
        confidence="medium",
        sources=[source],
        search_record=SearchRecord(
            executed=True,
            queries=["official docs query"],
            engines=["bing", "baidu"],
            sources=[source],
        ),
        review={"ran": True, "approved": True, "issues": [], "revision": "Answer body only."},
        calibration={"ran": True, "critique": "ok", "revision": "Answer body only."},
    )

    item = _evaluation_item(1, "Question?", package)

    assert item["search_record_structured"]["executed"] is True
    assert item["search_record_structured"]["queries"] == ["official docs query"]
    assert item["search_record_structured"]["sources"][0]["url"] == "https://example.com/docs"
    assert "official docs query" in item["search_record"]


def test_evaluation_does_not_flag_justified_low_confidence_uncertainty():
    package = AnswerPackage(
        question_id="q-justified-uncertainty",
        answer_text=(
            "无法确认 DeepSeek-V4 的公开参数量；当前只能给条件估算。"
            "假设参数量、精度和并行策略不同，结论会变化。\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [unit] NVIDIA DGX Spark - https://example.test/dgx-spark"
        ),
        classification="hard",
        confidence="low",
        verified_claims=[],
        unverified_claims=["DeepSeek-V4 参数量未从搜索结果中确认"],
        sources=[
            SourceEvidence(
                title="NVIDIA DGX Spark",
                url="https://example.test/dgx-spark",
                snippet="Official DGX Spark product information.",
                provider="unit",
                checked_at="2026-07-03T00:00:00Z",
            )
        ],
        calibration={"ran": True, "critique": "Missing model parameter evidence was disclosed."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
    )

    item = _evaluation_item(1, "Can 10 GB10 systems deploy DeepSeek-V4?", package)

    assert item["uncertainty_assessment"] == "justified"
    assert item["quality_flags"] == []


def test_evaluation_treats_hardware_evidence_gap_audit_markers_as_justified_uncertainty():
    source = SourceEvidence(
        title="NVIDIA DGX Spark official specifications",
        url="https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
        snippet="DGX Spark product specifications.",
        provider="direct-official",
        checked_at="2026-07-03T00:00:00Z",
    )
    package = AnswerPackage(
        question_id="q-hardware-evidence-gap",
        answer_text=(
            "I cannot confirm whether the requested GB10/DGX Spark cluster can deploy a full "
            "DeepSeek-V4-class model from the retrieved evidence.\n\n"
            "What is still missing:\n"
            "- DeepSeek-V4 public model parameters: total parameters, active parameters, MoE expert count, "
            "hidden size/layers, and precision or quantization target.\n"
            "- A supported parallelism plan for more than two GB10/DGX Spark nodes.\n"
            "- A benchmark or reproducible serving configuration for tok/s, latency, batch size, context length, "
            "and KV-cache memory.\n\n"
            "Safe conclusion:\n"
            "- The answer should stay at low confidence until those model and benchmark facts are found.\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [direct-official] NVIDIA DGX Spark official specifications - "
            "https://www.nvidia.com/en-us/products/workstations/dgx-spark/"
        ),
        classification="hard",
        confidence="low",
        verified_claims=[],
        unverified_claims=[
            "A benchmark or reproducible serving configuration for tok/s, latency, batch size, context length, and KV-cache memory.",
            "The answer should stay at low confidence until those model and benchmark facts are found.",
            "Removed unsupported hardware spec number: 85 GB",
            "Unsupported hardware deployment feasibility answer replaced: missing DeepSeek-V4 public model parameters or benchmark evidence.",
        ],
        sources=[source],
        calibration={"ran": True, "critique": "Missing public model parameter and benchmark evidence was disclosed."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
        search_record=SearchRecord(
            executed=True,
            queries=["DeepSeek-V4 total parameters", "NVIDIA DGX Spark official specifications"],
            engines=["bing", "baidu", "google"],
            sources=[source],
        ),
    )

    item = _evaluation_item(1, "Can 10 GB10 systems deploy DeepSeek-V4?", package)

    assert item["uncertainty_assessment"] == "justified"
    assert item["quality_flags"] == []


def test_evaluation_treats_research_stage_product_unknowns_as_justified_uncertainty():
    package = AnswerPackage(
        question_id="q-physical-ai-uncertainty",
        answer_text=(
            "The official source describes Genie 2 as a research system, not a shipped product or API. "
            "Cosmos production readiness and API accessibility are unknown from the current evidence. "
            "Genie 2 has no announced product or API timeline, so it should be treated as a research demo.\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [direct-official] Google DeepMind Genie 2 - https://deepmind.google/example"
        ),
        classification="research",
        confidence="low",
        verified_claims=[],
        unverified_claims=[
            "Based on the evidence collected in this search, the publicly documented landscape has no single dominant production API yet.",
            "The post explicitly frames it as a research system — it is not presented as a shipped product or API, and there is no timeline for that in the source.",
            "Cosmos’s actual production readiness and API accessibility are unknown from the current evidence.",
            "Genie 2 has no announced product or API timeline – it is still a research demo.",
            "No benchmark data is available in these results.",
        ],
        sources=[
            SourceEvidence(
                title="Google DeepMind Genie 2 official world model article",
                url="https://deepmind.google/example",
                snippet="Genie 2 is a research preview of a foundation world model.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            )
        ],
        calibration={"ran": True, "critique": "Research-stage uncertainty was disclosed."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
    )

    item = _evaluation_item(1, "Where are physical AI world models?", package)

    assert item["uncertainty_assessment"] == "justified"
    assert item["quality_flags"] == []


def test_evaluation_treats_unfetched_api_and_benchmark_gaps_as_justified_uncertainty():
    package = AnswerPackage(
        question_id="q-physical-ai-api-gaps",
        answer_text=(
            "The search found official world-model and robotics pages, but API and benchmark status "
            "remain low-confidence: one page was not fetched, no product or public API is mentioned, "
            "and no benchmark data is visible in the evidence.\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [direct-official] Google DeepMind Genie 2 - https://deepmind.google/example"
        ),
        classification="research",
        confidence="low",
        verified_claims=[],
        unverified_claims=[
            "The page content was not fetched, so exact release status and API availability are unconfirmed from this evidence.",
            "The framing is an article/blog post - no product or public API is mentioned.",
            "No public API, product timeline, or benchmark data is visible in the evidence.",
        ],
        sources=[
            SourceEvidence(
                title="Google DeepMind Genie 2 official world model article",
                url="https://deepmind.google/example",
                snippet="Genie 2 is a research preview of a foundation world model.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            )
        ],
        calibration={"ran": True, "critique": "Research-stage uncertainty was disclosed."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
    )

    item = _evaluation_item(1, "Where are physical AI world models?", package)

    assert item["uncertainty_assessment"] == "justified"
    assert item["quality_flags"] == []


def test_evaluation_treats_cxl_caveated_inferences_as_justified_uncertainty():
    package = AnswerPackage(
        question_id="q-cxl-caveated-inference",
        answer_text=(
            "CXL is cache-coherent interconnect evidence-backed by the official source. "
            "The AI-server implications are low-confidence because the search evidence did not provide "
            "GPU vendor support statements, deployment cases, or latency and throughput data.\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [direct-official] Compute Express Link Consortium About CXL - https://www.computeexpresslink.org/about-cxl/"
        ),
        classification="research",
        confidence="low",
        verified_claims=[],
        unverified_claims=[
            "The memory-pooling relationship is a reasonable inference from Memory Expansion, but the search evidence did not provide concrete GPU vendor support statements or deployment cases.",
            "The official page did not give a concrete implementation or performance metrics.",
            "The official page did not list specific GPU vendors or models.",
        ],
        sources=[
            SourceEvidence(
                title="Compute Express Link Consortium About CXL official page",
                url="https://www.computeexpresslink.org/about-cxl/",
                snippet="CXL is a cache-coherent interconnect for processors, memory expansion, and accelerators.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            )
        ],
        calibration={"ran": True, "critique": "Caveated inference was disclosed."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
    )

    item = _evaluation_item(1, "What is CXL?", package)

    assert item["uncertainty_assessment"] == "justified"
    assert item["quality_flags"] == []


def test_evaluation_treats_live_physical_ai_gap_samples_as_justified_uncertainty():
    package = AnswerPackage(
        question_id="q-physical-ai-live-gaps",
        answer_text=(
            "The publicly documented landscape has no single dominant production API yet visible. "
            "Cosmos details are unconfirmed because page content was not retrieved.\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [direct-official] Google DeepMind Genie 2 - https://deepmind.google/example"
        ),
        classification="research",
        confidence="low",
        verified_claims=[],
        unverified_claims=[
            "The blog post explicitly frames this as research - no product, public API, or timeline is mentioned.",
            "NVIDIA Cosmos is listed as a world foundation model platform for physical AI, but the page content was not retrieved, so exact capabilities are unconfirmed from this evidence.",
            "Genie 2 is a research system - there is no benchmark data, product timeline, or API access visible in the evidence.",
        ],
        sources=[
            SourceEvidence(
                title="Google DeepMind Genie 2 official world model article",
                url="https://deepmind.google/example",
                snippet="Genie 2 is a research preview of a foundation world model.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            )
        ],
        calibration={"ran": True, "critique": "Research-stage uncertainty was disclosed."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
    )

    item = _evaluation_item(1, "Where are physical AI world models?", package)

    assert item["uncertainty_assessment"] == "justified"
    assert item["quality_flags"] == []


def test_evaluation_treats_live_cxl_missing_deployment_samples_as_justified_uncertainty():
    package = AnswerPackage(
        question_id="q-cxl-live-gaps",
        answer_text=(
            "CXL facts are source-backed, but AI-server deployment evidence is low-confidence: "
            "the search did not find vendor support statements or deployment benchmarks.\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [direct-official] Compute Express Link Consortium About CXL - https://www.computeexpresslink.org/about-cxl/"
        ),
        classification="research",
        confidence="low",
        verified_claims=[],
        unverified_claims=[
            "The search did not find concrete GPU vendor support statements, deployment cases, or AI inference performance data.",
            "CXL and AI servers are at a standards-potential stage and lack actual deployment evidence.",
            "CXL value for KV-cache or cold model weights is logical reasoning, and search has no verification.",
        ],
        sources=[
            SourceEvidence(
                title="Compute Express Link Consortium About CXL official page",
                url="https://www.computeexpresslink.org/about-cxl/",
                snippet="CXL is a cache-coherent interconnect for processors, memory expansion, and accelerators.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            )
        ],
        calibration={"ran": True, "critique": "Caveated inference was disclosed."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
    )

    item = _evaluation_item(1, "What is CXL?", package)

    assert item["uncertainty_assessment"] == "justified"
    assert item["quality_flags"] == []


def test_evaluation_treats_live_cxl_official_facts_and_logic_as_justified_uncertainty():
    package = AnswerPackage(
        question_id="q-cxl-live-official-and-logic",
        answer_text=(
            "CXL 官方事实可以确认，但与 AI 服务器的部署关系仍是标准潜力。"
            "逻辑上，CXL 允许加速器访问统一内存地址空间；这一判断基于标准能力的一般理解，"
            "非搜索直接证据，厂商支持和实际部署仍未确认。\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [direct-official] Compute Express Link Consortium About CXL - https://www.computeexpresslink.org/about-cxl/"
        ),
        classification="research",
        confidence="low",
        verified_claims=[],
        unverified_claims=[
            "官方页面明确将AI和机器学习列为“新兴应用场景”，CXL 4.0规范已发布，带宽从64 GT/s翻倍至128 GT/s，并增加了捆绑端口和对内存RAS的增强",
            "CXL官方页面将AI/ML列为CXL的设计场景",
            "逻辑上，CXL允许加速器访问一个统一的内存地址空间，有助于缓解“内存墙”问题（如GPU本地内存容量有限）",
            "CXL官方描述的“Memory Expansion”直接支持内存池化",
        ],
        sources=[
            SourceEvidence(
                title="Compute Express Link Consortium About CXL official page",
                url="https://www.computeexpresslink.org/about-cxl/",
                snippet=(
                    "CXL is an industry-supported Cache-Coherent Interconnect for Processors, "
                    "Memory Expansion and Accelerators. CXL is designed to support emerging "
                    "applications such as Artificial Intelligence and Machine Learning. "
                    "CXL 4.0 doubles bandwidth from 64GTs to 128GTs."
                ),
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            )
        ],
        calibration={"ran": True, "critique": "CXL caveats were disclosed."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
    )

    item = _evaluation_item(1, "What is CXL?", package)

    assert item["uncertainty_assessment"] == "justified"
    assert item["quality_flags"] == []


def test_evaluation_treats_english_cxl_adoption_gap_as_justified_uncertainty():
    package = AnswerPackage(
        question_id="q-cxl-english-adoption-gap",
        answer_text=(
            "CXL is a cache-coherent interconnect. From the search evidence there is no confirmation "
            "that specific GPU vendors natively support CXL on current accelerator products; "
            "the standard covers the protocol, but adoption is a separate question.\n\n"
            "### Evidence check\n"
            "| Source | What is confirmed | What is missing |\n"
            "|--------|------------------|-----------------|\n"
            "| CXL homepage | CXL 4.0 bandwidth is 128 GT/s | GPU vendor adoption |\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [direct-official] Compute Express Link Consortium About CXL - https://www.computeexpresslink.org/about-cxl/"
        ),
        classification="research",
        confidence="low",
        verified_claims=[],
        unverified_claims=[
            "However, from the search evidence there is no confirmation that any specific GPU vendor natively supports CXL on their current accelerator products – the standard covers the protocol, but adoption is a separate question",
            "[CXL homepage](https://computeexpresslink.org/) 确认 Same as above; confirms CXL 4.0 specs (128 GT/s, bundled ports, enhanced RAS)",
            "No evidence that mainstream AI accelerators natively support CXL or utilize its cache-coherency",
            "CXL 4.0 bandwidth is 128 GT/s (confirmed by search)",
        ],
        sources=[
            SourceEvidence(
                title="Compute Express Link Consortium About CXL official page",
                url="https://www.computeexpresslink.org/about-cxl/",
                snippet="CXL 4.0 doubles bandwidth from 64GTs to 128GTs.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            )
        ],
        calibration={"ran": True, "critique": "CXL adoption gap was disclosed."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
    )

    item = _evaluation_item(1, "What is CXL?", package)

    assert item["uncertainty_assessment"] == "justified"
    assert item["quality_flags"] == []


def test_evaluation_treats_live_cxl_standard_potential_answer_as_justified_uncertainty():
    package = AnswerPackage(
        question_id="q-cxl-standard-potential",
        answer_text=(
            "Direct judgment: CXL is an open-standard, cache-coherent interconnect. "
            "The search evidence confirms the standard itself, but does not confirm actual "
            "GPU/accelerator-native support, deployment cases, or AI inference performance data. "
            "So the relationship today is a standard with strong potential, not a deployed reality.\n\n"
            "Key assumptions (logical inference from the standard, not from deployment evidence): "
            "CXL-attached memory is a capacity tier, not a performance tier; this is a general observation, "
            "not a specific HBM number from search.\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [direct-official] Compute Express Link Consortium About CXL - https://www.computeexpresslink.org/about-cxl/"
        ),
        classification="research",
        confidence="low",
        verified_claims=[],
        unverified_claims=[
            "The search evidence confirms the standard itself, including CXL 4.0’s doubled bandwidth to 128 GT/s, but does not confirm any actual GPU/accelerator-native support, deployment cases, or AI inference performance data",
            "So the relationship today is a standard with strong potential, not a deployed reality",
            "Bandwidth and latency trade-off: CXL 4.0’s 128 GT/s is a raw electrical spec",
            "This is significantly lower than the local memory bandwidth of current high-performance AI accelerators (a general observation, not a specific HBM number from search)",
        ],
        sources=[
            SourceEvidence(
                title="Compute Express Link Consortium About CXL official page",
                url="https://www.computeexpresslink.org/about-cxl/",
                snippet="CXL is a cache-coherent interconnect for processors, memory expansion, and accelerators. CXL 4.0 doubles bandwidth from 64GTs to 128GTs.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            )
        ],
        calibration={"ran": True, "critique": "CXL deployment uncertainty was disclosed."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
    )

    item = _evaluation_item(1, "What is CXL?", package)

    assert item["uncertainty_assessment"] == "justified"
    assert item["quality_flags"] == []


def test_evaluation_treats_physical_ai_research_to_product_stage_as_justified_uncertainty():
    package = AnswerPackage(
        question_id="q-physical-ai-research-to-product",
        answer_text=(
            "This is still a research-to-early-product transition zone, not a mature API market. "
            "The current evidence has no product or public API timeline for Genie and Cosmos remains unconfirmed.\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [direct-official] Google DeepMind Genie 2 - https://deepmind.google/example"
        ),
        classification="research",
        confidence="low",
        verified_claims=[],
        unverified_claims=[
            "As of mid-2026, the publicly documented landscape for AI physics foundation models, world models, and embodied physical AI is concentrated in a few major platform-level efforts, with no single dominant production API yet publicly available",
            "*Direct judgment**: This is still a research-to-early-product transition zone, not a mature API market",
            "The blog post frames it as research — no product or public API timeline is mentioned",
            "Cosmos’s production readiness, API accessibility, and whether it provides a public world model API are all unknown from the current evidence",
        ],
        sources=[
            SourceEvidence(
                title="Google DeepMind Genie 2 official world model article",
                url="https://deepmind.google/example",
                snippet="Genie 2 is a research system and a foundation world model.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            )
        ],
        calibration={"ran": True, "critique": "Research-to-product uncertainty was disclosed."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
    )

    item = _evaluation_item(1, "Where are physical AI world models?", package)

    assert item["uncertainty_assessment"] == "justified"
    assert item["quality_flags"] == []


def test_evaluation_treats_live_physical_ai_visible_efforts_answer_as_justified_uncertainty():
    package = AnswerPackage(
        question_id="q-physical-ai-visible-efforts",
        answer_text=(
            "As of mid-2026, there is no single dominant production API or product for AI physics "
            "foundation models, world models, or embodied physical AI. The field remains in a "
            "research-to-early-platform transition zone. The two most visible efforts are Google "
            "DeepMind's Genie series and NVIDIA's Cosmos + Isaac GR00T ecosystem. No benchmark data "
            "or third-party deployment scale is visible in the search evidence.\n\n"
            "NVIDIA Cosmos page content was not retrieved, so its exact capabilities remain unconfirmed. "
            "Isaac GR00T is the closest to a shipped product in the evidence, but it is a platform, "
            "not a single physics world model API.\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [direct-official] Google DeepMind Genie 2 - https://deepmind.google/example\n"
            "  2. [direct-official] NVIDIA Isaac GR00T - https://developer.nvidia.com/isaac/gr00t"
        ),
        classification="research",
        confidence="low",
        verified_claims=[],
        unverified_claims=[
            "As of mid-2026, there is no single dominant production API or product for AI physics foundation models, world models, or embodied physical AI",
            "The two most visible efforts are Google DeepMind's Genie series and NVIDIA's Cosmos + Isaac GR00T ecosystem",
            "No benchmark data (tok/s, latency, simulation fidelity) or third-party deployment scale is visible in the search evidence",
            "NVIDIA Cosmos page content was not retrieved, so exact capabilities are unconfirmed from this evidence",
            "This is the closest to a shipped product in the evidence, but it is a platform, not a single physics world model API",
        ],
        sources=[
            SourceEvidence(
                title="Google DeepMind Genie 2 official world model article",
                url="https://deepmind.google/example",
                snippet="Genie 2 is a large-scale foundation world model and research system.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            ),
            SourceEvidence(
                title="NVIDIA Isaac GR00T official page",
                url="https://developer.nvidia.com/isaac/gr00t",
                snippet="Isaac GR00T foundational technologies support humanoid robot development, simulation, and policy training.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            ),
        ],
        calibration={"ran": True, "critique": "Platform-stage uncertainty was disclosed."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
    )

    item = _evaluation_item(1, "Where are physical AI world models?", package)

    assert item["uncertainty_assessment"] == "justified"
    assert item["quality_flags"] == []


def test_evaluation_treats_chinese_missing_context_bullets_as_justified_uncertainty():
    package = AnswerPackage(
        question_id="q-chinese-live-gap-bullets",
        answer_text=(
            "低置信说明：搜索中未找到以下关键证据，因此只能作为标准潜力判断。"
            "还没有一个统一的、公开可用的生产级API或产品。\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [direct-official] Compute Express Link Consortium About CXL - https://www.computeexpresslink.org/about-cxl/"
        ),
        classification="research",
        confidence="low",
        verified_claims=[],
        unverified_claims=[
            "截至搜索时，AI物理基础模型、世界模型和具身物理AI领域还没有一个统一的、公开可用的生产级API或产品",
            "AI推理或训练场景的实际部署案例、性能基准（延迟、吞吐量）",
            "具体的CXL内存池化实现方案或产品落地信息",
        ],
        sources=[
            SourceEvidence(
                title="Compute Express Link Consortium About CXL official page",
                url="https://www.computeexpresslink.org/about-cxl/",
                snippet="CXL is a cache-coherent interconnect for processors, memory expansion, and accelerators.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            )
        ],
        calibration={"ran": True, "critique": "Caveated missing evidence was disclosed."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
    )

    item = _evaluation_item(1, "What is CXL?", package)

    assert item["uncertainty_assessment"] == "justified"
    assert item["quality_flags"] == []


def test_evaluation_treats_caveated_distributed_mechanism_claims_as_justified_uncertainty():
    package = AnswerPackage(
        question_id="q-distributed-mechanism",
        answer_text=(
            "这是低置信的基础解释：以下内容标注哪些信息来自搜索，哪些是一般性机制解释，"
            "并非全部直接来自搜索证据。分布式推理的节点耦合程度取决于并行策略。\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [direct-official] vLLM parallelism - https://docs.vllm.ai/en/stable/serving/parallelism_scaling/"
        ),
        classification="hard",
        confidence="low",
        verified_claims=[],
        unverified_claims=[
            "**张量并行（Tensor Parallelism）**：同一层的参数切分到不同GPU，每一步前向都需要跨节点同步",
            "**专家并行（Expert Parallelism）**：MoE模型将不同专家分配到不同节点，每个token需要路由到特定专家",
            "**数据并行（Data Parallelism）**：每个节点有完整模型副本，处理不同数据，只周期性同步",
            "它们直接印证了“将模型拆分到多个计算单元，并依赖通信协同”这个基本框架",
        ],
        sources=[
            SourceEvidence(
                title="vLLM official parallelism and scaling documentation",
                url="https://docs.vllm.ai/en/stable/serving/parallelism_scaling/",
                snippet="Distributed inference strategies include tensor parallel and pipeline parallel inference.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            )
        ],
        calibration={"ran": True, "critique": "Caveated mechanism explanation was disclosed."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
    )

    item = _evaluation_item(1, "分布式大模型是否可以理解为多个节点分工推理？", package)

    assert item["uncertainty_assessment"] == "justified"
    assert item["quality_flags"] == []


def test_evaluation_treats_key_assumption_mechanism_claims_as_justified_uncertainty():
    package = AnswerPackage(
        question_id="q-distributed-key-assumption",
        answer_text=(
            "不过“相对独立”的程度取决于具体并行策略。\n\n"
            "## 关键假设\n"
            "- 张量并行每一步前向计算都需要跨节点同步。\n"
            "- 专家并行涉及 token 路由和通信。\n\n"
            "## 推理路径\n"
            "这些是低置信的机制解释，具体通信代价仍需要按框架和硬件验证。\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [direct-official] NVIDIA parallelisms - https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html"
        ),
        classification="hard",
        confidence="low",
        verified_claims=[],
        unverified_claims=[
            "**张量并行**：同一层的参数被切分到不同 GPU，每一步前向计算都需要跨节点同步，高度耦合",
            "**专家并行**：每个 token 需要路由到特定的专家节点，涉及 all-to-all 通信",
        ],
        sources=[
            SourceEvidence(
                title="NVIDIA Megatron Bridge official parallelisms documentation",
                url="https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html",
                snippet="Tensor parallelism, pipeline parallelism, and expert parallelism can be combined.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            )
        ],
        calibration={"ran": True, "critique": "Mechanism caveats were disclosed."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
    )

    item = _evaluation_item(1, "分布式大模型是否可以理解为多个节点协同？", package)

    assert item["uncertainty_assessment"] == "justified"
    assert item["quality_flags"] == []


def test_evaluation_treats_live_strategy_and_framework_assumptions_as_justified_uncertainty():
    package = AnswerPackage(
        question_id="q-live-mechanism-assumptions",
        answer_text=(
            "这是低置信机制解释：结论取决于具体并行策略、推理框架和硬件拓扑。"
            "搜索证据确认了 H100/CXL/分布式推理的基础事实，但没有提供具体 benchmark，"
            "所以下列策略选择和框架假设只作为需要验证的推理路径。\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [direct-official] Official technical source - https://example.com/source"
        ),
        classification="hard",
        confidence="low",
        verified_claims=[],
        unverified_claims=[
            "文档中的策略选择逻辑（模型太大→拆分）支持这个假设",
            "假设推理框架没有额外优化（如连续批处理、权重共享）改变带宽占用",
        ],
        sources=[
            SourceEvidence(
                title="Official technical source",
                url="https://example.com/source",
                snippet="Official source for the searched technical topic.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            )
        ],
        calibration={"ran": True, "critique": "Mechanism assumptions were caveated."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
    )

    item = _evaluation_item(1, "technical mechanism question", package)

    assert item["uncertainty_assessment"] == "justified"
    assert item["quality_flags"] == []


def test_evaluation_treats_live_distributed_llm_mechanism_answer_as_justified_uncertainty():
    package = AnswerPackage(
        question_id="q-distributed-live-mechanism",
        answer_text=(
            "你的基本理解在方向上是成立的：分布式大模型推理确实涉及将模型拆分到多个计算节点，"
            "每个节点承担一部分计算，并通过互联通信协同完成一次推理。"
            "但“相对独立”这个说法需要加上重要的限定：节点之间的耦合程度取决于具体采用的并行策略，"
            "不能一概而论。\n\n"
            "以上策略的基本分类和存在性由搜索文档确认，但各策略的具体通信原语、耦合程度的精确量化、"
            "以及实际硬件带宽和延迟数值在搜索中未提供，因此无法给出具体的量化比较。\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [direct-official] NVIDIA Megatron Bridge official parallelisms documentation - https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html\n"
            "  2. [direct-official] vLLM official parallelism and scaling documentation - https://docs.vllm.ai/en/stable/serving/parallelism_scaling/"
        ),
        classification="hard",
        confidence="low",
        verified_claims=[],
        unverified_claims=[
            "这个框架得到了搜索到的官方文档的印证——vLLM 的分布式策略指南明确说明当模型超出单 GPU 或单节点时，需要结合张量并行和流水线并行跨节点部署，NVIDIA Megatron Bridge 文档也列出了多种并行策略（数据并行、张量并行、流水线并行、MoE 专家并行等），并确认模型副本或拆分后的参数之间需要通信保持一致",
            "**张量并行**（vLLM 文档建议用于单节点多 GPU）：将同一层的参数切分到不同 GPU，每一步前向计算都需要跨 GPU 同步，节点高度耦合",
            "**专家并行**（MoE 模型特有的策略，Megatron Bridge 文档提及）：将不同专家分配到不同节点，每个 token 需要路由到目标专家，通信量大，耦合较高",
            "**数据并行**（Megatron Bridge 文档提及）：每个节点有完整模型副本，独立处理不同数据，在推理时可看作各节点相对独立，但这实际上是多副本部署而非模型拆分，和你问的“各负责一部分”不是同一个场景",
            "以上策略的基本分类和存在性由搜索文档确认，但各策略的具体通信原语、耦合程度的精确量化、以及实际硬件（如网络带宽、延迟数值）在搜索中未提供，因此无法给出具体的量化比较",
        ],
        sources=[
            SourceEvidence(
                title="NVIDIA Megatron Bridge official parallelisms documentation",
                url="https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html",
                snippet="Megatron Bridge supports data parallelism, tensor parallelism, pipeline parallelism, and expert parallelism.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            ),
            SourceEvidence(
                title="vLLM official parallelism and scaling documentation",
                url="https://docs.vllm.ai/en/stable/serving/parallelism_scaling/",
                snippet="For multi-node multi-GPU inference, combine tensor parallelism with pipeline parallelism.",
                provider="direct-official",
                checked_at="2026-07-04T00:00:00Z",
            ),
        ],
        calibration={"ran": True, "critique": "Distributed inference caveats were disclosed."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
    )

    item = _evaluation_item(1, "分布式大模型是否可以理解为多个节点协同？", package)

    assert item["uncertainty_assessment"] == "justified"
    assert item["quality_flags"] == []


def test_evaluation_still_flags_low_confidence_with_chinese_fit_claim():
    package = AnswerPackage(
        question_id="q-unsupported-chinese-fit",
        answer_text=(
            "虽然搜索证据不足，但10台GB10有可能在内存层面装下DeepSeek-V4，并且权重肯定放得下。\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [unit] NVIDIA DGX Spark - https://example.test/dgx-spark"
        ),
        classification="hard",
        confidence="low",
        verified_claims=[],
        unverified_claims=["10台GB10有可能在内存层面装下DeepSeek-V4，并且权重肯定放得下"],
        sources=[
            SourceEvidence(
                title="NVIDIA DGX Spark",
                url="https://example.test/dgx-spark",
                snippet="Official DGX Spark product information.",
                provider="unit",
                checked_at="2026-07-04T00:00:00Z",
            )
        ],
        calibration={"ran": True, "critique": "Unsupported positive fit claim remains."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
    )

    item = _evaluation_item(1, "Can 10 GB10 systems deploy DeepSeek-V4?", package)

    assert item["uncertainty_assessment"] == "needs_attention"
    assert "unverified_claims" in item["quality_flags"]


def test_evaluation_still_flags_low_confidence_with_unsupported_positive_claim():
    package = AnswerPackage(
        question_id="q-unsupported-low-confidence",
        answer_text=(
            "DeepSeek-V4 can definitely run at full precision across ten GB10 systems.\n\n"
            "搜索记录:\n"
            "- 联网搜索: 已执行\n"
            "- 搜索结果:\n"
            "  1. [unit] NVIDIA DGX Spark - https://example.test/dgx-spark"
        ),
        classification="hard",
        confidence="low",
        verified_claims=[],
        unverified_claims=["DeepSeek-V4 can run at full precision across ten GB10 systems"],
        sources=[
            SourceEvidence(
                title="NVIDIA DGX Spark",
                url="https://example.test/dgx-spark",
                snippet="Official DGX Spark product information.",
                provider="unit",
                checked_at="2026-07-03T00:00:00Z",
            )
        ],
        calibration={"ran": True, "critique": "Unsupported positive claim remains."},
        review={"ran": True, "approved": True, "issues": [], "revision": ""},
        memory_updates=[],
    )

    item = _evaluation_item(1, "Can 10 GB10 systems deploy DeepSeek-V4?", package)

    assert item["uncertainty_assessment"] == "needs_attention"
    assert "low_confidence" in item["quality_flags"]
    assert "unverified_claims" in item["quality_flags"]


def test_default_evaluation_questions_include_foundational_distributed_model_check():
    assert len(DEFAULT_EVALUATION_QUESTIONS) == 4
    assert any("分布式大模型" in question for question in DEFAULT_EVALUATION_QUESTIONS)
    assert any("搜索证据不足" in question or "基础解释" in question for question in DEFAULT_EVALUATION_QUESTIONS)


class StaticSearchClient:
    def search(self, query, limit=5):
        return [
            SearchResult(
                title="Feishu message reply API 2026",
                url="https://open.feishu.cn/document/server-docs/im-v1/message/reply",
                snippet="The Feishu message reply API supports replying to received messages from a bot.",
                provider="unit",
                checked_at="2026-06-27T00:00:00Z",
            )
        ][:limit]


class RejectedRuntime(FakeAgentRuntime):
    def generate_answer(self, question, context):
        return "unsafe unsupported original answer"

    def review_answer(self, answer, context):
        return {
            "ran": True,
            "approved": False,
            "issues": ["unsupported model-memory claims"],
            "revision": "unsafe unsupported original answer",
        }
