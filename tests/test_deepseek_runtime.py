import json

import pytest

from search_assistant.config import Settings
from search_assistant.runtime import DeepSeekChatRuntime
from search_assistant.runtime.pydantic_ai import runtime_from_settings


def test_deepseek_runtime_uses_pydantic_ai_runner_contract():
    calls = []

    def agent_runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append(
            {
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
                "instructions": instructions,
                "prompt": json.loads(prompt),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout_seconds,
            }
        )
        return "真实回答：2 + 2 = 4。"

    runtime = DeepSeekChatRuntime(
        api_key="test-key",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        agent_runner=agent_runner,
    )

    answer = runtime.generate_answer(
        "What is 2 + 2?",
        {"classification": "simple", "memory": [], "search_results": []},
    )

    assert answer == "真实回答：2 + 2 = 4。"
    assert calls[0]["model"] == "deepseek-v4-flash"
    assert calls[0]["api_key"] == "test-key"
    assert calls[0]["base_url"] == "https://api.deepseek.com"
    assert "high-accuracy personal search assistant" in calls[0]["instructions"]
    assert "MUST treat search_results as the primary evidence" in calls[0]["instructions"]
    assert "Do not answer factual questions from model memory when search_results are available" in calls[0]["instructions"]
    assert "Only provide conditional deployment estimates" in calls[0]["instructions"]
    assert "do not claim fit, deployability, tok/s, or throughput" in calls[0]["instructions"]
    assert "hardware deployment questions" in calls[0]["instructions"]
    assert "ConnectX-7/CX7" in calls[0]["instructions"]
    assert "ModelScope" in calls[0]["instructions"]
    assert "Only cite URLs present in search_results" in calls[0]["instructions"]
    assert calls[0]["prompt"]["question"] == "What is 2 + 2?"
    assert calls[0]["prompt"]["search_results"] == []
    assert calls[0]["timeout_seconds"] == 60.0


def test_deepseek_calibration_requires_evidence_for_deployment_estimates():
    calls = []

    def agent_runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append(
            {
                "instructions": instructions,
                "prompt": json.loads(prompt),
                "temperature": temperature,
                "timeout_seconds": timeout_seconds,
            }
        )
        return "revised answer"

    runtime = DeepSeekChatRuntime(
        api_key="test-key",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        agent_runner=agent_runner,
    )

    result = runtime.calibrate(
        "No public benchmark exists.",
        {
            "classification": "hard",
            "unverified_claims": ["10 GB10 DeepSeek V4 tok/s"],
            "memory": [],
            "search_results": [{"title": "NVIDIA GB10", "url": "https://example.com", "snippet": "128GB"}],
        },
    )

    assert result["ran"] is True
    assert "Preserve conditional deployment estimates only when search_results contain model parameter evidence" in calls[0]["instructions"]
    assert "remove fit/deployability/tok/s claims" in calls[0]["instructions"]
    assert "Return only the final user-facing revised answer" in calls[0]["instructions"]
    assert calls[0]["prompt"]["unverified_claims"] == ["10 GB10 DeepSeek V4 tok/s"]
    assert calls[0]["temperature"] == 0.1
    assert calls[0]["timeout_seconds"] == 60.0


def test_deepseek_runtime_prompts_reject_unverified_hardware_spec_numbers():
    calls = []

    def agent_runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append({"instructions": instructions, "prompt": json.loads(prompt)})
        if "Return only JSON" in instructions:
            return json.dumps(
                {
                    "approved": True,
                    "issues": ["Removed unsupported bandwidth numbers."],
                    "revision": "Use HBM bandwidth as the mechanism, without unsupported numeric specs.",
                }
            )
        return "Use HBM bandwidth as the mechanism, without unsupported numeric specs."

    runtime = DeepSeekChatRuntime(api_key="test-key", agent_runner=agent_runner)
    context = {
        "classification": "hard",
        "search_results": [
            {
                "title": "NVIDIA H100",
                "url": "https://www.nvidia.com/en-us/data-center/h100/",
                "snippet": "H100 supports FP8 Tensor Cores and higher inference performance.",
            }
        ],
    }

    runtime.generate_answer("为什么H100的tok/s高？", context)
    runtime.calibrate("H100 bandwidth is 3.35 TB/s.", {**context, "unverified_claims": ["3.35 TB/s"]})
    runtime.review_answer("H100 bandwidth is 3.35 TB/s.", {**context, "question": "为什么H100的tok/s高？"})

    assert "Do not introduce hardware spec numbers" in calls[0]["instructions"]
    assert "bandwidth, NVLink, HBM capacity, or tok/s" in calls[0]["instructions"]
    assert "Do not introduce hardware spec numbers" in calls[1]["instructions"]
    assert "Remove or explicitly mark unsupported hardware spec numbers" in calls[2]["instructions"]
    assert "Do not include example spec numbers in next-step verification" in calls[0]["instructions"]
    assert "Do not include example spec numbers in next-step verification" in calls[1]["instructions"]
    assert "Do not include example spec numbers in next-step verification" in calls[2]["instructions"]
    assert "Do not include model-size or latency example calculations" in calls[0]["instructions"]
    assert "Do not include model-size or latency example calculations" in calls[1]["instructions"]
    assert "Do not include model-size or latency example calculations" in calls[2]["instructions"]


def test_deepseek_runtime_prompts_do_not_attribute_unasked_details_to_user():
    calls = []

    def agent_runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append({"instructions": instructions, "prompt": json.loads(prompt)})
        if "Return only JSON" in instructions:
            return json.dumps({"approved": True, "issues": [], "revision": "answer"})
        return "answer"

    runtime = DeepSeekChatRuntime(api_key="test-key", agent_runner=agent_runner)
    context = {
        "question": "分布式大模型是否可以理解为多个节点分工推理？",
        "classification": "hard",
        "search_results": [
            {
                "title": "Distributed inference",
                "url": "https://example.com/distributed",
                "snippet": "Distributed inference uses tensor parallelism and pipeline parallelism.",
            }
        ],
    }

    runtime.generate_answer(context["question"], context)
    runtime.calibrate("draft", context)
    runtime.review_answer("answer", context)

    assert "Do not attribute details" in calls[0]["instructions"]
    assert "unless they appear in the user's question" in calls[0]["instructions"]
    assert "Do not attribute details" in calls[1]["instructions"]
    assert "Do not attribute details" in calls[2]["instructions"]


def test_deepseek_runtime_final_review_agent_checks_answer_against_sources_and_feedback():
    calls = []

    def agent_runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append(
            {
                "instructions": instructions,
                "prompt": json.loads(prompt),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return json.dumps(
            {
                "approved": False,
                "issues": ["Missing ModelScope source coverage."],
                "revision": "reviewed answer with ModelScope caveat",
            }
        )

    runtime = DeepSeekChatRuntime(
        api_key="test-key",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        agent_runner=agent_runner,
    )

    result = runtime.review_answer(
        "draft answer",
        {
            "question": "Can 10 GB10 systems run DeepSeek-V4?",
            "classification": "hard",
            "search_queries": ["DGX Spark official specs"],
            "search_results": [{"title": "DGX Spark", "url": "https://example.com/dgx", "snippet": "128GB"}],
            "experience": [
                {
                    "title": "User feedback: search and answer correction",
                    "body": "Do not miss ModelScope or CX7.",
                }
            ],
        },
    )

    assert result["ran"] is True
    assert result["approved"] is False
    assert result["issues"] == ["Missing ModelScope source coverage."]
    assert result["revision"] == "reviewed answer with ModelScope caveat"
    assert "final answer review agent" in calls[0]["instructions"]
    assert "search_results as the primary evidence" in calls[0]["instructions"]
    assert "user feedback" in calls[0]["instructions"]
    assert "Return only JSON" in calls[0]["instructions"]
    assert calls[0]["prompt"]["answer"] == "draft answer"
    assert calls[0]["prompt"]["experience"][0]["title"] == "User feedback: search and answer correction"
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["max_tokens"] <= 1200


def test_deepseek_runtime_review_retries_once_on_parse_failure():
    calls = []

    def agent_runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append(temperature)
        if len(calls) == 1:
            return "not json at all"
        return json.dumps(
            {
                "approved": True,
                "issues": [],
                "revision": "reviewed answer",
            }
        )

    runtime = DeepSeekChatRuntime(api_key="test-key", agent_runner=agent_runner)

    result = runtime.review_answer(
        "draft answer",
        {
            "question": "What is CXL?",
            "classification": "research",
            "search_queries": ["CXL memory pooling"],
            "search_results": [{"title": "CXL", "url": "https://example.com/cxl", "snippet": "memory pooling"}],
        },
    )

    assert len(calls) == 2, "the review must be retried once after a parse failure"
    assert result["approved"] is True
    assert result["parse_failed"] is False
    assert result["retried_after_parse_failure"] is True
    assert result["revision"] == "reviewed answer"


def test_deepseek_runtime_review_falls_back_when_retry_also_fails():
    calls = []

    def agent_runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append(temperature)
        return "still not json"

    runtime = DeepSeekChatRuntime(api_key="test-key", agent_runner=agent_runner)

    result = runtime.review_answer(
        "draft answer",
        {
            "question": "What is CXL?",
            "classification": "research",
            "search_queries": ["CXL memory pooling"],
            "search_results": [{"title": "CXL", "url": "https://example.com/cxl", "snippet": "memory pooling"}],
        },
    )

    assert len(calls) == 2
    assert result["approved"] is False
    assert result["parse_failed"] is True
    assert "did not return JSON" in result["issues"][0]
    assert result["revision"] == "draft answer"


def test_deepseek_runtime_plans_search_queries_from_intent_and_experience():
    calls = []

    def agent_runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append(
            {
                "instructions": instructions,
                "prompt": json.loads(prompt),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return json.dumps(
            [
                "DGX Spark official specifications interconnect memory",
                "DeepSeek-V3 model card total active parameters",
            ]
        )

    runtime = DeepSeekChatRuntime(
        api_key="test-key",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        agent_runner=agent_runner,
    )

    queries = runtime.plan_search_queries(
        "10台GB10并联跑满血DeepSeek-V4，需要怎么查参数和互联限制？",
        {
            "classification": "hard",
            "memory": [],
            "experience": [
                {
                    "title": "User feedback: search and answer correction",
                    "body": "Do not hardcode keywords; include model hubs and interconnect specs when relevant.",
                }
            ],
        },
    )

    assert queries == [
        "DGX Spark official specifications interconnect memory",
        "DeepSeek-V3 model card total active parameters",
    ]
    assert "infer the user's research intent" in calls[0]["instructions"]
    assert "Do not rely on fixed keyword templates" in calls[0]["instructions"]
    assert "official documentation" in calls[0]["instructions"]
    assert "model hubs" in calls[0]["instructions"]
    assert "accelerator inference throughput" in calls[0]["instructions"]
    assert "tokens per second" in calls[0]["instructions"]
    assert "HBM/HBM3/HBM3e memory bandwidth" in calls[0]["instructions"]
    assert "KV cache" in calls[0]["instructions"]
    assert "decode throughput" in calls[0]["instructions"]
    assert "ModelScope/魔搭/魔塔" in calls[0]["instructions"]
    assert "Hugging Face" in calls[0]["instructions"]
    assert "GitHub" in calls[0]["instructions"]
    assert "user feedback" in calls[0]["instructions"]
    assert calls[0]["prompt"]["experience"][0]["title"] == "User feedback: search and answer correction"
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["max_tokens"] <= 800


def test_deepseek_runtime_requires_model_parameter_evidence_before_deployment_estimates():
    calls = []

    def agent_runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append({"instructions": instructions, "prompt": json.loads(prompt)})
        return "answer"

    runtime = DeepSeekChatRuntime(api_key="test-key", agent_runner=agent_runner)

    runtime.generate_answer(
        "Can 10 GB10 systems deploy DeepSeek-V4?",
        {
            "classification": "hard",
            "search_results": [
                {
                    "title": "DGX Spark",
                    "url": "https://example.com/dgx",
                    "snippet": "128 GB unified memory, ConnectX-7",
                }
            ],
        },
    )

    assert "Only provide conditional deployment estimates" in calls[0]["instructions"]
    assert "model parameter evidence" in calls[0]["instructions"]
    assert "do not claim fit, deployability, tok/s, or throughput" in calls[0]["instructions"]
    assert "When direct benchmarks are missing, still provide a conditional engineering estimate" not in calls[0]["instructions"]


def test_deepseek_runtime_search_planner_accepts_numbered_text_fallback():
    runtime = DeepSeekChatRuntime(
        api_key="test-key",
        agent_runner=lambda *args: "1. first query\n2. second query",
    )

    assert runtime.plan_search_queries("question", {}) == ["first query", "second query"]


def test_deepseek_runtime_search_planner_accepts_fenced_json_array():
    runtime = DeepSeekChatRuntime(
        api_key="test-key",
        agent_runner=lambda *args: '```json\n["first query", "second query"]\n```',
    )

    assert runtime.plan_search_queries("question", {}) == ["first query", "second query"]


def test_deepseek_runtime_review_prompt_requires_safe_rewrites_to_be_approved():
    calls = []

    def agent_runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append({"instructions": instructions, "prompt": json.loads(prompt)})
        return json.dumps(
            {
                "approved": True,
                "issues": ["Original wording was too indirect."],
                "revision": "direct safe revised answer",
            }
        )

    runtime = DeepSeekChatRuntime(api_key="test-key", agent_runner=agent_runner)

    result = runtime.review_answer(
        "indirect draft",
        {
            "question": "How should I verify sources?",
            "classification": "hard",
            "search_results": [{"title": "Source", "url": "https://example.com", "snippet": "evidence"}],
        },
    )

    assert result["approved"] is True
    assert result["revision"] == "direct safe revised answer"
    assert "If you can produce a safe corrected revision" in calls[0]["instructions"]
    assert "set approved=true" in calls[0]["instructions"]


def test_deepseek_runtime_review_accepts_fenced_json_response():
    runtime = DeepSeekChatRuntime(
        api_key="test-key",
        agent_runner=lambda *args: (
            "```json\n"
            '{"approved": true, "issues": [], "revision": "reviewed answer"}'
            "\n```"
        ),
    )

    result = runtime.review_answer(
        "fallback answer",
        {
            "question": "Where are world models now?",
            "classification": "research",
            "search_results": [{"title": "Source", "url": "https://example.com", "snippet": "evidence"}],
        },
    )

    assert result["approved"] is True
    assert result["issues"] == []
    assert result["revision"] == "reviewed answer"


def test_deepseek_runtime_uses_answer_strategy_for_visible_reasoning_without_hidden_cot():
    calls = []

    def agent_runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append({"instructions": instructions, "prompt": json.loads(prompt)})
        return "answer with visible reasoning summary"

    runtime = DeepSeekChatRuntime(api_key="test-key", agent_runner=agent_runner)
    answer_strategy = {
        "mode": "technical_explanation",
        "style": "technical_partner",
        "visible_reasoning": [
            "direct_judgment",
            "key_assumptions",
            "reasoning_path",
            "counterpoints_or_risks",
            "next_verification",
        ],
        "hidden_reasoning_policy": "do_not_reveal_chain_of_thought",
        "max_clarifying_questions": 1,
    }

    runtime.generate_answer(
        "为什么H100的tok/s高？",
        {
            "classification": "hard",
            "answer_strategy": answer_strategy,
            "search_results": [],
        },
    )

    assert calls[0]["prompt"].get("answer_strategy") == answer_strategy
    assert "visible reasoning summary" in calls[0]["instructions"]
    assert "reasoning snapshot" in calls[0]["instructions"]
    assert "not as a mechanical checklist" in calls[0]["instructions"]
    assert "required_sections as reasoning moves" in calls[0]["instructions"]
    assert "Do not reveal hidden chain-of-thought" in calls[0]["instructions"]
    assert "technical partner" in calls[0]["instructions"]
    assert "For discussion or brainstorming mode" in calls[0]["instructions"]
    assert "compare plausible hypotheses" in calls[0]["instructions"]
    assert "challenge the user's assumptions" in calls[0]["instructions"]
    assert "ask at most one clarifying question" in calls[0]["instructions"]


def test_deepseek_runtime_calibration_and_review_preserve_answer_strategy_structure():
    calls = []

    def agent_runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append({"instructions": instructions, "prompt": json.loads(prompt)})
        if "Return only JSON" in instructions:
            return json.dumps({"approved": True, "issues": [], "revision": "reviewed answer"})
        return "calibrated answer"

    runtime = DeepSeekChatRuntime(api_key="test-key", agent_runner=agent_runner)
    answer_strategy = {
        "mode": "brainstorming_discussion",
        "style": "technical_partner",
        "visible_reasoning": ["competing_hypotheses", "pushback", "decision_criteria"],
        "hidden_reasoning_policy": "do_not_reveal_chain_of_thought",
        "max_clarifying_questions": 1,
    }

    runtime.calibrate(
        "draft",
        {
            "classification": "hard",
            "answer_strategy": answer_strategy,
            "search_results": [{"title": "Source", "url": "https://example.com", "snippet": "evidence"}],
        },
    )
    runtime.review_answer(
        "answer",
        {
            "question": "我们讨论一下H100 tok/s机制",
            "classification": "hard",
            "answer_strategy": answer_strategy,
            "search_results": [{"title": "Source", "url": "https://example.com", "snippet": "evidence"}],
        },
    )

    assert calls[0]["prompt"].get("answer_strategy") == answer_strategy
    assert calls[1]["prompt"].get("answer_strategy") == answer_strategy
    assert "preserve the requested visible reasoning structure" in calls[0]["instructions"]
    assert "keep the answer natural rather than mechanical" in calls[0]["instructions"]
    assert "without flattening the answer into a rigid refusal" in calls[0]["instructions"]
    assert "preserve useful visible reasoning structure" in calls[1]["instructions"]
    assert "overly rigid or refusal-only" in calls[1]["instructions"]
    assert "mechanical checklist" in calls[1]["instructions"]
    assert "Do not collapse a structured answer into a terse summary" in calls[1]["instructions"]
    assert "revise inside those sections" in calls[1]["instructions"]


def test_deepseek_runtime_requires_answer_strategy_sections_for_inspectable_reasoning():
    calls = []

    def agent_runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append({"instructions": instructions, "prompt": json.loads(prompt)})
        if "Return only JSON" in instructions:
            return json.dumps({"approved": True, "issues": [], "revision": "sectioned answer"})
        return "sectioned answer"

    runtime = DeepSeekChatRuntime(api_key="test-key", agent_runner=agent_runner)
    answer_strategy = {
        "mode": "technical_explanation",
        "style": "technical_partner",
        "reasoning_contract": "must_show_compact_user_visible_reasoning",
        "required_sections": [
            "direct_judgment",
            "key_assumptions",
            "evidence_check",
            "reasoning_path",
            "counterpoints_or_risks",
            "next_verification",
        ],
        "minimum_reasoning_detail": "explain_mechanism_or_calculation_before_final_recommendation",
        "hidden_reasoning_policy": "do_not_reveal_chain_of_thought",
    }
    context = {
        "question": "Why is H100 tok/s high?",
        "classification": "hard",
        "answer_strategy": answer_strategy,
        "search_results": [{"title": "NVIDIA H100", "url": "https://example.com/h100", "snippet": "HBM3"}],
    }

    runtime.generate_answer("Why is H100 tok/s high?", context)
    runtime.calibrate("draft", context)
    runtime.review_answer("answer", context)

    assert all(call["prompt"].get("answer_strategy") == answer_strategy for call in calls)
    assert "Follow answer_strategy.required_sections" in calls[0]["instructions"]
    assert "Do not answer with only a conclusion" in calls[0]["instructions"]
    assert "Do not replace the answer with a bare source-insufficiency refusal" in calls[0]["instructions"]
    assert "Follow answer_strategy.required_sections" in calls[1]["instructions"]
    assert "Follow answer_strategy.required_sections" in calls[2]["instructions"]
    assert "missing required user-visible sections" in calls[2]["instructions"]


def test_deepseek_runtime_uses_partner_style_temperature_for_discussion_generation():
    calls = []

    def agent_runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append(
            {
                "instructions": instructions,
                "prompt": json.loads(prompt),
                "temperature": temperature,
            }
        )
        return "discussion answer"

    runtime = DeepSeekChatRuntime(api_key="test-key", agent_runner=agent_runner)

    runtime.generate_answer(
        "你怎么看 H100 tok/s 的瓶颈？",
        {
            "classification": "hard",
            "answer_strategy": {
                "mode": "brainstorming_discussion",
                "style": "technical_partner",
                "reasoning_contract": "must_show_compact_user_visible_reasoning",
            },
            "search_results": [{"title": "NVIDIA H100", "url": "https://example.com/h100", "snippet": "HBM3"}],
        },
    )

    assert calls[0]["temperature"] == 0.32
    assert "Avoid a canned report voice" in calls[0]["instructions"]
    assert "show the compact reasoning path" in calls[0]["instructions"]


def test_deepseek_runtime_allows_caveated_foundational_answer_when_search_relevance_is_weak():
    calls = []

    def agent_runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append({"instructions": instructions, "prompt": json.loads(prompt)})
        if "Return only JSON" in instructions:
            return json.dumps({"approved": True, "issues": [], "revision": "caveated conceptual answer"})
        return "caveated conceptual answer"

    runtime = DeepSeekChatRuntime(api_key="test-key", agent_runner=agent_runner)
    context = {
        "question": "分布式大模型是否可以理解为多个节点分工推理？",
        "classification": "hard",
        "search_results": [{"title": "Weak page", "url": "https://example.com/weak", "snippet": "unrelated"}],
        "source_relevance_issue": "搜索结果相关性不足",
        "allow_foundational_fallback": True,
    }

    runtime.generate_answer(context["question"], context)
    runtime.calibrate("draft", context)
    runtime.review_answer("answer", context)

    assert all(call["prompt"]["allow_foundational_fallback"] is True for call in calls)
    assert all(call["prompt"]["source_relevance_issue"] == "搜索结果相关性不足" for call in calls)
    assert "Do not refuse only because search relevance is weak" in calls[0]["instructions"]
    assert "avoid current/numeric/vendor-specific claims" in calls[0]["instructions"]
    assert "Do not defer the answer with phrases like" in calls[0]["instructions"]
    assert "preserve a basic conceptual answer" in calls[1]["instructions"]
    assert "do not defer the answer until more sources are provided" in calls[1]["instructions"]
    assert "approve a caveated foundational concept explanation" in calls[2]["instructions"]
    assert "rewrite source-insufficiency deferrals" in calls[2]["instructions"]


def test_deepseek_runtime_passes_foundational_answer_policy_without_weak_relevance_flag():
    calls = []

    def agent_runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append({"instructions": instructions, "prompt": json.loads(prompt)})
        if "Return only JSON" in instructions:
            return json.dumps({"approved": True, "issues": [], "revision": "conceptual answer"})
        return "conceptual answer"

    runtime = DeepSeekChatRuntime(api_key="test-key", agent_runner=agent_runner)
    policy = "For stable concept questions, do not refuse solely because live search evidence is thin."
    context = {
        "question": "分布式大模型是否可以理解为多个节点分工推理？",
        "classification": "hard",
        "search_results": [
            {
                "title": "分布式大模型推理并行概述",
                "url": "https://example.com/distributed-llm-inference",
                "snippet": "分布式大模型推理可以由多个节点分担模型计算。",
            }
        ],
        "foundational_answer_policy": policy,
    }

    runtime.generate_answer(context["question"], context)
    runtime.calibrate("draft", context)
    runtime.review_answer("answer", context)

    assert all(call["prompt"]["foundational_answer_policy"] == policy for call in calls)
    assert all(call["prompt"]["allow_foundational_fallback"] is False for call in calls)
    assert "If foundational_answer_policy is present" in calls[0]["instructions"]
    assert "If foundational_answer_policy is present" in calls[1]["instructions"]
    assert "If foundational_answer_policy is present" in calls[2]["instructions"]


def test_deepseek_runtime_receives_active_skills_as_reviewed_guidance():
    calls = []

    def agent_runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append({"instructions": instructions, "prompt": json.loads(prompt)})
        if "Return only JSON" in instructions:
            return json.dumps({"approved": True, "issues": [], "revision": "reviewed answer"})
        if "JSON array" in instructions:
            return json.dumps(["H100 HBM bandwidth KV cache"])
        return "answer"

    runtime = DeepSeekChatRuntime(api_key="test-key", agent_runner=agent_runner)
    active_skills = [
        {
            "name": "H100 Throughput Reasoning",
            "path": "skills/active/h100-throughput-reasoning/SKILL.md",
            "content": "Always explain HBM bandwidth, KV cache, prefill/decode split, and interconnect.",
        }
    ]
    context = {
        "classification": "hard",
        "active_skills": active_skills,
        "search_results": [{"title": "NVIDIA H100", "url": "https://example.com/h100", "snippet": "HBM3"}],
    }

    runtime.plan_search_queries("Why is H100 tok/s high?", context)
    runtime.generate_answer("Why is H100 tok/s high?", context)
    runtime.calibrate("draft", context)
    runtime.review_answer("answer", {**context, "question": "Why is H100 tok/s high?"})

    assert all(call["prompt"].get("active_skills") == active_skills for call in calls)
    assert "Use active_skills as reviewed local guidance" in calls[0]["instructions"]
    assert "Use active_skills as reviewed local guidance" in calls[1]["instructions"]
    assert "Use active_skills as reviewed local guidance" in calls[2]["instructions"]
    assert "Use active_skills as reviewed local guidance" in calls[3]["instructions"]


def test_deepseek_runtime_passes_configured_timeout_to_agent_runner():
    calls = []

    def agent_runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append(timeout_seconds)
        return "answer"

    runtime = DeepSeekChatRuntime(
        api_key="test-key",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        timeout_seconds=12.5,
        agent_runner=agent_runner,
    )

    assert runtime.generate_answer("x", {"search_results": []}) == "answer"
    assert calls == [12.5]


def test_runtime_from_settings_requires_real_glm_key_by_default(tmp_path):
    settings = Settings.from_env({"SEARCH_ASSISTANT_DATA_DIR": str(tmp_path)})

    with pytest.raises(RuntimeError, match="GLM_API_KEY"):
        runtime_from_settings(settings)


def test_runtime_from_settings_allows_fake_only_when_explicit(tmp_path):
    settings = Settings.from_env(
        {
            "SEARCH_ASSISTANT_DATA_DIR": str(tmp_path),
            "SEARCH_ASSISTANT_MODEL_PROVIDER": "fake",
            "SEARCH_ASSISTANT_ALLOW_FAKE_RUNTIME": "true",
        }
    )

    runtime = runtime_from_settings(settings)

    assert runtime.generate_answer("x", {}) == "This is a local deterministic answer."
