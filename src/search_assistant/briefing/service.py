from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, date, datetime
import json
from pathlib import Path
import re
import time
import uuid
from typing import Protocol
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from search_assistant.contracts import (
    BriefingSynthesis,
    BriefingTheme,
    BriefingUnderstanding,
    CollectedSource,
    DailyBriefing,
    ProviderTraceEvent,
    SourceCandidate,
    SourceQualityVerdict,
    TopicSubscription,
    default_briefing_understanding,
    default_source_quality_verdict,
    normalize_briefing_understanding,
    normalize_source_quality_verdict,
)
from search_assistant.evolution.service import (
    DomainKnowledgeCandidateService,
    _PRIMARY_SOURCE_WEIGHT,
    _WEAK_SOURCE_WEIGHT,
    source_authority_weight,
)
from search_assistant.knowledge_graph.embedding import (
    EmbeddingProvider,
    build_embedding_provider,
    cosine_similarity,
    _embedding_cache_key,
)
from search_assistant.knowledge_graph.extractor import extend_graph_from_briefing, is_meaningful_entity_name
from search_assistant.knowledge_graph.service import DomainKnowledgeGraphService
from search_assistant.memory.store import MemoryStore
from search_assistant.search.provider import SearchClient, SearchOutcome, SearchResult, search_with_provider_events
from search_assistant.search.source_registry import (
    normalize_source_recipe,
    source_contract_for_query,
    source_recipe_summary,
)


class BriefingRuntime(Protocol):
    def run_text_task(
        self,
        instructions: str,
        payload: dict[str, object],
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
    ) -> str:
        ...

    def understand_briefing(self, topic: str, context: dict[str, object]) -> BriefingUnderstanding:
        ...

    def plan_briefing_queries(self, topic: str, context: dict[str, object]) -> list[str]:
        ...

    def synthesize_briefing(
        self,
        topic: str,
        sources: list[CollectedSource],
        context: dict[str, object],
    ) -> BriefingSynthesis | None:
        ...


@dataclass(frozen=True)
class BriefingCollectionTrace:
    sources: list[CollectedSource]
    source_candidates: list[SourceCandidate]
    provider_events: list[ProviderTraceEvent]


@dataclass(frozen=True)
class BriefingSearchTask:
    index: int
    requested_platform: str
    query: str
    source_slug: str
    priority: int
    timeout_seconds: float
    budget_share: float

    @property
    def tier(self) -> str:
        return f"tier-{self.priority}"


REPORT_CONTRACT = {
    "language": "zh-CN",
    "required_sections": [
        "搜索方向",
        "关键词",
        "搜索内容总结",
        "简短总结",
        "详细总结",
        "AI 分析判断",
        "下一步搜索方向",
        "落地建议",
        "原文链接",
    ],
    "detail_sections": "模型按本轮证据自行组织 2 至 5 个分析块，不使用固定技术地图模板。",
    "readability": {
        "keep_existing_headings": True,
        "target_total_chars": "3500-4500",
        "short_summary_chars": "120-220",
        "detail_block_paragraphs": "1-2",
        "detail_paragraph_chars": "160-220",
        "preserve_original_urls": True,
        "forbidden_fixed_labels": ["结论：", "依据：", "意义："],
    },
}

REPORT_SKILL_PATH = Path(__file__).resolve().parents[3] / "skills" / "content-collection-report" / "SKILL.md"


SOURCE_QUALITY_JUDGE_INSTRUCTIONS = (
    "You are a source-quality boundary judge for a technology-intelligence briefing. "
    "Judge only from the supplied title, snippet, url/domain, provider, query, topic, and briefing_understanding_summary. "
    "Do not use outside knowledge, do not browse, and do not decide provider permissions, budgets, or final ranking. "
    "Return ONLY one valid JSON object with keys: technical_value, source_type, marketing_level, "
    "evidence_density, authority, keep_recommendation, rationale. "
    "technical_value must be high, medium, or low. source_type must be primary, secondary, community, "
    "marketing, aggregator, or unknown. marketing_level must be none, mixed, or dominant. "
    "evidence_density must be high, medium, or low. authority must be primary, secondary, or weak. "
    "keep_recommendation must be keep, borderline, or reject. "
    "Important boundaries: a marketing-style title alone is not a rejection when the snippet contains concrete "
    "technical mechanisms, data, benchmarks, configurations, or reproduction details; weak authority does not "
    "mean invalid; mixed marketing does not erase technical value; community discussions can have high technical "
    "value when they include versions, bugs, configuration, or reproducible evidence. Pure lead-generation, "
    "training enrollment, empty press copy, and aggregation/search pages should be rejected or borderline with low "
    "technical value depending on the visible evidence."
)


def _parse_source_quality_verdict(raw: str) -> SourceQualityVerdict | None:
    stripped = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    return normalize_source_quality_verdict(parsed)


@dataclass(frozen=True)
class BriefingIntentProfile:
    primary_intent: str
    label: str
    summary: str
    search_focus: tuple[str, ...]
    synthesis_focus: tuple[str, ...]
    keyword_hints: tuple[str, ...]
    query_templates: tuple[str, ...]

    def model_dump(self) -> dict[str, object]:
        return {
            "primary_intent": self.primary_intent,
            "label": self.label,
            "summary": self.summary,
            "search_focus": list(self.search_focus),
            "synthesis_focus": list(self.synthesis_focus),
        }


_INTENT_PROFILES: dict[str, BriefingIntentProfile] = {
    "concept_explanation": BriefingIntentProfile(
        primary_intent="concept_explanation",
        label="概念解释",
        summary="用户更像是在问一个概念、术语或产品是什么，报告应先讲清定义、边界和常见语境。",
        search_focus=("定义与官方说明", "概念边界", "典型使用场景", "易混淆对象"),
        synthesis_focus=("先用通俗语言解释概念", "区分不同语境下的含义", "避免直接扩展成产业综述"),
        keyword_hints=("定义", "概念边界", "典型语境", "易混淆点"),
        query_templates=(
            "{topic} definition concept technical context",
            "{topic} official documentation architecture overview",
            "{topic} use cases terminology common misconceptions",
        ),
    ),
    "technical_tracking": BriefingIntentProfile(
        primary_intent="technical_tracking",
        label="技术追踪",
        summary="用户更像是在跟踪某项技术，报告应关注实现路线、论文/开源、评测和部署边界。",
        search_focus=("技术架构", "论文与开源", "评测基准", "部署接口"),
        synthesis_focus=("提炼实现机制", "说明可验证证据", "指出评测和工程边界"),
        keyword_hints=("技术架构", "开源项目", "论文", "评测", "部署"),
        query_templates=(
            "{topic} technical architecture implementation evaluation",
            "{topic} open source repository paper dataset benchmark",
            "{topic} deployment workflow data interface case study",
        ),
    ),
    "industry_trend": BriefingIntentProfile(
        primary_intent="industry_trend",
        label="产业趋势",
        summary="用户更像是在看产业发展，报告应把政策、市场、企业动作和真实落地证据分开讲。",
        search_focus=("政策与市场", "产业链与生态", "公司/产品动作", "真实落地案例"),
        synthesis_focus=("区分宏观信号和工程证据", "避免把宣传材料当成落地结论", "给出产业判断边界"),
        keyword_hints=("政策", "产业链", "市场规模", "企业动作", "落地案例"),
        query_templates=(
            "{topic} policy market industry chain report",
            "{topic} company product release investment adoption",
            "{topic} case study deployment metrics ecosystem",
        ),
    ),
    "engineering_landing": BriefingIntentProfile(
        primary_intent="engineering_landing",
        label="工程落地",
        summary="用户更像是在找怎么落地，报告应关注架构、接口、数据流、验收指标和失败回滚。",
        search_focus=("系统架构", "接口与数据流", "部署验证", "回滚与运维"),
        synthesis_focus=("还原端到端链路", "明确集成点和约束", "优先输出可执行建议"),
        keyword_hints=("系统架构", "接口", "数据流", "验证", "回滚"),
        query_templates=(
            "{topic} architecture integration workflow data interface",
            "{topic} deployment operations validation rollback",
            "{topic} case study implementation metrics",
        ),
    ),
    "comparison_decision": BriefingIntentProfile(
        primary_intent="comparison_decision",
        label="对比选型",
        summary="用户更像是在比较方案，报告应给出差异、取舍、适用场景和选型判断。",
        search_focus=("方案差异", "评测对比", "迁移成本", "适用场景"),
        synthesis_focus=("按决策维度比较", "说明证据支持和缺口", "给出低风险选择建议"),
        keyword_hints=("对比", "差异", "适用场景", "选型", "限制"),
        query_templates=(
            "{topic} comparison architecture tradeoffs benchmark",
            "{topic} vs official documentation differences",
            "{topic} migration decision criteria limitations",
        ),
    ),
}


_DEFAULT_BRIEFING_INTENT = _INTENT_PROFILES["technical_tracking"]


def _intent_profile_from_understanding(understanding: BriefingUnderstanding) -> BriefingIntentProfile:
    return _INTENT_PROFILES.get(understanding.primary_intent, _DEFAULT_BRIEFING_INTENT)


def _understanding_from_legacy_intent(topic: str, intent: BriefingIntentProfile) -> BriefingUnderstanding:
    return normalize_briefing_understanding(
        topic,
        {
            "primary_intent": intent.primary_intent,
            "secondary_intents": [],
            "temporal_focus": "current",
            "research_focus": list(intent.search_focus) or [topic],
            "evidence_preferences": [
                "official documentation",
                "technical architecture",
                "papers and benchmarks",
                "open source repositories",
                "deployment evidence",
            ],
            "comparison_dimensions": list(intent.synthesis_focus),
            "domain_profile": {
                "domain": topic,
                "key_concepts": [topic],
                "subtopics": list(intent.search_focus),
                "ambiguous_terms": [],
                "excluded_meanings": [],
                "evidence_anchors": [],
                "preferred_source_types": [
                    "official documentation",
                    "research papers",
                    "benchmarks",
                    "repositories",
                    "case studies",
                ],
            },
        },
    )


def _briefing_understanding_metadata(understanding: BriefingUnderstanding) -> dict[str, object]:
    profile = understanding.domain_profile
    return {
        "primary_intent": understanding.primary_intent,
        "secondary_intents": understanding.secondary_intents,
        "temporal_focus": understanding.temporal_focus,
        "domain": profile.domain,
        "research_focus_count": len(understanding.research_focus),
        "evidence_anchor_count": len(profile.evidence_anchors),
        "comparison_dimension_count": len(understanding.comparison_dimensions),
    }


# Legacy compatibility only. The daily briefing main path uses
# BriefingUnderstanding from the runtime and must not call this keyword matcher.
def _has_intent_marker(text: str, compact_text: str, markers: tuple[str, ...]) -> bool:
    for marker in markers:
        normalized_marker = marker.lower()
        compact_marker = normalized_marker.replace(" ", "")
        if normalized_marker in text or compact_marker in compact_text:
            return True
    return False


def _topic_terms(topic: str) -> list[str]:
    terms = re.findall(r"[A-Za-z][A-Za-z0-9+._/-]{1,}|[\u4e00-\u9fff]{2,}", topic.lower())
    compact = re.sub(r"\s+", "", topic.lower())
    if compact and compact not in terms and len(compact) >= 2:
        terms.insert(0, compact)
    return list(dict.fromkeys(term for term in terms if _is_meaningful_query(term)))


def _classify_briefing_intent(
    topic: str,
    feedback: list[dict[str, object]] | None = None,
) -> BriefingIntentProfile:
    feedback_text = " ".join(str(item.get("body") or "") for item in (feedback or [])[:3])
    text = f"{topic} {feedback_text}".lower()
    compact_text = re.sub(r"\s+", "", text)

    if _has_intent_marker(
        text,
        compact_text,
        (
            "对比",
            "比较",
            "区别",
            "差异",
            "哪个",
            "选型",
            "vs",
            "versus",
            "compare",
            "comparison",
            "tradeoff",
            "better",
            "which",
        ),
    ):
        return _INTENT_PROFILES["comparison_decision"]
    if _has_intent_marker(
        text,
        compact_text,
        (
            "是什么",
            "什么是",
            "解释",
            "概念",
            "入门",
            "介绍一下",
            "what is",
            "define",
            "definition",
            "meaning",
        ),
    ):
        return _INTENT_PROFILES["concept_explanation"]
    if _has_intent_marker(
        text,
        compact_text,
        (
            "部署",
            "接入",
            "落地",
            "架构",
            "方案",
            "工作流",
            "接口",
            "集成",
            "生产",
            "workflow",
            "architecture",
            "deployment",
            "deploy",
            "integration",
            "implementation",
            "case study",
        ),
    ):
        return _INTENT_PROFILES["engineering_landing"]
    if _has_intent_marker(
        text,
        compact_text,
        (
            "产业",
            "发展",
            "趋势",
            "政策",
            "市场",
            "投融资",
            "公司",
            "生态",
            "industry",
            "market",
            "trend",
            "policy",
            "investment",
            "business",
            "development",
        ),
    ):
        return _INTENT_PROFILES["industry_trend"]
    if _has_intent_marker(
        text,
        compact_text,
        (
            "技术",
            "论文",
            "模型",
            "大模型",
            "开源",
            "评测",
            "benchmark",
            "paper",
            "repository",
            "open source",
            "model",
            "dataset",
        ),
    ):
        return _INTENT_PROFILES["technical_tracking"]
    return _DEFAULT_BRIEFING_INTENT


CHANNEL_QUERIES: tuple[tuple[str, str], ...] = (
    ("全球网页", "{topic} AI industry technology product release research"),
    ("中文网页", "{topic} 人工智能 技术 产业 落地 发布"),
    ("微信公众号（百度公开索引）", "site:mp.weixin.qq.com {topic}"),
    ("抖音", "site:douyin.com {topic}"),
    ("哔哩哔哩", "site:bilibili.com {topic}"),
    ("今日头条（百度公开索引）", "site:toutiao.com {topic}"),
    ("小红书（百度公开索引）", "site:xiaohongshu.com {topic}"),
    ("小红书分享链接（百度公开索引）", "site:xhslink.com {topic}"),
    ("知乎", "site:zhihu.com {topic}"),
    ("LinkedIn", "site:linkedin.com {topic} AI"),
    ("X", "site:x.com {topic} AI"),
    ("Reddit", "site:reddit.com {topic} AI"),
    ("YouTube", "site:youtube.com {topic} AI"),
    ("工程社区", "site:infoq.cn OR site:csdn.net OR site:36kr.com {topic}"),
)


THEMES: tuple[tuple[str, tuple[str, ...], str, str, str], ...] = (
    (
        "工业智能体与生产协同",
        ("agent", "智能体", "mes", "erp", "scada", "plc", "调度", "工单"),
        "把自然语言任务、生产约束和企业系统接口连接起来的 Agent 编排层。",
        "关键不在聊天，而在受控工具调用、权限、工单闭环和人工接管。",
        "已出现明确场景，但跨系统权限、审计和可靠执行仍决定能否规模化。",
    ),
    (
        "数字孪生、仿真与虚拟调试",
        ("digital twin", "数字孪生", "simulation", "仿真", "virtual commissioning", "虚拟调试"),
        "用设备、产线或工厂的可计算模型在物理部署前验证工艺和控制策略。",
        "核心是模型输入、约束求解、what-if 推演和与现场数据同步，不是可视化大屏。",
        "工程软件和平台侧较成熟，和实时生产数据、控制系统的闭环仍需验证。",
    ),
    (
        "机器视觉质检与质量追溯",
        ("aoi", "machine vision", "机器视觉", "视觉", "质检", "缺陷", "质量"),
        "从缺陷识别延伸到原因定位、批次追溯和工艺参数回写。",
        "核心技术包括多模态视觉、异常检测、缺陷样本闭环和质量数据关联。",
        "视觉识别已较常见，跨批次泛化和与工艺调参闭环仍是难点。",
    ),
    (
        "预测性维护与设备优化",
        ("predictive maintenance", "预测性维护", "设备维护", "停机", "oee", "参数调优"),
        "将传感器时序、故障模式和维修工单串成提前预警与执行闭环。",
        "价值来自时间序列异常检测、剩余寿命估计、因果排查和维修调度联动。",
        "局部设备预测已可落地，数据质量和跨设备迁移决定覆盖范围。",
    ),
    (
        "工业软件 Copilot 与工程自动化",
        ("copilot", "hmi", "cad", "cae", "ansys", "工业软件", "工程"),
        "把需求、规范和工程资产转为辅助建模、代码生成、仿真配置或控制界面。",
        "关键是领域知识检索、结构化工程资产、可验证生成和版本控制。",
        "目前多为工程师增强工具，进入安全关键控制环节前需要严格验证。",
    ),
)


GENERIC_EVIDENCE_THEMES: tuple[tuple[str, tuple[str, ...], str, str, str], ...] = (
    (
        "政策、规模与产业链信号",
        (
            "政策", "规划", "行动计划", "产业链", "规模", "报告", "白皮书", "全景分析",
            "万亿", "亿元", "市场", "产业", "发布", "大会", "waic", "naai",
        ),
        "把政策文件、产业规模、产业链梳理和大会/报告类材料视为环境信号，用来判断资金、监管、供给链和需求侧正在如何给技术落地设定边界。",
        "它能说明行业为什么加速投入、哪些环节被反复强调，但不能直接证明某个系统已经形成可复制的工程闭环。",
        "宏观信号阶段；需要继续核验原始政策文本、统计口径、企业案例和技术指标之间是否相互支撑。",
    ),
    (
        "技术底座、数据与开源生态",
        (
            "dataset", "benchmark", "github", "hugging face", "modelscope", "open source", "paper",
            "repository", "数据集", "评测", "基准", "开源", "论文", "模型", "框架", "算力",
            "api", "数据", "架构",
        ),
        "把数据集、评测基准、模型/框架、开源仓库和工程接口视为技术底座，用来判断产业发展是否具备可复现、可评估和可集成的基础。",
        "这类来源决定日报能不能从口号进入实现层：是否有可获得的数据、可比较的指标、可复用的软件组件和清晰的系统接口。",
        "技术线索阶段；需要补查原始论文、模型卡、仓库 README、benchmark 定义和真实部署约束。",
    ),
    (
        "应用落地与业务转型案例",
        (
            "应用", "落地", "实践", "案例", "场景", "解决方案", "客户", "部署", "转型",
            "升级", "生产", "制造", "医疗", "教育", "金融", "政务", "工业", "工作流",
        ),
        "把行业案例和业务流程描述还原为输入数据、模型或规则处理、系统接口、人工复核和结果写回的工作流假设。",
        "它能帮助识别哪些环节已经从演示进入业务流程，也能暴露公开材料尚未说明的审批、回滚、异常接管和量化收益。",
        "案例线索阶段；没有现场指标、接口说明和失败处理机制前，不应把宣传材料直接视为规模化能力。",
    ),
    (
        "教育传播、公众讨论与弱证据线索",
        (
            "教程", "课程", "纪录片", "科普", "入门", "讲完", "学习", "培训", "视频",
            "论坛", "人山人海", "观点", "讨论", "评论", "bilibili", "youtube",
        ),
        "把课程、纪录片、公开视频和公众讨论作为关注度与知识扩散信号，而不是直接作为技术实现证据。",
        "这类来源能说明话题热度、人才供给和概念传播，但通常缺少系统架构、数据接口、评测指标和部署边界。",
        "弱证据阶段；适合引出下一轮检索方向，不适合单独支撑技术结论。",
    ),
    (
        "综合产业动态与待核验证据",
        (),
        "把无法归入明确技术或政策类别的公开材料先作为综合线索保留，再等待原始技术文档、案例指标或代码材料校验。",
        "它避免遗漏潜在线索，但在证据等级上低于论文、产品文档、开源仓库和带指标的落地案例。",
        "待核验线索阶段；下一步应回到原始来源、技术材料和可量化结果。",
    ),
)


CAD_THEMES: tuple[tuple[str, tuple[str, ...], str, str, str], ...] = (
    (
        "生成式 CAD 与参数化建模",
        (
            "text-to-cad", "text to cad", "cad generation", "cad copilot", "parametric cad",
            "参数化建模", "cad生成", "文本到cad", "特征建模", "草图约束",
        ),
        "把自然语言、草图或参考图转换为可编辑的草图、特征树和参数，而不是只生成不可修改的三维网格。",
        "工程可用性的分界线在于生成结果能否保留尺寸、约束、特征依赖和版本历史，从而进入后续修改与制造流程。",
        "文本和图像到概念几何已较活跃；稳定生成可编辑的参数化实体仍需要约束求解、特征有效性校验和人工复核。",
    ),
    (
        "二维工程图理解、矢量化与图纸生成",
        (
            "engineering drawing", "2d drawing", "technical drawing", "drawing generation", "drawing understanding",
            "dwg", "dxf", "vectorization", "图纸生成", "工程图", "二维图", "矢量化", "标注识别",
        ),
        "将扫描图、PDF 或二维草图解析为几何、尺寸、标注和图层语义，再生成可编辑的二维 CAD 图纸。",
        "它决定存量图纸能否进入结构化工程数据链路，也决定自动出图是否能通过尺寸、公差和标准图框校核。",
        "OCR 与线条矢量化已有成熟组件；尺寸语义、符号标准和复杂装配图的拓扑还原仍需领域规则参与。",
    ),
    (
        "三维模型到二维工程图的自动出图",
        (
            "3d to 2d", "3d-to-2d", "drawing extraction", "orthographic projection", "section view",
            "投影视图", "三维转二维", "自动出图", "剖视图", "三视图", "尺寸标注",
        ),
        "从 B-Rep 或参数化三维模型中选择视角，生成主视图、剖视图、隐藏线、尺寸和 BOM 关联的二维交付物。",
        "该路线把三维设计数据直接复用于加工、检验和供应链沟通，减少人工制图时的视图漏选与版本漂移。",
        "常规投影和出图能力已存在于 CAD 软件；AI 的价值主要在视图选择、标注建议和规则冲突提示，必须由制图标准校验。",
    ),
    (
        "几何表示、约束校验与制造可用性",
        (
            "b-rep", "brep", "boundary representation", "geometric constraint", "constraint solving",
            "gdt", "gd&t", "tolerance", "几何约束", "公差", "边界表示", "可制造性",
        ),
        "围绕 B-Rep、草图约束、拓扑一致性和公差规则建立生成后的可验证层，而不是把模型输出直接当成工程交付。",
        "这层校验决定 AI 图纸能否被下游 CAD、CAM、CAE 和质量体系消费，也是防止貌似正确但不可制造几何的关键。",
        "几何内核和规则校验可复用，但把学习模型输出稳定映射到可求解、可制造的实体仍处于工程化整合阶段。",
    ),
)

_CAD_TOPIC_MARKERS = (
    "cad", "工程图", "二维", "三维", "2d", "3d", "dwg", "dxf", "b-rep", "brep",
    "参数化", "建模", "制图", "投影", "图纸", "草图", "公差",
)
_CAD_EVIDENCE_ANCHORS = (
    "cad", "工程图", "dwg", "dxf", "b-rep", "brep", "参数化", "草图约束", "几何约束",
    "尺寸标注", "公差", "投影视图", "剖视图", "三视图", "机械制图", "实体建模",
)


_PLATFORM_TERMS = {
    "抖音", "哔哩哔哩", "b站", "微信公众号", "微信", "今日头条", "头条", "小红书", "知乎",
    "linkedin", "youtube", "reddit", "发布在", "原文链接",
}
_NOISE_TERMS = {"已经收获", "先问问", "你的手和脑", "请关注", "点击查看", "完整视频"}
_MARKETING_CONTENT_DOMAINS = {
    "mp.weixin.qq.com",
    "weixin.qq.com",
    "toutiao.com",
    "xiaohongshu.com",
    "xhslink.com",
    "baijiahao.baidu.com",
    "sohu.com",
}
_MARKETING_STRONG_PHRASES = (
    "广告合作",
    "本文为广告",
    "商业推广",
    "商务合作",
    "商业合作",
    "软文",
    "赞助商",
    "招商加盟",
    "代理加盟",
    "限时优惠",
    "限时福利",
    "免费领取",
    "扫码领取",
    "扫码加",
    "加微信",
    "联系微信",
    "私信领取",
    "私信进群",
    "领取资料",
    "点击购买",
    "购买链接",
    "优惠券",
    "训练营",
    "报名入口",
    "课程报名",
    "带货",
    "引流",
    "裂变",
    "涨粉",
    "变现",
    "advertorial",
    "sponsored",
    "business cooperation",
    "commercial cooperation",
    "scan qr",
    "add wechat",
    "limited offer",
    "limited discount",
    "coupon",
    "buy now",
    "training camp",
    "course enrollment",
)
_MARKETING_WEAK_PHRASES = (
    "重磅福利",
    "速看",
    "震惊",
    "必看",
    "看完就懂",
    "一文看懂",
    "保姆级",
    "干货满满",
    "建议收藏",
    "点赞关注",
    "关注不迷路",
    "转发收藏",
    "评论区",
    "错过再等一年",
    "全网最全",
    "你还不知道",
    "must read",
    "save this",
    "follow us",
)
_SOURCE_EVIDENCE_PHRASES = (
    "paper",
    "论文",
    "arxiv",
    "github",
    "benchmark",
    "基准",
    "dataset",
    "数据集",
    "white paper",
    "official",
    "官方",
    "release",
    "源码",
    "architecture",
    "架构",
    "evaluation",
    "评测",
    "case study",
    "案例",
    "接口",
    "模型",
    "系统",
    "算法",
    "部署",
    "开源",
)

# Tighter phrases used only to classify which collected sources are primary
# evidence for the model-source quota (papers, repositories, model hubs,
# benchmarks, official docs).  The broad phrase list above stays for relevance.
_PRIMARY_EVIDENCE_PHRASES = (
    "paper",
    "论文",
    "arxiv",
    "github",
    "repository",
    "repo",
    "源码",
    "开源",
    "benchmark",
    "基准",
    "dataset",
    "数据集",
    "white paper",
    "白皮书",
    "official documentation",
    "官方文档",
    "release notes",
    "model card",
    "模型卡",
    "technical report",
    "技术报告",
)


def _is_primary_evidence_source(source: CollectedSource) -> bool:
    """A source counts as primary evidence when its domain is authoritative
    (papers, repositories, model hubs) or its title/snippet carries primary
    evidence phrases such as paper/repository/benchmark/official docs."""
    if source_authority_weight(source.url) >= _PRIMARY_SOURCE_WEIGHT:
        return True
    text = f"{source.title} {source.snippet}".lower()
    return any(phrase in text for phrase in _PRIMARY_EVIDENCE_PHRASES)


def _is_weak_evidence_source(source: CollectedSource) -> bool:
    """A source is weak when it comes from a marketing-funnel domain or its
    text carries strong marketing phrases (广告, 免费领取, 加微信, ...)."""
    if source_authority_weight(source.url) <= _WEAK_SOURCE_WEIGHT:
        return True
    text = f"{source.title} {source.snippet}".lower()
    return any(phrase in text for phrase in _MARKETING_STRONG_PHRASES)


_GENERIC_REFERENCE_DOMAINS = {
    "amap.com",
    "baike.baidu.com",
    "hanyuguoxue.com",
    "iciba.com",
    "jingyan.baidu.com",
    "zhidao.baidu.com",
    "mayoclinic.org",
    "wikipedia.org",
}
_SEARCH_DUMP_MARKERS = (
    "not_struct",
    "renderflags",
    "cosmicpkgs",
    "pagestyleupgrade",
    "\"hastop\"",
    "\"isencoding\"",
    "-->",
)
_PROMPT_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "system prompt",
    "developer message",
    "follow these instructions instead",
)
_CAD_MEDICAL_MARKERS = (
    "coronary artery disease",
    "cardiovascular disease",
    "heart disease",
    "冠状动脉疾病",
    "心血管疾病",
)
_LOGIN_PATH_MARKERS = ("/login", "/signin", "/sign-in", "/auth/login", "/account/login")
_QUERY_STOPWORDS = {
    "and",
    "for",
    "from",
    "global",
    "industry",
    "product",
    "release",
    "research",
    "site",
    "technology",
    "the",
    "with",
}
_TECHNICAL_SIGNAL_MARKERS = (
    "industrial ai",
    "manufacturing ai",
    "\u5de5\u4e1aai",
    "\u5de5\u4e1a\u4eba\u5de5\u667a\u80fd",
    "\u667a\u80fd\u5236\u9020",
    "\u4e91\u8fb9\u534f\u540c",
    "\u8fb9\u7f18\u63a8\u7406",
    "\u65f6\u5e8f\u6570\u636e",
    "\u5f02\u5e38\u68c0\u6d4b",
    "\u4f20\u611f\u5668",
    "robotics",
    "\u673a\u5668\u4eba",
)
_INDUSTRIAL_ANCHORS = (
    "industrial",
    "manufactur",
    "factory",
    "maintenance",
    "\u5de5\u4e1a",
    "\u5236\u9020",
    "\u5de5\u5382",
    "\u8bbe\u5907",
    "\u8bbe\u5907\u7ef4\u62a4",
    "\u9884\u6d4b\u6027\u7ef4\u62a4",
    "\u8d28\u68c0",
    "\u7f3a\u9677",
)
_INDUSTRIAL_ACRONYM_ANCHORS = ("mes", "erp", "scada", "plc")


def _clean_planned_queries(queries: object) -> list[str]:
    if not isinstance(queries, list):
        return []
    cleaned: list[str] = []
    for raw_query in queries:
        query = " ".join(str(raw_query).replace("site:", "").split())
        if not _is_meaningful_query(query):
            continue
        if query.lower() in {item.lower() for item in cleaned}:
            continue
        cleaned.append(query[:180])
        if len(cleaned) >= 8:
            break
    return cleaned


def _is_meaningful_feedback(value: str) -> bool:
    return _is_meaningful_query(value) and len(value) >= 6


def _is_meaningful_query(value: str) -> bool:
    normalized = " ".join(value.split()).strip()
    if len(normalized) < 3:
        return False
    lowered = normalized.lower()
    if any(term in lowered for term in _PLATFORM_TERMS | _NOISE_TERMS):
        return False
    return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", normalized))


def _load_report_skill() -> str:
    try:
        return REPORT_SKILL_PATH.read_text(encoding="utf-8")
    except OSError:
        return "Follow the fixed Chinese content collection report contract and synthesize technology themes, not search logs."


def _has_substantive_detailed_summary(value: str) -> bool:
    text = value.strip()
    if len(text) < 280:
        return False
    heading_count = len(re.findall(r"^###\s+\S", text, flags=re.MULTILINE))
    if heading_count >= 2:
        return True
    return heading_count == 1 and len(text) >= 500


def _is_substantive_synthesis(synthesis: BriefingSynthesis) -> bool:
    if not synthesis.themes or len(synthesis.analysis_judgment.strip()) < 60:
        return False

    # New model route: the detailed narrative owns the shape of the analysis.
    # Themes are evidence anchors, not a mandatory per-theme form.
    if synthesis.detailed_summary.strip():
        return (
            len(synthesis.short_summary.strip()) >= 75
            and _has_substantive_detailed_summary(synthesis.detailed_summary)
            and all(
                len(theme.analysis.strip()) >= 30 and bool(theme.source_urls)
                for theme in synthesis.themes
            )
        )

    # Accept older model responses while the deterministic fallback remains
    # available, but do not let this legacy shape dictate new generations.
    if len(synthesis.short_summary.strip()) < 60:
        return False
    if len(synthesis.key_signal_interpretation.strip()) < 80:
        return False
    return all(
        len(theme.what_is_happening.strip()) >= 45
        and len(theme.core_technology.strip()) >= 20
        and len(theme.data_and_workflow.strip()) >= 45
        and len(theme.why_it_matters.strip()) >= 30
        and len(theme.maturity.strip()) >= 10
        and bool(theme.source_urls)
        for theme in synthesis.themes
    )


def _is_cad_topic(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _CAD_TOPIC_MARKERS)


def _has_cad_anchor(title: str, snippet: str) -> bool:
    text = f"{title} {snippet}".lower()
    for marker in _CAD_EVIDENCE_ANCHORS:
        if marker.isascii() and re.fullmatch(r"[a-z0-9-]+", marker):
            if re.search(rf"\b{re.escape(marker)}\b", text):
                return True
        elif marker in text:
            return True
    return False


def _deterministic_technical_queries(
    topic: str,
    understanding: BriefingUnderstanding | None = None,
    intent: BriefingIntentProfile | None = None,
) -> list[str]:
    if isinstance(understanding, BriefingIntentProfile):
        intent = understanding
        understanding = None
    if understanding is None:
        understanding = (
            _understanding_from_legacy_intent(topic, intent)
            if intent is not None
            else default_briefing_understanding(topic)
        )
    profile = understanding.domain_profile
    candidate_terms = [
        topic,
        *understanding.research_focus,
        profile.domain,
        *profile.key_concepts,
        *profile.subtopics,
        *profile.evidence_anchors,
    ]
    focus_terms: list[str] = []
    for candidate in candidate_terms:
        term = " ".join(str(candidate or "").split())
        if term and _is_meaningful_query(term) and term.lower() not in {item.lower() for item in focus_terms}:
            focus_terms.append(term[:120])
        if len(focus_terms) >= 5:
            break

    queries: list[str] = []
    for focus in focus_terms:
        if focus.lower() == topic.lower():
            queries.append(f"{topic} technical evidence architecture implementation")
        else:
            queries.append(f"{topic} {focus} evidence")
    for dimension in understanding.comparison_dimensions[:2]:
        if _is_meaningful_query(dimension):
            queries.append(f"{topic} {dimension} comparison tradeoffs")
    for preference in understanding.evidence_preferences[:3]:
        if _is_meaningful_query(preference):
            queries.append(f"{topic} {preference} evidence")
    if understanding.temporal_focus in {"current", "near_future"}:
        queries.append(f"{topic} current deployment evidence")

    cleaned: list[str] = []
    for query in queries:
        normalized = " ".join(query.split())[:180]
        if normalized and normalized.lower() not in {item.lower() for item in cleaned}:
            cleaned.append(normalized)
        if len(cleaned) >= 3:
            break
    return cleaned or [f"{topic} technical evidence architecture implementation"]


def _duration_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 3)


def _count_knowledge_layers(validation_results: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in validation_results:
        layer = str(item.get("knowledge_layer") or "unreviewed_candidate")
        counts[layer] = counts.get(layer, 0) + 1
    return counts


def _safe_trace_error(exc: Exception) -> str:
    text = str(exc)
    for marker in ("api_key", "API key", "Authorization", "token", "secret", "password"):
        text = text.replace(marker, "[redacted]")
    return text[:500]


def _order_plan_by_source_recipe(
    plans: list[tuple[str, str]],
    source_recipe: dict[str, float] | None,
) -> list[tuple[str, str]]:
    if not plans:
        return []
    if not source_recipe:
        return plans
    recipe = normalize_source_recipe(source_recipe)
    indexed = list(enumerate(plans))
    return [
        plan
        for _, plan in sorted(
            indexed,
            key=lambda item: (
                -recipe.get(_source_slug_for_plan(*item[1]), recipe.get("general-web", 1.0)),
                item[0],
            ),
        )
    ]


def _source_slug_for_plan(platform: str, query: str) -> str:
    lowered = f"{platform} {query}".lower()
    if "mp.weixin.qq.com" in lowered or "weixin.qq.com" in lowered:
        return "wechat-public-index"
    if "bilibili.com" in lowered or "b站" in lowered or "bilibili" in lowered:
        return "bilibili"
    if "zhihu.com" in lowered or "知乎" in lowered:
        return "zhihu"
    if "toutiao.com" in lowered or "今日头条" in lowered:
        return "toutiao-public-index"
    if "xiaohongshu.com" in lowered or "xhslink.com" in lowered or "小红书" in lowered:
        return "xiaohongshu-public-index"
    if "youtube.com" in lowered:
        return "youtube"
    if "reddit.com" in lowered:
        return "reddit"
    if "linkedin.com" in lowered:
        return "linkedin-public-index"
    if "site:x.com" in lowered or "twitter.com" in lowered:
        return "x-public-index"
    if "github.com" in lowered or "arxiv.org" in lowered:
        return "mcp-public"
    return "general-web"


class DailyBriefingService:
    def __init__(
        self,
        store: MemoryStore,
        search_client: SearchClient,
        runtime: BriefingRuntime | None = None,
        max_queries: int = 24,
        results_per_query: int = 10,
        max_sources: int = 50,
        model_max_sources: int = 12,
        search_budget_seconds: float = 90.0,
        timezone_name: str = "Asia/Shanghai",
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self.store = store
        self.search_client = search_client
        self.runtime = runtime
        self.max_queries = max_queries
        self.results_per_query = results_per_query
        self.max_sources = max_sources
        self.model_max_sources = model_max_sources
        self.search_budget_seconds = search_budget_seconds
        self.timezone_name = timezone_name
        self.embedding_provider = embedding_provider or build_embedding_provider()

    def run(
        self,
        topic: str,
        user_id: str,
        chat_id: str,
        run_date: date | None = None,
    ) -> DailyBriefing:
        run_id = f"brief_run_{uuid.uuid4().hex}"
        run_started = time.monotonic()
        try:
            phase_started = time.monotonic()
            subscription = self.store.upsert_topic(user_id, chat_id, topic)
            self._ensure_project_state()
            feedback = self.store.list_topic_feedback(subscription.id)
            knowledge_context = self._knowledge_context(subscription.topic)
            briefing_understanding = self._understand_briefing(
                subscription.topic,
                feedback,
                knowledge_context,
            )
            briefing_intent = _intent_profile_from_understanding(briefing_understanding)
            search_plan = self.build_search_plan(
                subscription,
                feedback,
                understanding=briefing_understanding,
                knowledge_context=knowledge_context,
            )
            self._record_trace_event(
                run_id,
                "briefing",
                "plan_queries",
                phase_started,
                metadata={
                    "topic": subscription.topic,
                    "intent": briefing_intent.primary_intent,
                    "intent_label": briefing_intent.label,
                    "briefing_understanding": _briefing_understanding_metadata(briefing_understanding),
                    "query_count": len(search_plan),
                    "feedback_count": len(feedback),
                    "knowledge_context": self._knowledge_context_metadata(knowledge_context),
                },
            )
            self._record_checkpoint(
                run_id,
                subscription.topic,
                "planned",
                payload={
                    "topic_id": subscription.id,
                    "briefing_intent": briefing_intent.model_dump(),
                    "briefing_understanding": briefing_understanding.model_dump(mode="json"),
                    "query_count": len(search_plan),
                    "feedback_count": len(feedback),
                    "queries": [query for _, query in search_plan[:8]],
                    "knowledge_context": self._knowledge_context_metadata(knowledge_context),
                },
            )

            phase_started = time.monotonic()
            collection = self._collect_sources_with_trace(
                subscription,
                search_plan,
                feedback,
                run_id=run_id,
                understanding=briefing_understanding,
            )
            sources = collection.sources
            ranked_sources = sorted(sources, key=lambda source: source.importance_score, reverse=True)[
                : self.max_sources
            ]
            self._record_trace_event(
                run_id,
                "briefing",
                "collect_sources",
                phase_started,
                metadata={
                    "topic": subscription.topic,
                    "raw_source_count": len(sources),
                    "ranked_source_count": len(ranked_sources),
                    "max_sources": self.max_sources,
                },
            )
            self._record_checkpoint(
                run_id,
                subscription.topic,
                "sources_collected",
                payload={
                    "raw_source_count": len(sources),
                    "ranked_source_count": len(ranked_sources),
                    "source_urls": [source.url for source in ranked_sources[:10]],
                },
            )

            phase_started = time.monotonic()
            model_sources = self._select_model_sources(ranked_sources, self.model_max_sources)
            synthesis = self._synthesize(
                subscription.topic,
                model_sources,
                search_plan,
                fallback_sources=ranked_sources,
                briefing_intent=briefing_intent,
                knowledge_context=knowledge_context,
            )
            self._record_trace_event(
                run_id,
                "briefing",
                "synthesize_report",
                phase_started,
                metadata={
                    "topic": subscription.topic,
                    "intent": briefing_intent.primary_intent,
                    "intent_label": briefing_intent.label,
                    "model_source_count": len(model_sources),
                    "primary_model_source_count": sum(
                        1 for source in model_sources if _is_primary_evidence_source(source)
                    ),
                    "theme_count": len(synthesis.themes),
                    "used_runtime": self.runtime is not None,
                    "synthesis_source": "fallback" if getattr(self, "last_synthesis_fallback", False) else "llm",
                    "knowledge_context": self._knowledge_context_metadata(knowledge_context),
                },
            )
            self._record_checkpoint(
                run_id,
                subscription.topic,
                "synthesized",
                payload={
                    "theme_count": len(synthesis.themes),
                    "model_source_count": len(model_sources),
                    "primary_model_source_count": sum(
                        1 for source in model_sources if _is_primary_evidence_source(source)
                    ),
                    "used_runtime": self.runtime is not None,
                    "knowledge_context": self._knowledge_context_metadata(knowledge_context),
                },
            )

            resolved_date = run_date or self._briefing_date()
            briefing = DailyBriefing(
                id=f"brief_{uuid.uuid4().hex}",
                topic_id=subscription.id,
                user_id=user_id,
                chat_id=chat_id,
                topic=subscription.topic,
                run_date=resolved_date,
                search_directions=[f"{platform}: {query}" for platform, query in search_plan],
                keywords=self._keywords(
                    subscription.topic,
                    feedback,
                    search_plan,
                    intent=briefing_intent,
                    understanding=briefing_understanding,
                ),
                sources=ranked_sources,
                source_candidates=collection.source_candidates,
                provider_events=collection.provider_events,
                synthesis=synthesis,
                markdown="",
                created_at=datetime.now(UTC).isoformat(),
            )
            briefing.markdown = self.render_markdown(briefing)
            self.store.record_daily_briefing(briefing)

            phase_started = time.monotonic()
            candidate_service = DomainKnowledgeCandidateService(
                self.store, embedding_provider=self.embedding_provider
            )
            candidate_ids = candidate_service.capture_briefing(briefing)
            validation_results = [candidate_service.record_validation_gate(candidate_id) for candidate_id in candidate_ids]
            knowledge_layers = _count_knowledge_layers(validation_results)
            try:
                graph_extension = extend_graph_from_briefing(self.store, briefing)
            except Exception:
                graph_extension = {"graph_id": None, "added_entities": 0, "added_relations": 0}
            self._record_trace_event(
                run_id,
                "evolution",
                "capture_domain_knowledge_candidates",
                phase_started,
                metadata={
                    "topic": subscription.topic,
                    "candidate_count": len(candidate_ids),
                    "validation_gate_passed": sum(1 for item in validation_results if item["validated"]),
                    "validation_gate_failed": sum(1 for item in validation_results if not item["validated"]),
                    "knowledge_layers": knowledge_layers,
                },
            )
            self._record_checkpoint(
                run_id,
                subscription.topic,
                "candidates_captured",
                payload={
                    "briefing_id": briefing.id,
                    "candidate_ids": candidate_ids,
                    "validation_gate_passed": sum(1 for item in validation_results if item["validated"]),
                    "validation_gate_failed": sum(1 for item in validation_results if not item["validated"]),
                    "knowledge_layers": knowledge_layers,
                },
            )
            self.store.add_project_ledger_entry(
                entry_type="briefing_run",
                subject=subscription.topic,
                status="completed",
                summary=f"Generated daily briefing with {len(ranked_sources)} retained sources.",
                evidence_refs=[
                    f"briefing:{briefing.id}",
                    *[source.url for source in ranked_sources[:8]],
                    *[f"candidate:{candidate_id}" for candidate_id in candidate_ids[:8]],
                ],
                risk="Public search coverage can be incomplete or blocked; the report should be read with source URLs.",
                rollback="Use an isolated --data-dir for tests, or delete the generated briefing/data directory.",
                metadata={
                    "run_id": run_id,
                    "topic_id": subscription.id,
                    "intent": briefing_intent.primary_intent,
                    "intent_label": briefing_intent.label,
                    "briefing_understanding": _briefing_understanding_metadata(briefing_understanding),
                    "knowledge_context": self._knowledge_context_metadata(knowledge_context),
                    "query_count": len(search_plan),
                    "source_count": len(ranked_sources),
                    "candidate_count": len(candidate_ids),
                    "validation_gate_passed": sum(1 for item in validation_results if item["validated"]),
                    "validation_gate_failed": sum(1 for item in validation_results if not item["validated"]),
                    "knowledge_layers": knowledge_layers,
                    "run_date": briefing.run_date.isoformat(),
                },
            )
            self._record_trace_event(
                run_id,
                "briefing",
                "run",
                run_started,
                metadata={
                    "topic": subscription.topic,
                    "status": "completed",
                    "intent": briefing_intent.primary_intent,
                    "intent_label": briefing_intent.label,
                    "briefing_understanding": _briefing_understanding_metadata(briefing_understanding),
                    "knowledge_context": self._knowledge_context_metadata(knowledge_context),
                    "query_count": len(search_plan),
                    "source_count": len(ranked_sources),
                },
            )
            self._record_checkpoint(
                run_id,
                subscription.topic,
                "completed",
                payload={
                    "briefing_id": briefing.id,
                    "source_count": len(ranked_sources),
                    "candidate_count": len(candidate_ids),
                    "graph_extension": graph_extension,
                },
            )
            return briefing
        except Exception as exc:
            self._record_trace_event(
                run_id,
                "briefing",
                "run",
                run_started,
                status="failed",
                metadata={"topic": topic},
                error=_safe_trace_error(exc),
            )
            self._record_checkpoint(
                run_id,
                topic,
                "failed",
                status="failed",
                payload={"error": _safe_trace_error(exc)},
            )
            raise

    def _understand_briefing(
        self,
        topic: str,
        feedback: list[dict[str, object]],
        knowledge_context: dict[str, object],
    ) -> BriefingUnderstanding:
        context = {
            "feedback": feedback[:5],
            "knowledge_context_summary": self._knowledge_context_for_understanding(knowledge_context),
        }
        if self.runtime is not None:
            try:
                return normalize_briefing_understanding(
                    topic,
                    self.runtime.understand_briefing(topic, context),
                )
            except Exception:
                pass
        return default_briefing_understanding(topic)

    @staticmethod
    def _knowledge_context_for_understanding(knowledge_context: dict[str, object]) -> dict[str, object]:
        return {
            "policy": knowledge_context.get("policy"),
            "reviewed_graph_hits": [
                {
                    "entity_id": str(item.get("entity_id") or ""),
                    "entity_name": str(item.get("entity_name") or ""),
                    "summary": str(item.get("summary") or item.get("description") or "")[:240],
                }
                for item in (knowledge_context.get("reviewed_graph_hits") or [])[:4]
                if isinstance(item, dict)
            ],
            "validated_candidates": [
                {
                    "topic": str(item.get("topic") or ""),
                    "claim": str(item.get("claim") or "")[:240],
                    "knowledge_layer": str(item.get("knowledge_layer") or ""),
                }
                for item in (knowledge_context.get("validated_candidates") or [])[:3]
                if isinstance(item, dict)
            ],
            "procedural_rules": [
                {
                    "kind": str(item.get("kind") or ""),
                    "content": str(item.get("content") or "")[:240],
                }
                for item in (knowledge_context.get("procedural_rules") or [])[:3]
                if isinstance(item, dict)
            ],
        }

    def build_search_plan(
        self,
        subscription: TopicSubscription,
        feedback: list[dict[str, object]],
        intent: BriefingIntentProfile | None = None,
        understanding: BriefingUnderstanding | None = None,
        knowledge_context: dict[str, object] | None = None,
    ) -> list[tuple[str, str]]:
        if understanding is None:
            understanding = (
                _understanding_from_legacy_intent(subscription.topic, intent)
                if intent is not None
                else default_briefing_understanding(subscription.topic)
            )
        else:
            understanding = normalize_briefing_understanding(subscription.topic, understanding)
        briefing_intent = _intent_profile_from_understanding(understanding)
        knowledge_context = knowledge_context or self._knowledge_context(subscription.topic)
        generated_plans: list[tuple[str, str]] = []
        if self.runtime is not None:
            try:
                generated = self.runtime.plan_briefing_queries(
                    subscription.topic,
                    {
                        "feedback": feedback[:5],
                        "channels": [platform for platform, _ in CHANNEL_QUERIES],
                        "briefing_intent": briefing_intent.model_dump(),
                        "briefing_understanding": understanding.model_dump(mode="json"),
                        "domain_profile": understanding.domain_profile.model_dump(mode="json"),
                        "research_focus": understanding.research_focus,
                        "knowledge_context": knowledge_context,
                        "report_skill": _load_report_skill(),
                        "source_recipe": source_recipe_summary(subscription.source_recipe),
                    },
                )
            except Exception:
                generated = []
            generated_plans.extend(("技术路线", query) for query in _clean_planned_queries(generated))
        deterministic_plans = [
            ("技术路线（确定性保障）", query)
            for query in _deterministic_technical_queries(subscription.topic, understanding=understanding)
        ]
        feedback_plans: list[tuple[str, str]] = []
        for item in feedback[:3]:
            note = " ".join(str(item.get("body") or "").split())
            if note and _is_meaningful_feedback(note):
                feedback_plans.append(("案例反馈", f"{subscription.topic} {note[:160]}"))
        knowledge_plans = self._knowledge_guided_queries(subscription.topic, knowledge_context)
        channel_plans = [(platform, template.format(topic=subscription.topic)) for platform, template in CHANNEL_QUERIES]
        plans = self._compose_search_plan(
            generated_plans,
            deterministic_plans,
            feedback_plans,
            knowledge_plans,
            channel_plans,
        )
        unique: list[tuple[str, str]] = []
        seen: set[str] = set()
        for platform, query in plans:
            normalized = query.lower().strip()
            if normalized and normalized not in seen:
                unique.append((platform, query))
                seen.add(normalized)
        return _order_plan_by_source_recipe(unique, subscription.source_recipe)[: self.max_queries]

    def _compose_search_plan(
        self,
        generated_plans: list[tuple[str, str]],
        deterministic_plans: list[tuple[str, str]],
        feedback_plans: list[tuple[str, str]],
        knowledge_plans: list[tuple[str, str]],
        channel_plans: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """Compose query plans while reserving room for the original channels."""

        if self.max_queries >= len(channel_plans):
            non_channel_budget = max(0, self.max_queries - len(channel_plans))
            high_value = generated_plans[:4] + feedback_plans + knowledge_plans + deterministic_plans + generated_plans[4:]
            return high_value[:non_channel_budget] + channel_plans
        return generated_plans + deterministic_plans + feedback_plans + channel_plans + knowledge_plans

    def add_feedback(
        self,
        topic: str,
        user_id: str,
        chat_id: str,
        body: str,
        source_url: str | None = None,
    ) -> str:
        subscription = self.store.upsert_topic(user_id, chat_id, topic)
        enriched_body = body.strip()
        case_context = self._inspect_case_source(source_url)
        if case_context:
            enriched_body = "\n".join(part for part in (enriched_body, case_context) if part)
        feedback_id = self.store.add_topic_feedback(subscription.id, user_id, chat_id, enriched_body, source_url)
        signal_type, scope = self._feedback_signal_type(enriched_body, source_url)
        self.store.add_topic_feedback_signal(
            subscription.id,
            user_id,
            chat_id,
            signal_type=signal_type,
            scope=scope,
            body=enriched_body,
            source_url=source_url,
            metadata={"feedback_id": feedback_id},
        )
        memory_layer = "preference" if signal_type == "style" else "run_experience"
        self.store.add_layered_memory_item(
            memory_layer,
            f"briefing_{signal_type}_feedback",
            enriched_body,
            source_id=feedback_id,
            confidence="medium",
            status="active",
            user_id=user_id,
            chat_id=chat_id,
            metadata={"topic_id": subscription.id, "scope": scope, "source_url": source_url},
        )
        return feedback_id

    @staticmethod
    def _feedback_signal_type(body: str, source_url: str | None) -> tuple[str, str]:
        lowered = body.lower()
        if any(term in lowered for term in ("too long", "too verbose", "readability", "style", "简洁", "太长", "难读")):
            return "style", "topic"
        if any(term in lowered for term in ("provider", "agent reach", "search source", "没走", "来源", "搜索源")):
            return "provider", "provider"
        if any(term in lowered for term in ("wrong", "incorrect", "事实", "错误", "纠正")):
            return "fact_correction", "source" if source_url else "topic"
        if source_url:
            return "case", "source"
        if any(term in lowered for term in ("evidence", "source", "证据", "原文", "引用")):
            return "evidence", "source"
        return "general", "topic"

    def _knowledge_context(self, topic: str) -> dict[str, object]:
        """Retrieve reviewed graph and self-evolution context before planning.

        Context is routed by memory type:

        - **Semantic** memory: reviewed graph hits (``reviewed_graph_hits``).
        - **Episodic** memory: recent trajectory logs that mention the topic
          (``episodic_trajectories``), so prior runs of the same topic can
          inform the plan without being treated as current facts.
        - **Procedural** memory: layered ``domain_knowledge``/``run_experience``
          items (``procedural_rules``), i.e. distilled workflow guidance.

        Self-evolution candidates are split by release state: validated
        candidates can be used as planning context, while weak signals only
        suggest what to verify next.  None of these categories is treated as a
        current factual source.
        """

        graph_service = DomainKnowledgeGraphService(self.store, embedding_provider=self.embedding_provider)
        seeded_graphs = 0
        try:
            if not graph_service.list_graphs():
                seeded_graphs = int(graph_service.seed_default_graphs()["seeded_graphs"])
            graph_hits = graph_service.query_relevant(topic, limit=4, min_score=5.0)
        except Exception:
            graph_hits = []

        latest_layers = self._latest_candidate_layers()
        validated_candidates: list[dict[str, object]] = []
        weak_signals: list[dict[str, object]] = []
        topic_terms = set(_topic_terms(topic))
        for candidate in self.store.list_domain_knowledge_candidates():
            if str(candidate.get("status")) == "deprecated":
                continue
            layer = latest_layers.get(str(candidate.get("id")), "unreviewed_candidate")
            if not self._candidate_matches_topic(candidate, topic, topic_terms):
                continue
            item = {
                "id": candidate["id"],
                "topic": candidate["topic"],
                "claim": str(candidate.get("claim", ""))[:360],
                "confidence": candidate.get("confidence"),
                "knowledge_layer": layer,
                "evidence_url_count": len(
                    [
                        evidence
                        for evidence in candidate.get("evidence", [])
                        if str(evidence.get("url", "")).startswith(("http://", "https://"))
                    ]
                ),
            }
            if str(candidate.get("status")) == "validated":
                validated_candidates.append(item)
            elif layer == "weak_signal":
                weak_signals.append(item)

        return {
            "policy": (
                "Use reviewed graph hits, episodic trajectories and validated self-evolution candidates only as "
                "planning and framing context; re-open original sources before making current factual claims. "
                "Use weak signals only for next research directions, not as evidence."
            ),
            "seeded_graphs": seeded_graphs,
            "reviewed_graph_hits": [hit.model_dump(mode="json") for hit in graph_hits],
            "episodic_trajectories": self._episodic_trajectories(topic, topic_terms),
            "procedural_rules": self._procedural_rules(topic, topic_terms),
            "validated_candidates": validated_candidates[:3],
            "weak_signals": weak_signals[:3],
        }

    def _episodic_trajectories(
        self,
        topic: str,
        topic_terms: set[str],
        limit: int = 3,
    ) -> list[dict[str, object]]:
        """Return recent trajectory logs that mention the topic (episodic memory).

        Each returned item is a compact summary of a past run on this topic:
        when it happened and what the trajectory tried.  It is planning context
        only and must not be quoted as a current fact.
        """
        try:
            logs = self.store.list_trajectory_logs()
        except Exception:
            return []
        episodic: list[dict[str, object]] = []
        for log in reversed(logs):
            payload = log.get("payload") or {}
            text = " ".join(
                str(value)
                for value in (
                    payload.get("question"),
                    payload.get("query"),
                    payload.get("answer"),
                    payload.get("summary"),
                    payload.get("topic"),
                )
                if value
            ).lower()
            if not text:
                continue
            matched = [term for term in topic_terms if term.lower() in text]
            if not matched:
                continue
            episodic.append(
                {
                    "trajectory_id": str(log.get("id") or ""),
                    "question_id": str(log.get("question_id") or ""),
                    "source": str(log.get("source") or ""),
                    "created_at": str(log.get("created_at") or ""),
                    "matched_terms": matched[:4],
                    "summary": text[:240],
                }
            )
            if len(episodic) >= limit:
                break
        return episodic

    def _procedural_rules(
        self,
        topic: str,
        topic_terms: set[str],
        limit: int = 3,
    ) -> list[dict[str, object]]:
        """Return layered procedural memory items relevant to the topic.

        ``run_experience`` items capture distilled workflow guidance from past
        runs (e.g. which search strategy worked); ``domain_knowledge`` items are
        reviewed domain facts.  Only active items are returned.
        """
        try:
            items = self.store.list_layered_memory_items(limit=500)
        except Exception:
            return []
        rules: list[dict[str, object]] = []
        for item in items:
            if item.status != "active":
                continue
            if item.layer not in ("run_experience", "domain_knowledge"):
                continue
            text = " ".join((item.content, str(item.metadata.get("topic") or ""))).lower()
            matched = [term for term in topic_terms if term.lower() in text]
            if not matched:
                continue
            rules.append(
                {
                    "layer": item.layer,
                    "kind": item.kind,
                    "content": item.content[:360],
                    "confidence": item.confidence,
                    "source_id": item.source_id,
                    "matched_terms": matched[:4],
                }
            )
            if len(rules) >= limit:
                break
        return rules

    def _knowledge_guided_queries(
        self,
        topic: str,
        knowledge_context: dict[str, object],
    ) -> list[tuple[str, str]]:
        queries: list[tuple[str, str]] = []
        for hit in knowledge_context.get("reviewed_graph_hits", [])[:5]:
            if not isinstance(hit, dict):
                continue
            entity_name = str(hit.get("entity_name") or "").strip()
            # Skip generic terms and tokenizer fragments so the graph-guided
            # queries carry discriminating vocabulary (e.g. "受控工单编排")
            # instead of noise (e.g. "能体", "部署").
            if entity_name and is_meaningful_entity_name(entity_name):
                queries.append(("知识图谱补充", f"{topic} {entity_name} implementation evidence"))
        # Validated self-evolution candidates are intentionally NOT turned into raw
        # search queries here. Directly concatenating claim terms from a previous
        # topic (e.g. "loop engineering agent harness") into a new topic pollutes the
        # search plan with out-of-context vocabulary. They remain available as
        # framing context via knowledge_context["validated_candidates"] during
        # synthesis, where the model can use them for analogies without affecting
        # retrieval keywords.
        return queries

    def _latest_candidate_layers(self) -> dict[str, str]:
        latest: dict[str, str] = {}
        for gate in self.store.list_gate_records(gate_type="domain_knowledge_candidate_validation"):
            candidate_id = str(gate["subject_id"])
            if candidate_id in latest:
                continue
            metadata = gate.get("metadata") or {}
            latest[candidate_id] = str(metadata.get("knowledge_layer") or "unreviewed_candidate")
        return latest

    def _candidate_matches_topic(self, candidate: dict[str, object], topic: str, topic_terms: set[str]) -> bool:
        """Decide whether a self-evolution candidate belongs to the current topic.

        Semantic matching (embedding cosine similarity on the topic fields) is
        tried first so cross-lingual or paraphrased topics can still match.
        The literal fallback only inspects the candidate ``topic`` field and
        requires at least two overlapping terms, avoiding the previous
        topic+claim+applies_when haystack that matched nearly every candidate
        through generic AI vocabulary in claims.
        """
        if self._candidate_topic_semantic_match(topic, str(candidate.get("topic", ""))):
            return True
        if not topic_terms:
            return False
        candidate_topic = str(candidate.get("topic", "")).lower()
        if not candidate_topic:
            return False
        matched = [term for term in topic_terms if term.lower() in candidate_topic]
        return len(matched) >= 2

    def _candidate_topic_semantic_match(self, topic: str, candidate_topic: str) -> bool:
        """Return True when the candidate topic is semantically close to the query topic."""
        if not topic.strip() or not candidate_topic.strip():
            return False
        provider = self.embedding_provider
        if provider is None or provider.__class__.__name__ == "NullEmbeddingProvider":
            return False
        query_vector = self._embedding_cached(topic.strip())
        candidate_vector = self._embedding_cached(candidate_topic.strip())
        if not query_vector or not candidate_vector:
            return False
        return cosine_similarity(query_vector, candidate_vector) >= 0.78

    def _embedding_cached(self, text: str) -> list[float] | None:
        """Return an embedding vector for ``text``, using the SQLite cache."""
        provider = self.embedding_provider
        if provider is None or provider.__class__.__name__ == "NullEmbeddingProvider":
            return None
        cache_key = _embedding_cache_key(text)
        cached = self.store.get_entity_embedding(cache_key)
        if cached:
            return cached
        try:
            vectors = provider.embed([text])
        except Exception:
            return None
        if not vectors or not vectors[0]:
            return None
        vector = vectors[0]
        self.store.upsert_entity_embedding(
            cache_key,
            provider.__class__.__name__,
            getattr(provider, "model", "unknown"),
            vector,
        )
        return vector

    @staticmethod
    def _knowledge_context_metadata(knowledge_context: dict[str, object]) -> dict[str, object]:
        return {
            "seeded_graphs": knowledge_context.get("seeded_graphs", 0),
            "reviewed_graph_hits": len(knowledge_context.get("reviewed_graph_hits", [])),
            "episodic_trajectories": len(knowledge_context.get("episodic_trajectories", [])),
            "procedural_rules": len(knowledge_context.get("procedural_rules", [])),
            "validated_candidates": len(knowledge_context.get("validated_candidates", [])),
            "weak_signals": len(knowledge_context.get("weak_signals", [])),
        }

    def _inspect_case_source(self, source_url: str | None) -> str:
        if not source_url:
            return ""
        normalized_url = self._normalize_url(source_url)
        if not normalized_url:
            return ""
        try:
            results = self.search_client.search(normalized_url, limit=3)
        except Exception:
            return ""
        if not results:
            return ""
        matched = next(
            (result for result in results if self._normalize_url(result.url) == normalized_url),
            results[0],
        )
        title = " ".join(matched.title.split())[:240]
        snippet = " ".join(matched.snippet.split())[:800]
        if not title and not snippet:
            return ""
        return f"案例公开索引：{title}\n{snippet}".strip()

    def _collect_sources(
        self,
        subscription: TopicSubscription,
        search_plan: list[tuple[str, str]],
        feedback: list[dict[str, object]],
        run_id: str | None = None,
        understanding: BriefingUnderstanding | None = None,
    ) -> list[CollectedSource]:
        return self._collect_sources_with_trace(
            subscription,
            search_plan,
            feedback,
            run_id=run_id,
            understanding=understanding,
        ).sources

    def _collect_sources_with_trace(
        self,
        subscription: TopicSubscription,
        search_plan: list[tuple[str, str]],
        feedback: list[dict[str, object]],
        run_id: str | None = None,
        understanding: BriefingUnderstanding | None = None,
    ) -> BriefingCollectionTrace:
        understanding = understanding or default_briefing_understanding(subscription.topic)
        by_url: dict[str, CollectedSource] = {}
        query_results: list[tuple[int, str, str, SearchOutcome]] = []
        source_candidates: list[SourceCandidate] = []
        provider_events: list[ProviderTraceEvent] = []
        search_tasks = self._budgeted_search_tasks(subscription, search_plan)
        tier_budgets = self._tier_budget_seconds(search_tasks)

        def record_provider_event(event: ProviderTraceEvent) -> None:
            provider_events.append(event)

        def persist_provider_events() -> None:
            if not provider_events:
                return
            try:
                self.store.record_provider_trace_events(provider_events, run_id=run_id, topic_id=subscription.id)
            except Exception:
                return

        def annotate_event(event: ProviderTraceEvent, task: BriefingSearchTask) -> ProviderTraceEvent:
            return event.model_copy(update={"tier": task.tier, "budget_share": task.budget_share})

        def budget_event(
            task: BriefingSearchTask,
            status: str,
            reason: str,
            started: float | None = None,
        ) -> ProviderTraceEvent:
            elapsed_ms = None if started is None else round((time.monotonic() - started) * 1000, 3)
            return ProviderTraceEvent(
                provider="briefing-search-budget",
                query=task.query,
                status=status,
                result_count=0,
                reason=reason,
                elapsed_ms=elapsed_ms,
                tier=task.tier,
                budget_share=task.budget_share,
                checked_at=datetime.now(UTC).isoformat(),
            )

        if not search_tasks:
            return BriefingCollectionTrace(sources=[], source_candidates=[], provider_events=[])

        def run_tier(tier_tasks: list[BriefingSearchTask], tier_budget_seconds: float) -> None:
            if tier_budget_seconds <= 0:
                for task in tier_tasks:
                    record_provider_event(budget_event(task, "skipped", "tier_budget_is_zero"))
                    self._record_provider_health(
                        run_id,
                        task.requested_platform,
                        task.query,
                        [],
                        0.0,
                        ok=False,
                        error="tier_budget_is_zero",
                    )
                return

            workers = max(1, min(6, len(tier_tasks)))
            deadline = time.monotonic() + tier_budget_seconds
            next_position = 0
            pending: dict[Future[SearchOutcome], tuple[BriefingSearchTask, float]] = {}
            executor = ThreadPoolExecutor(max_workers=workers)

            def submit_next_task() -> bool:
                nonlocal next_position
                if next_position >= len(tier_tasks):
                    return False
                if time.monotonic() >= deadline:
                    return False
                task = tier_tasks[next_position]
                started = time.monotonic()
                future = executor.submit(self._search_with_trace, task.query)
                pending[future] = (task, started)
                next_position += 1
                record_provider_event(budget_event(task, "called", "tier_query_started", started))
                return True

            def collect_completed_future(future: Future[SearchOutcome]) -> None:
                task, started = pending.pop(future)
                try:
                    outcome = future.result()
                except Exception as exc:
                    outcome = SearchOutcome(
                        results=[],
                        provider_events=[
                            ProviderTraceEvent(
                                provider="briefing-search",
                                query=task.query,
                                status="error",
                                result_count=0,
                                error=f"{type(exc).__name__}: {exc}"[:240],
                                checked_at=datetime.now(UTC).isoformat(),
                            )
                        ],
                    )
                query_results.append((task.index, task.requested_platform, task.query, outcome))
                for event in outcome.provider_events:
                    record_provider_event(annotate_event(event, task))
                failed_events = [
                    event for event in outcome.provider_events if event.status in {"error", "timeout"}
                ]
                self._record_provider_health(
                    run_id,
                    task.requested_platform,
                    task.query,
                    outcome.results,
                    _duration_ms(started),
                    ok=not failed_events,
                    error="; ".join(
                        str(event.error or event.reason or event.status) for event in failed_events
                    )
                    or None,
                )

            def expire_overdue_queries() -> None:
                now = time.monotonic()
                for future, (task, started) in list(pending.items()):
                    if future.done():
                        continue
                    if now - started < task.timeout_seconds:
                        continue
                    pending.pop(future)
                    cancelled = future.cancel()
                    status = "skipped" if cancelled else "timeout"
                    reason = (
                        "query_timeout_before_worker_start"
                        if cancelled
                        else f"exceeded_{task.timeout_seconds:g}s_query_timeout"
                    )
                    record_provider_event(budget_event(task, status, reason, started))
                    self._record_provider_health(
                        run_id,
                        task.requested_platform,
                        task.query,
                        [],
                        _duration_ms(started),
                        ok=False,
                        error=reason,
                    )

            try:
                while len(pending) < workers and submit_next_task():
                    pass
                while pending or next_position < len(tier_tasks):
                    expire_overdue_queries()
                    while len(pending) < workers and submit_next_task():
                        pass
                    if not pending:
                        break
                    remaining_tier_seconds = deadline - time.monotonic()
                    if remaining_tier_seconds <= 0:
                        break
                    remaining_query_seconds = min(
                        max(0.001, task.timeout_seconds - (time.monotonic() - started))
                        for task, started in pending.values()
                    )
                    completed, _pending = wait(
                        list(pending),
                        timeout=min(remaining_tier_seconds, remaining_query_seconds),
                        return_when=FIRST_COMPLETED,
                    )
                    if not completed:
                        continue
                    completed_in_order = sorted(
                        (future for future in completed if future in pending),
                        key=lambda item: pending[item][0].index,
                    )
                    for future in completed_in_order:
                        collect_completed_future(future)

                for future in list(pending):
                    if future.done():
                        collect_completed_future(future)
                        continue
                    task, started = pending.pop(future)
                    cancelled = future.cancel()
                    status = "skipped" if cancelled else "timeout"
                    reason = (
                        "tier_budget_expired_before_worker_start"
                        if cancelled
                        else f"exceeded_{tier_budget_seconds:g}s_tier_budget"
                    )
                    record_provider_event(budget_event(task, status, reason, started))
                    self._record_provider_health(
                        run_id,
                        task.requested_platform,
                        task.query,
                        [],
                        _duration_ms(started),
                        ok=False,
                        error=reason,
                    )

                while next_position < len(tier_tasks):
                    task = tier_tasks[next_position]
                    record_provider_event(budget_event(task, "skipped", "tier_budget_exhausted_before_query_start"))
                    self._record_provider_health(
                        run_id,
                        task.requested_platform,
                        task.query,
                        [],
                        0.0,
                        ok=False,
                        error="tier_budget_exhausted_before_query_start",
                    )
                    next_position += 1
            finally:
                # Do not block the daily briefing on slow public endpoints. Completed
                # results are retained; queued calls are cancelled when possible.
                executor.shutdown(wait=False, cancel_futures=True)

        for priority in sorted({task.priority for task in search_tasks}):
            tier_tasks = [task for task in search_tasks if task.priority == priority]
            run_tier(tier_tasks, tier_budgets.get(priority, 0.0))

        for _index, requested_platform, query, outcome in sorted(query_results, key=lambda item: item[0]):
            for result in outcome.results:
                normalized_url = self._normalize_url(result.url)
                title = self._compact_source_text(result.title, 240)
                snippet = self._compact_source_text(result.snippet, 900)
                relevance_score = max(
                    self._relevance_score(subscription.topic, title, snippet),
                    self._query_relevance_score(query, title, snippet),
                )
                quality_verdict = self._source_quality_verdict(
                    topic=subscription.topic,
                    query=query,
                    url=normalized_url,
                    title=title,
                    snippet=snippet,
                    provider=result.provider,
                    understanding=understanding,
                )
                importance_score = self._importance_score(
                    subscription.topic,
                    query,
                    title,
                    snippet,
                    result.provider,
                    verdict=quality_verdict,
                )
                rejection = self._source_rejection_reason(
                    subscription.topic,
                    query,
                    normalized_url,
                    title,
                    snippet,
                    verdict=quality_verdict,
                )
                if rejection is not None:
                    rejected = self._source_candidate(
                        subscription=subscription,
                        requested_platform=requested_platform,
                        query=query,
                        result=result,
                        normalized_url=normalized_url,
                        title=title,
                        snippet=snippet,
                        status=rejection[0],
                        reason=rejection[1],
                        relevance_score=relevance_score,
                        importance_score=importance_score,
                    )
                    source_candidates.append(rejected)
                    self.store.record_source_candidate(rejected)
                    continue
                candidate = CollectedSource(
                    id=f"src_{uuid.uuid4().hex}",
                    topic_id=subscription.id,
                    user_id=subscription.user_id,
                    title=title,
                    url=normalized_url,
                    snippet=snippet,
                    platform=self._platform_from_url(normalized_url, requested_platform),
                    provider=result.provider,
                    query=query,
                    relevance_score=relevance_score,
                    importance_score=importance_score,
                    retrieved_at=result.checked_at,
                )
                prior = by_url.get(normalized_url)
                if prior is not None and candidate.importance_score <= prior.importance_score:
                    duplicate = self._source_candidate(
                        subscription=subscription,
                        requested_platform=requested_platform,
                        query=query,
                        result=result,
                        normalized_url=normalized_url,
                        title=title,
                        snippet=snippet,
                        status="duplicate",
                        reason="same normalized URL already has an equal or stronger candidate",
                        relevance_score=relevance_score,
                        importance_score=importance_score,
                    )
                    source_candidates.append(duplicate)
                    self.store.record_source_candidate(duplicate)
                    continue
                if prior is None or candidate.importance_score > prior.importance_score:
                    by_url[normalized_url] = candidate

        for item in feedback:
            source_url = str(item.get("source_url") or "").strip()
            if not source_url or source_url in by_url:
                continue
            body = str(item.get("body") or "")
            by_url[source_url] = CollectedSource(
                id=f"src_{uuid.uuid4().hex}",
                topic_id=subscription.id,
                user_id=subscription.user_id,
                title="用户提供案例",
                url=source_url,
                snippet=body,
                platform="用户案例",
                provider="user-feedback",
                query=subscription.topic,
                relevance_score=10.0,
                importance_score=12.0,
                retrieved_at=str(item.get("created_at") or datetime.now(UTC).isoformat()),
            )
            feedback_candidate = SourceCandidate(
                id=f"cand_{uuid.uuid4().hex}",
                topic_id=subscription.id,
                user_id=subscription.user_id,
                title=by_url[source_url].title,
                url=source_url,
                snippet=self._compact_source_text(body, 900),
                platform=by_url[source_url].platform,
                provider="user-feedback",
                query=subscription.topic,
                status="feedback_seed",
                reason="user supplied a public case URL",
                relevance_score=10.0,
                importance_score=12.0,
                retrieved_at=by_url[source_url].retrieved_at,
                created_at=datetime.now(UTC).isoformat(),
            )
            source_candidates.append(feedback_candidate)
            self.store.record_source_candidate(feedback_candidate)

        persisted: list[CollectedSource] = []
        for source in by_url.values():
            stored_source = self.store.upsert_collected_source(source)
            persisted.append(stored_source)
            accepted = SourceCandidate(
                id=f"cand_{uuid.uuid4().hex}",
                topic_id=subscription.id,
                user_id=subscription.user_id,
                title=stored_source.title,
                url=stored_source.url,
                snippet=stored_source.snippet,
                platform=stored_source.platform,
                provider=stored_source.provider,
                query=stored_source.query,
                status="accepted",
                reason="accepted for briefing ranking",
                relevance_score=stored_source.relevance_score,
                importance_score=stored_source.importance_score,
                retrieved_at=stored_source.retrieved_at,
                created_at=datetime.now(UTC).isoformat(),
            )
            source_candidates.append(accepted)
            self.store.record_source_candidate(accepted)
        persist_provider_events()
        return BriefingCollectionTrace(
            sources=persisted,
            source_candidates=source_candidates,
            provider_events=provider_events,
        )

    @staticmethod
    def _budgeted_search_tasks(
        subscription: TopicSubscription,
        search_plan: list[tuple[str, str]],
    ) -> list[BriefingSearchTask]:
        recipe = normalize_source_recipe(subscription.source_recipe)
        total_weight = sum(recipe.values()) or 1.0
        tasks: list[BriefingSearchTask] = []
        for index, (requested_platform, query) in enumerate(search_plan):
            contract = source_contract_for_query(requested_platform, query)
            weight = recipe.get(contract.slug, contract.default_budget_share)
            tasks.append(
                BriefingSearchTask(
                    index=index,
                    requested_platform=requested_platform,
                    query=query,
                    source_slug=contract.slug,
                    priority=max(1, int(contract.priority)),
                    timeout_seconds=max(0.1, float(contract.timeout_seconds)),
                    budget_share=round(weight / total_weight, 4),
                )
            )
        return sorted(tasks, key=lambda task: (task.priority, task.index))

    def _tier_budget_seconds(self, tasks: list[BriefingSearchTask]) -> dict[int, float]:
        if not tasks:
            return {}
        present_shares_by_priority: dict[int, dict[str, float]] = {}
        for task in tasks:
            present_shares_by_priority.setdefault(task.priority, {})[task.source_slug] = task.budget_share
        tier_weights = {
            priority: sum(shares.values())
            for priority, shares in present_shares_by_priority.items()
        }
        total_present_weight = sum(tier_weights.values()) or 1.0
        budget = max(0.0, self.search_budget_seconds)
        return {
            priority: budget * weight / total_present_weight
            for priority, weight in tier_weights.items()
        }

    def _search(self, query: str) -> list[SearchResult]:
        return self.search_client.search(query, limit=self.results_per_query)

    def _record_trace_event(
        self,
        run_id: str,
        event_type: str,
        name: str,
        started: float,
        status: str = "completed",
        metadata: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        try:
            self.store.add_trace_event(
                run_id=run_id,
                event_type=event_type,
                name=name,
                status=status,
                duration_ms=_duration_ms(started),
                metadata=metadata or {},
                error=error,
            )
        except Exception:
            # Observability should never make a briefing fail.
            return

    def _record_checkpoint(
        self,
        run_id: str,
        subject: str,
        step: str,
        status: str = "completed",
        payload: dict[str, object] | None = None,
    ) -> None:
        try:
            self.store.add_run_checkpoint(
                run_id=run_id,
                workflow="daily_briefing",
                subject=subject,
                step=step,
                status=status,
                payload=payload or {},
            )
        except Exception:
            # Checkpoint metadata should never make a briefing fail.
            return

    def _ensure_project_state(self) -> None:
        try:
            self.store.ensure_project_ledger_snapshot(
                project_id="pydantic-ai-tech-intel-briefing",
                objective="Generate source-backed technology intelligence briefings with reviewable self-evolution.",
                phase="agentops-readiness",
                status="active",
                next_decision="Run eval replay, then release domain knowledge only after eval and human review gates pass.",
                open_blockers=[],
                constraints={
                    "knowledge_release": "candidate-only until eval gate and human review pass",
                    "search": "public source coverage can be incomplete or blocked",
                    "privacy": "local configuration and secrets must not be persisted in audit logs",
                },
                decisions=[
                    {
                        "id": "agentops-p1-gates",
                        "decision": "Record automatic evidence gates, checkpoints, provider health, and release ledger.",
                    }
                ],
                evidence_refs=["README.md", "docs/development.md"],
                version="1",
            )
        except Exception:
            return

    def _record_provider_health(
        self,
        run_id: str | None,
        requested_platform: str,
        query: str,
        results: list[SearchResult],
        duration_ms: float,
        ok: bool,
        error: str | None = None,
    ) -> None:
        if run_id is None:
            return
        providers = sorted({result.provider for result in results if result.provider})
        provider_label = ", ".join(providers[:3]) if providers else self.search_client.__class__.__name__
        if len(providers) > 3:
            provider_label = f"{provider_label}, +{len(providers) - 3} more"
        try:
            self.store.record_search_provider_health(
                run_id=run_id,
                requested_platform=requested_platform,
                provider=provider_label,
                query=query,
                ok=ok,
                result_count=len(results),
                duration_ms=duration_ms,
                error=error,
            )
        except Exception:
            return

    def _search_with_trace(self, query: str) -> SearchOutcome:
        return search_with_provider_events(self.search_client, query, limit=self.results_per_query)

    def _select_model_sources(
        self,
        ranked_sources: list[CollectedSource],
        limit: int,
    ) -> list[CollectedSource]:
        """Choose up to ``limit`` sources for the synthesis model with a
        primary-evidence quota.

        Primary sources (papers, repositories, model hubs, benchmarks, official
        docs) are taken first, then neutral sources, then at most a quarter of
        the limit (capped at two) marketing-funnel sources.  A batch made only
        of weak sources still fills the limit so the report can name the
        evidence limitation instead of synthesizing from nothing.
        """
        if limit <= 0 or not ranked_sources:
            return []
        primary = [source for source in ranked_sources if _is_primary_evidence_source(source)]
        weak = [
            source
            for source in ranked_sources
            if not _is_primary_evidence_source(source) and _is_weak_evidence_source(source)
        ]
        neutral = [
            source
            for source in ranked_sources
            if not _is_primary_evidence_source(source) and not _is_weak_evidence_source(source)
        ]
        selected: list[CollectedSource] = primary[:limit]
        selected.extend(neutral[: limit - len(selected)])
        if weak:
            # Cap marketing-funnel sources unless the batch has nothing else;
            # an all-weak batch still fills the limit so the report can name
            # the evidence limitation instead of synthesizing from nothing.
            weak_cap = limit if not (primary or neutral) else min(2, max(1, limit // 4))
            selected.extend(weak[: min(weak_cap, limit - len(selected))])
        if not selected and weak:
            selected = weak[:limit]
        return selected

    def _synthesize(
        self,
        topic: str,
        sources: list[CollectedSource],
        search_plan: list[tuple[str, str]],
        fallback_sources: list[CollectedSource] | None = None,
        briefing_intent: BriefingIntentProfile | None = None,
        knowledge_context: dict[str, object] | None = None,
    ) -> BriefingSynthesis:
        # Track whether the deterministic fallback was used so the caller can
        # annotate the report (synthesis_source=fallback) and audit it.  A
        # silent fallback hides degraded report quality from both users and
        # operators.
        self.last_synthesis_fallback = False
        if self.runtime is not None:
            try:
                generated = self.runtime.synthesize_briefing(
                    topic,
                    sources,
                    {
                        "report_contract": REPORT_CONTRACT,
                        "report_skill": _load_report_skill(),
                        "search_plan": search_plan,
                        "readability": REPORT_CONTRACT["readability"],
                        "briefing_intent": briefing_intent.model_dump() if briefing_intent else None,
                        "knowledge_context": knowledge_context or {},
                    },
                )
            except Exception as exc:
                generated = None
                self._record_synthesis_fallback(topic, sources, exc)
            substantive = generated is not None and _is_substantive_synthesis(generated)
            if substantive:
                return self._preserve_generated_detail(generated, topic, sources)
            self._record_synthesis_fallback(topic, sources, None)
        self.last_synthesis_fallback = True
        return self._fallback_synthesis(topic, fallback_sources or sources)

    def _record_synthesis_fallback(
        self,
        topic: str,
        sources: list[CollectedSource],
        exc: Exception | None,
    ) -> None:
        """Persist an audit trail entry when briefing synthesis degrades to the
        deterministic fallback (LLM error, timeout, or an insubstantive reply)."""
        reason = (
            f"briefing synthesis failed or returned insubstantive content: {type(exc).__name__}: {exc}"
            if exc is not None
            else "briefing synthesis returned content that failed the substantive check"
        )
        try:
            gate_id = self.store.add_gate_record(
                gate_type="briefing_synthesis_fallback",
                subject_type="briefing_topic",
                subject_id=topic.strip() or "unspecified",
                result="failed",
                reason=reason,
                evidence_refs=[str(source.url) for source in sources[:5]],
                metadata={"source_count": len(sources)},
            )
            self.store.add_project_ledger_entry(
                entry_type="briefing_synthesis_fallback",
                subject=topic.strip() or "unspecified",
                status="degraded",
                summary=reason,
                evidence_refs=[gate_id, *[str(source.url) for source in sources[:5]]],
                risk="Deterministic fallback templates were used; the report may miss evidence-driven detail.",
                rollback="Re-run the briefing after fixing the synthesis timeout or model availability.",
                metadata={"source_count": len(sources)},
            )
        except Exception:
            # Audit recording must never break the briefing itself.
            pass

    def _preserve_generated_detail(
        self,
        generated: BriefingSynthesis,
        topic: str,
        sources: list[CollectedSource],
    ) -> BriefingSynthesis:
        # A generated detailed summary has already selected its own evidence
        # structure.  Do not inject catalogue themes merely to reach a count.
        if generated.detailed_summary.strip() or len(generated.themes) >= 3 or len(sources) < 3:
            return generated
        fallback = self._fallback_synthesis(topic, sources)
        if len(fallback.themes) < 3:
            return generated
        return generated.model_copy(update={"themes": fallback.themes})

    def _fallback_synthesis(self, topic: str, sources: list[CollectedSource]) -> BriefingSynthesis:
        themed_sources: dict[str, list[CollectedSource]] = {}
        is_cad = _is_cad_topic(topic)
        is_industrial = self._requires_industrial_anchor(topic, "")
        theme_catalog = CAD_THEMES + THEMES if is_cad else (THEMES if is_industrial else ())
        for source in sources:
            haystack = f"{source.title} {source.snippet}".lower()
            matched = False
            for name, markers, _, _, _ in theme_catalog:
                if any(self._matches_technical_marker(haystack, marker) for marker in markers):
                    themed_sources.setdefault(name, []).append(source)
                    matched = True
                    break
            if not matched:
                if is_cad:
                    themed_sources.setdefault("工程图与 CAD 工具链线索", []).append(source)
                else:
                    themed_sources.setdefault(self._generic_evidence_theme_name(source), []).append(source)

        themes: list[BriefingTheme] = []
        for name, markers, technology, importance, maturity in theme_catalog:
            evidence = themed_sources.get(name, [])
            if not evidence:
                continue
            themes.append(
                BriefingTheme(
                    name=name,
                    analysis=self._fallback_theme_analysis(evidence, technology, importance, maturity),
                    what_is_happening=self._evidence_summary(evidence),
                    core_technology=technology,
                    why_it_matters=importance,
                    maturity=maturity,
                    data_and_workflow=self._evidence_workflow(evidence),
                    source_urls=[source.url for source in evidence[:4]],
                )
            )
        if is_cad:
            residual = themed_sources.get("工程图与 CAD 工具链线索", [])
            if residual:
                themes.append(
                    BriefingTheme(
                        name="工程图与 CAD 工具链线索",
                        analysis=self._fallback_theme_analysis(
                            residual,
                            "从产品发布、部署经验和行业讨论中识别可重复的技术路线与约束。",
                            "它提供了需求侧和供给侧的信号，但需要继续回到技术实现和量化指标核验。",
                            "信息线索阶段，需补充原始技术材料和现场指标。",
                        ),
                        what_is_happening=self._evidence_summary(residual),
                        core_technology="从产品发布、部署经验和行业讨论中识别可重复的技术路线与约束。",
                        why_it_matters="它提供了需求侧和供给侧的信号，但需要继续回到技术实现和量化指标核验。",
                        maturity="信息线索阶段，需补充原始技术材料和现场指标。",
                        data_and_workflow=self._evidence_workflow(residual),
                        source_urls=[source.url for source in residual[:4]],
                    )
                )
        else:
            for name, _, technology, importance, maturity in GENERIC_EVIDENCE_THEMES:
                evidence = themed_sources.get(name, [])
                if not evidence:
                    continue
                themes.append(
                    BriefingTheme(
                        name=self._display_generic_evidence_theme_name(topic, name),
                        analysis=self._fallback_theme_analysis(evidence, technology, importance, maturity),
                        what_is_happening=self._evidence_summary(evidence),
                        core_technology=technology,
                        why_it_matters=importance,
                        maturity=maturity,
                        data_and_workflow=self._evidence_workflow(evidence),
                        source_urls=[source.url for source in evidence[:4]],
                    )
                )
        if not themes:
            themes.append(
                BriefingTheme(
                    name="待补充技术主题",
                    analysis="本轮没有足够的原始技术材料，不能据此还原具体实现路径或工程边界。",
                    what_is_happening="本轮公开检索没有返回可用于技术归纳的有效线索。",
                    core_technology="需要从原始论文、产品文档、代码仓库或公开演讲补齐。",
                    why_it_matters="没有足够原始材料时，不应把搜索空白误写成技术结论。",
                    maturity="未判定。",
                    data_and_workflow="尚无可用于还原输入、处理和回写流程的公开材料。",
                )
            )

        source_count = len(sources)
        theme_names = "、".join(theme.name for theme in themes[:4])
        if _is_cad_topic(topic):
            return BriefingSynthesis(
                search_content_summary=(
                    f"本轮围绕“{topic}”保留了 {source_count} 条高相关公开线索，内容可归为{theme_names}。"
                    "讨论的共同目标不是把文本或图片直接变成展示模型，而是把输入转成可编辑的 CAD 几何、二维图纸和可校验的工程约束。"
                ),
                short_summary=(
                    "这批线索指向同一条工程化路径：前端用语言、草图、扫描图或三维模型提供设计意图，"
                    "中间层生成参数化特征、投影视图与标注，末端以几何约束、公差和制图标准校验输出。"
                    "真正的技术门槛不在图像生成，而在生成结果能否继续编辑、版本化并进入 CAD/CAM/CAE 工具链。"
                ),
                detailed_summary=self._fallback_detailed_summary(themes, key_signal_interpretation=(
                    "本轮线索共同指向从视觉化概念模型走向可编辑、可校验工程实体的迁移；"
                    "后续需要核验每个方案是否输出 B-Rep 或原生特征树，以及约束、公差和制图规则是否由确定性工具执行。"
                )),
                themes=themes,
                key_signal_interpretation=(
                    "从来源中反复出现的 CAD、二维/三维互转、参数化和工程图关键词可以看出，"
                    "产品路线正在从“视觉上像零件”的网格生成，转向“拓扑、尺寸和约束可被工程软件读取”的结构化模型。"
                    "比较方案时应逐项核验：输入是否保留设计意图、输出是否为 B-Rep/特征树而非网格、"
                    "二维图的尺寸与公差是否可编辑、以及是否有规则校验阻止不可制造结果。"
                ),
                analysis_judgment=(
                    "跨来源的结论是，生成式模型适合缩短草图到初版几何、三维到二维出图和图纸数字化的准备时间，"
                    "但不应取代几何内核、约束求解器和制图规范。可落地方案需要把 LLM 或视觉模型限定在候选生成与语义提取，"
                    "再由 CAD API、规则引擎和工程师审批完成确定性校验与发布。"
                ),
                next_search_directions=[
                    "追踪 Text-to-CAD、B-Rep 生成、草图约束求解和工程图理解的原始论文、数据集与开源实现。",
                    "核验各方案导出的 STEP、DXF、DWG、B-Rep 或原生特征树是否能在目标 CAD 软件中继续编辑。",
                    "收集自动标注、投影视图和 GD&T 校验的评测集与人工复核指标，区分演示图与可发布工程图。",
                ],
                landing_suggestions=[
                    "先选取规则明确的零件族，建立草图、参数表、三维模型和二维工程图的成对基准集。",
                    "将 AI 输出限定为候选特征、视图和标注建议，由 CAD 几何内核执行约束求解、干涉检查和导出校验。",
                    "将图纸版本、尺寸变更、公差规则和审批结果写入可追溯数据模型，避免二维图与三维源模型脱节。",
                ],
            )
        return BriefingSynthesis(
            search_content_summary=(
                f"本轮围绕“{topic}”保留了 {source_count} 条公开线索，可分为{theme_names}。"
                "这些材料需要分层阅读：政策、规模和活动类内容提供产业环境信号；数据、开源、评测和案例材料才更接近实现证据。"
            ),
            short_summary=(
                f"本轮关于“{topic}”的有效阅读方式，是先把来源拆成宏观产业信号、技术底座、应用场景和弱证据四层。"
                "只有能说明数据输入、模型或工具链、系统接口、评估指标与人工接管机制的材料，才足以支撑技术落地判断。"
            ),
            detailed_summary=self._fallback_detailed_summary(themes, key_signal_interpretation=(
                "公开材料的证据强度并不相同：报告、政策和视频能说明关注度与投入方向，"
                "但是否真正形成产业能力，还要回到原始技术资料、数据集/评测、接口说明、客户案例和可复现指标。"
            )),
            themes=themes,
            key_signal_interpretation=(
                "多条线索同时出现时，应优先看它们是否共享同一数据闭环：输入来自何处、模型或规则如何做判断、"
                "结果如何写回业务/工程系统、以及失败后由谁接管。能回答这四个问题的项目，才可能从演示走向可运营能力。"
            ),
            analysis_judgment=(
                "跨来源归纳看，下一阶段的竞争不只是模型能力，而是行业语义、系统接口、评估指标和责任边界的产品化。"
                "因此应把“是否有 Agent/大模型”降为筛选条件，把“是否形成可审计的闭环与量化收益”升为排序条件。"
            ),
            next_search_directions=[
                "继续追踪每个主题的原始论文、产品技术文档、代码仓库和公开架构图。",
                "补齐数据输入、系统接口、评估指标和人工接管机制，验证是否形成可运营闭环。",
                "对重复出现的项目追踪后续版本、客户案例和部署限制，区分演示与规模化落地。",
            ],
            landing_suggestions=[
                "先选一个边界清晰、可量化的流程建立数据和评估基线。",
                "将模型输出限定为建议、工单或仿真方案，保留审批、回滚和完整审计记录。",
                "优先建设来源、实体、指标和项目之间的关联库，让后续日报可做连续对比。",
            ],
        )

    @staticmethod
    def _display_generic_evidence_theme_name(topic: str, generic_name: str) -> str:
        """Render a generic evidence theme as '{topic}的{suffix}'.

        The fallback path keeps stable, predictable headings.  Trying to make
        them evidence-driven with a hand-maintained generic-word list proved
        to be a maintenance trap (every new generic expression needed a new
        entry, and the list could also filter out real technical words), so
        the fallback uses fixed suffixes instead.  Report quality is handled
        by avoiding the fallback in the first place (dedicated synthesis
        timeout) and by auditing degradation (synthesis_source=fallback).
        """
        clean_topic = " ".join(topic.split()).strip(" ：:，,。")
        if not clean_topic:
            clean_topic = "本主题"
        suffixes = {
            "政策、规模与产业链信号": "政策、规模与产业链信号",
            "技术底座、数据与开源生态": "技术底座、数据与开源生态",
            "应用落地与业务转型案例": "应用落地与业务流程线索",
            "教育传播、公众讨论与弱证据线索": "传播讨论与弱证据线索",
            "综合产业动态与待核验证据": "待核验证据线索",
        }
        suffix = suffixes.get(generic_name, "证据线索")
        return f"{clean_topic}的{suffix}"

    def _generic_evidence_theme_name(self, source: CollectedSource) -> str:
        text = f"{source.title} {source.snippet} {source.platform} {source.provider}".lower()
        best_name = "综合产业动态与待核验证据"
        best_score = 0
        for name, markers, *_ in GENERIC_EVIDENCE_THEMES:
            if not markers:
                continue
            score = sum(1 for marker in markers if self._matches_technical_marker(text, marker.lower()))
            if score > best_score:
                best_name = name
                best_score = score
        return best_name

    def _fallback_theme_analysis(
        self,
        evidence: list[CollectedSource],
        technology: str,
        importance: str,
        maturity: str,
    ) -> str:
        evidence_summary = self._evidence_summary(evidence)
        workflow = self._evidence_workflow(evidence)
        return (
            f"{evidence_summary}\n\n"
            f"能够支撑的工作假设是：{technology}{workflow}\n\n"
            f"工程含义是：{importance}证据边界是：{maturity} "
            "因此不应把公开线索直接写成已验证的规模化能力。"
        )

    @staticmethod
    def _fallback_detailed_summary(
        themes: list[BriefingTheme],
        *,
        key_signal_interpretation: str,
    ) -> str:
        """Render a readable fallback without resurrecting the old field checklist."""
        blocks: list[str] = []
        for theme in themes:
            body = theme.analysis.strip()
            if not body:
                details = [
                    theme.what_is_happening.strip(),
                    theme.core_technology.strip(),
                    theme.data_and_workflow.strip(),
                    theme.why_it_matters.strip(),
                    theme.maturity.strip(),
                ]
                body = " ".join(part for part in details if part)
            if body:
                blocks.extend([f"### {theme.name}", body, ""])
        if key_signal_interpretation.strip():
            blocks.extend(["### 仍需核验的技术分界线", key_signal_interpretation.strip()])
        return "\n".join(blocks).strip()

    def render_markdown(self, briefing: DailyBriefing) -> str:
        synthesis = briefing.synthesis
        lines = [
            "# 内容搜集报告",
            "",
            f"- 选题：{briefing.topic}",
            f"- 日期：{briefing.run_date.isoformat()}",
            f"- 原始线索：{len(briefing.sources)} 条",
            "",
            "## 搜索方向",
            *[f"- {direction}" for direction in briefing.search_directions],
            "",
            "## 关键词",
            ", ".join(briefing.keywords),
            "",
            "## 搜索内容总结",
            synthesis.search_content_summary,
            "",
            "## 简短总结",
            synthesis.short_summary,
            "",
            "## 详细总结",
            "",
        ]
        detailed_summary = synthesis.detailed_summary.strip() or self._fallback_detailed_summary(
            synthesis.themes,
            key_signal_interpretation=synthesis.key_signal_interpretation,
        )
        detailed_summary = self._compact_detailed_summary_body(detailed_summary)
        lines.extend(detailed_summary.splitlines())
        lines.extend(
            [
                "",
                "## AI 分析判断",
                synthesis.analysis_judgment,
                "",
                "## 下一步搜索方向",
                *[f"- {direction}" for direction in synthesis.next_search_directions],
                "",
                "## 落地建议",
                *[f"- {suggestion}" for suggestion in synthesis.landing_suggestions],
                "",
                "## 原文链接",
            ]
        )
        for index, source in enumerate(briefing.sources, start=1):
            lines.append(f"{index}. [{source.title}]({source.url}) | {source.platform} | 重要度 {source.importance_score:.1f}")
        return "\n".join(lines) + "\n"

    @classmethod
    def _compact_detailed_summary_body(
        cls,
        markdown: str,
        *,
        max_paragraph_chars: int = 220,
        max_paragraphs_per_heading: int = 2,
    ) -> str:
        """Shorten detailed-summary prose while preserving model-selected headings."""
        lines = markdown.splitlines()
        blocks: list[tuple[str, list[str]]] = []
        current_heading: str | None = None
        current_body: list[str] = []
        preamble: list[str] = []

        for line in lines:
            if line.startswith("### "):
                if current_heading is not None:
                    blocks.append((current_heading, current_body))
                elif current_body:
                    preamble.extend(current_body)
                current_heading = line
                current_body = []
            else:
                current_body.append(line)
        if current_heading is not None:
            blocks.append((current_heading, current_body))
        elif current_body:
            preamble.extend(current_body)

        seen_sentences: set[str] = set()
        output: list[str] = []
        if preamble:
            output.extend(cls._compact_body_lines(preamble, seen_sentences, max_paragraph_chars, 1))
        for heading, body in blocks:
            compact_body = cls._compact_body_lines(
                body,
                seen_sentences,
                max_paragraph_chars,
                max_paragraphs_per_heading,
            )
            output.append(heading)
            output.extend(compact_body or ["本节材料较少，保留为待核验线索。"])
            output.append("")
        return "\n".join(output).strip()

    @classmethod
    def _compact_body_lines(
        cls,
        lines: list[str],
        seen_sentences: set[str],
        max_paragraph_chars: int,
        max_paragraphs: int,
    ) -> list[str]:
        text = "\n".join(lines).strip()
        if not text:
            return []
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
        compact: list[str] = []
        for paragraph in paragraphs:
            shortened = cls._compact_paragraph_without_repetition(
                paragraph,
                seen_sentences,
                max_chars=max_paragraph_chars,
            )
            if shortened:
                compact.append(shortened)
            if len(compact) >= max_paragraphs:
                break
        return compact

    @staticmethod
    def _compact_paragraph_without_repetition(
        paragraph: str,
        seen_sentences: set[str],
        *,
        max_chars: int,
    ) -> str:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[。！？；;])", " ".join(paragraph.split()))
            if sentence.strip()
        ]
        if not sentences:
            sentences = [" ".join(paragraph.split()).strip()]
        kept: list[str] = []
        for sentence in sentences:
            fingerprint = re.sub(r"\W+", "", sentence.lower())[:60]
            if fingerprint and fingerprint in seen_sentences:
                continue
            if fingerprint:
                seen_sentences.add(fingerprint)
            if len("".join(kept)) + len(sentence) > max_chars and kept:
                break
            kept.append(sentence)
        text = "".join(kept) or sentences[0]
        if len(text) > max_chars:
            text = text[: max_chars - 1].rstrip("，,；;。 ") + "…"
        return text

    @staticmethod
    def _normalize_url(url: str) -> str:
        value = url.strip()
        if not value.startswith(("https://", "http://")):
            return ""
        return value.rstrip("/")

    def _briefing_date(self) -> date:
        try:
            return datetime.now(ZoneInfo(self.timezone_name)).date()
        except ZoneInfoNotFoundError:
            return datetime.now(UTC).date()

    @staticmethod
    def _compact_source_text(value: str, limit: int) -> str:
        return " ".join(str(value or "").split())[:limit]

    @staticmethod
    def _is_search_page_dump(title: str, snippet: str) -> bool:
        text = f"{title} {snippet}".lower()
        return any(marker in text for marker in _SEARCH_DUMP_MARKERS)

    @staticmethod
    def _platform_from_url(url: str, fallback: str) -> str:
        host = urlparse(url).netloc.lower()
        mappings = {
            "bilibili.com": "哔哩哔哩",
            "douyin.com": "抖音",
            "mp.weixin.qq.com": "微信公众号",
            "toutiao.com": "今日头条",
            "xiaohongshu.com": "小红书",
            "linkedin.com": "LinkedIn",
            "reddit.com": "Reddit",
            "youtube.com": "YouTube",
            "x.com": "X",
            "zhihu.com": "知乎",
        }
        return next((name for domain, name in mappings.items() if domain in host), fallback)

    @staticmethod
    def _keywords(
        topic: str,
        feedback: list[dict[str, object]],
        search_plan: list[tuple[str, str]],
        intent: BriefingIntentProfile | None = None,
        understanding: BriefingUnderstanding | None = None,
    ) -> list[str]:
        candidates = [topic]
        if understanding is not None:
            profile = understanding.domain_profile
            candidates.extend(understanding.research_focus)
            candidates.extend(profile.key_concepts)
            candidates.extend(profile.subtopics)
            candidates.extend(profile.evidence_anchors)
            candidates.extend(profile.preferred_source_types)
        elif intent is not None:
            candidates.extend(intent.search_focus)
        candidates.extend(
            query
            for platform, query in search_plan
            if platform == "技术路线" and _is_meaningful_query(query)
        )
        for item in feedback[:3]:
            note = " ".join(str(item.get("body") or "").split())
            if _is_meaningful_feedback(note):
                candidates.extend(re.findall(r"[A-Za-z][A-Za-z0-9+._/-]{1,}|[\u4e00-\u9fff]{2,}", note))
        keywords: list[str] = []
        for candidate in candidates:
            normalized = candidate.strip()
            if normalized and _is_meaningful_query(normalized) and normalized not in keywords:
                keywords.append(normalized)
            if len(keywords) >= 12:
                break
        return keywords or [topic]

    @staticmethod
    def _relevance_score(topic: str, title: str, snippet: str) -> float:
        terms = [term.lower() for term in re.findall(r"[A-Za-z0-9+._/-]{2,}|[\u4e00-\u9fff]{2,}", topic)]
        haystack = f"{title} {snippet}".lower()
        return float(sum(1 for term in terms if term in haystack))

    @staticmethod
    def _has_exact_cjk_topic_phrase(topic: str, title: str, snippet: str) -> bool:
        phrases = re.findall(r"[\u4e00-\u9fff]{4,}", topic)
        if not phrases:
            return False
        haystack = f"{title} {snippet}"
        return any(phrase in haystack for phrase in phrases)

    @staticmethod
    def _cjk_topic_match_score(topic: str, title: str, snippet: str) -> int:
        haystack = f"{title} {snippet}"
        best_score = 0
        for phrase in re.findall(r"[\u4e00-\u9fff]{3,}", topic):
            if phrase in haystack:
                best_score = max(best_score, 4)
                continue
            chunks = {phrase[index : index + 2] for index in range(0, len(phrase) - 1)}
            matched_chunks = {chunk for chunk in chunks if chunk in haystack}
            best_score = max(best_score, len(matched_chunks))
        return best_score

    @staticmethod
    def _query_relevance_score(query: str, title: str, snippet: str) -> float:
        terms = [
            term.lower()
            for term in re.findall(r"[A-Za-z][A-Za-z0-9+._/-]{1,}|[\u4e00-\u9fff]{2,}", query)
            if term.lower() not in _QUERY_STOPWORDS
        ]
        haystack = f"{title} {snippet}".lower()
        return float(sum(1 for term in dict.fromkeys(terms) if term in haystack))

    def _source_quality_verdict(
        self,
        *,
        topic: str,
        query: str,
        url: str,
        title: str,
        snippet: str,
        provider: str,
        understanding: BriefingUnderstanding,
    ) -> SourceQualityVerdict:
        prefilter = self._deterministic_source_prefilter_rejection(url, title, snippet)
        if prefilter is not None:
            return SourceQualityVerdict(
                technical_value="low",
                source_type="aggregator" if prefilter[0] == "rejected_search_page_dump" else "unknown",
                marketing_level="none",
                evidence_density="low",
                authority="weak",
                keep_recommendation="reject",
                rationale=prefilter[1],
            )

        obvious = self._obvious_source_quality_verdict(topic, query, url, title, snippet)
        if obvious is not None:
            return obvious

        if self.runtime is not None:
            try:
                raw = self.runtime.run_text_task(
                    SOURCE_QUALITY_JUDGE_INSTRUCTIONS,
                    self._source_quality_payload(
                        topic=topic,
                        query=query,
                        url=url,
                        title=title,
                        snippet=snippet,
                        provider=provider,
                        understanding=understanding,
                    ),
                    temperature=0.0,
                    max_tokens=500,
                    timeout_seconds=20.0,
                )
                parsed = _parse_source_quality_verdict(raw)
                if parsed is not None:
                    return parsed
            except Exception:
                pass
        return self._fallback_source_quality_verdict(topic, query, url, title, snippet)

    @staticmethod
    def _source_quality_payload(
        *,
        topic: str,
        query: str,
        url: str,
        title: str,
        snippet: str,
        provider: str,
        understanding: BriefingUnderstanding,
    ) -> dict[str, object]:
        profile = understanding.domain_profile
        return {
            "topic": topic,
            "query": query,
            "url": url,
            "domain": urlparse(url).netloc.lower().removeprefix("www."),
            "title": title,
            "snippet": snippet,
            "provider": provider,
            "briefing_understanding_summary": {
                "primary_intent": understanding.primary_intent,
                "secondary_intents": understanding.secondary_intents,
                "temporal_focus": understanding.temporal_focus,
                "domain": profile.domain,
                "research_focus": understanding.research_focus[:5],
                "evidence_preferences": understanding.evidence_preferences[:5],
                "preferred_source_types": profile.preferred_source_types[:5],
            },
        }

    @classmethod
    def _obvious_source_quality_verdict(
        cls,
        topic: str,
        query: str,
        url: str,
        title: str,
        snippet: str,
    ) -> SourceQualityVerdict | None:
        if source_authority_weight(url) < _PRIMARY_SOURCE_WEIGHT:
            return None
        if not (
            cls._relevance_score(topic, title, snippet) > 0
            or cls._query_relevance_score(query, title, snippet) > 0
            or cls._has_exact_cjk_topic_phrase(topic, title, snippet)
            or cls._cjk_topic_match_score(topic, title, snippet) >= 2
        ):
            return None
        return SourceQualityVerdict(
            technical_value="high",
            source_type="primary",
            marketing_level="none",
            evidence_density="high",
            authority="primary",
            keep_recommendation="keep",
            rationale="authoritative source with visible topic/query relevance",
        )

    @classmethod
    def _fallback_source_quality_verdict(
        cls,
        topic: str,
        query: str,
        url: str,
        title: str,
        snippet: str,
    ) -> SourceQualityVerdict:
        # Legacy fallback only: used when no LLM judge is available or parsing
        # fails. Do not extend the old marketing/technical phrase inventories;
        # the active boundary judge is the structured SourceQualityVerdict path.
        if cls._is_marketing_account_content(url, title, snippet):
            return SourceQualityVerdict(
                technical_value="low",
                source_type="marketing",
                marketing_level="dominant",
                evidence_density="low",
                authority="weak",
                keep_recommendation="reject",
                rationale="legacy fallback detected dominant marketing funnel content",
            )
        topic_score = cls._relevance_score(topic, title, snippet)
        query_score = cls._query_relevance_score(query, title, snippet)
        technical_hits = cls._technical_signal_count(title, snippet)
        relevant = (
            topic_score >= 2
            or query_score >= 2
            or cls._has_exact_cjk_topic_phrase(topic, title, snippet)
            or cls._cjk_topic_match_score(topic, title, snippet) >= 2
            or technical_hits > 0
        )
        if not relevant:
            return SourceQualityVerdict(
                technical_value="low",
                source_type="unknown",
                marketing_level="mixed",
                evidence_density="low",
                authority="weak",
                keep_recommendation="reject",
                rationale="legacy fallback found insufficient topic, query, or technical signal",
            )
        authority = "primary" if source_authority_weight(url) >= _PRIMARY_SOURCE_WEIGHT else "secondary"
        return SourceQualityVerdict(
            technical_value="high" if technical_hits >= 2 or authority == "primary" else "medium",
            source_type="primary" if authority == "primary" else "secondary",
            marketing_level="none",
            evidence_density="high" if authority == "primary" or technical_hits >= 2 else "medium",
            authority=authority,
            keep_recommendation="keep" if technical_hits or authority == "primary" else "borderline",
            rationale="legacy deterministic fallback retained a relevant source",
        )

    @staticmethod
    def _matches_technical_marker(text: str, marker: str) -> bool:
        if marker.isascii() and re.fullmatch(r"[a-z0-9]+", marker):
            return bool(re.search(rf"\b{re.escape(marker)}\b", text))
        return marker in text

    @staticmethod
    def _is_generic_reference_domain(url: str) -> bool:
        host = urlparse(url).netloc.lower().removeprefix("www.")
        return any(host == domain or host.endswith(f".{domain}") for domain in _GENERIC_REFERENCE_DOMAINS)

    @staticmethod
    def _is_marketing_domain(url: str) -> bool:
        host = urlparse(url).netloc.lower().removeprefix("www.")
        return any(host == domain or host.endswith(f".{domain}") for domain in _MARKETING_CONTENT_DOMAINS)

    @classmethod
    def _is_marketing_account_content(cls, url: str, title: str, snippet: str) -> bool:
        # Legacy fallback only. The active source-quality boundary path uses
        # SourceQualityVerdict from the hybrid LLM judge and does not expand
        # this phrase-based marketing classifier.
        text = f"{title} {snippet}".lower()
        compact_text = re.sub(r"\s+", "", text)
        strong_hits = sum(
            1 for phrase in _MARKETING_STRONG_PHRASES if phrase in text or phrase in compact_text
        )
        weak_hits = sum(1 for phrase in _MARKETING_WEAK_PHRASES if phrase in text or phrase in compact_text)
        evidence_hits = sum(1 for phrase in _SOURCE_EVIDENCE_PHRASES if phrase in text or phrase in compact_text)
        technical_hits = cls._technical_signal_count(title, snippet)
        on_marketing_prone_domain = cls._is_marketing_domain(url)

        if strong_hits >= 2:
            return True
        if on_marketing_prone_domain and strong_hits >= 1 and weak_hits >= 1:
            return True
        if on_marketing_prone_domain and strong_hits >= 1 and evidence_hits == 0 and technical_hits == 0:
            return True
        if weak_hits >= 3 and evidence_hits == 0 and technical_hits == 0:
            return True
        return False

    @classmethod
    def _technical_signal_count(cls, title: str, snippet: str) -> int:
        # Legacy fallback/scoring compatibility only. Do not add domain-specific
        # markers here to handle source-quality bad cases; use
        # SourceQualityVerdict instead.
        text = f"{title} {snippet}".lower()
        theme_hits = sum(
            1
            for _, markers, *_ in CAD_THEMES + THEMES
            if any(cls._matches_technical_marker(text, marker) for marker in markers if marker != "\u5de5\u7a0b")
        )
        explicit_hits = sum(cls._matches_technical_marker(text, marker) for marker in _TECHNICAL_SIGNAL_MARKERS)
        return theme_hits + explicit_hits

    @staticmethod
    def _requires_industrial_anchor(topic: str, query: str) -> bool:
        text = f"{topic} {query}".lower()
        return any(marker in text for marker in ("industrial", "manufactur", "\u5de5\u4e1a", "\u5236\u9020"))

    @staticmethod
    def _has_industrial_anchor(title: str, snippet: str) -> bool:
        text = f"{title} {snippet}".lower()
        return any(marker in text for marker in _INDUSTRIAL_ANCHORS) or any(
            re.search(rf"\b{re.escape(marker)}\b", text) for marker in _INDUSTRIAL_ACRONYM_ANCHORS
        )

    @classmethod
    def _is_report_source_candidate(
        cls,
        topic: str,
        query: str,
        url: str,
        title: str,
        snippet: str,
    ) -> bool:
        # Legacy fallback/API compatibility. The daily collection path now uses
        # deterministic prefiltering plus SourceQualityVerdict and deterministic
        # rank/quota decisions.
        if cls._is_generic_reference_domain(url):
            return False
        host = urlparse(url).netloc.lower().removeprefix("www.")
        path = urlparse(url).path.lower()
        if host in {"baidu.com", "m.baidu.com"} or any(marker in path for marker in _LOGIN_PATH_MARKERS):
            return False
        text = f"{title} {snippet}".lower()
        if cls._is_search_page_dump(title, snippet):
            return False
        if cls._is_marketing_account_content(url, title, snippet):
            return False
        if _is_cad_topic(f"{topic} {query}") and any(marker in text for marker in _CAD_MEDICAL_MARKERS):
            return False
        if _is_cad_topic(f"{topic} {query}") and not _has_cad_anchor(title, snippet):
            return False
        if cls._requires_industrial_anchor(topic, query) and not cls._has_industrial_anchor(title, snippet):
            return False
        topic_score = cls._relevance_score(topic, title, snippet)
        query_score = cls._query_relevance_score(query, title, snippet)
        if cls._has_exact_cjk_topic_phrase(topic, title, snippet):
            return True
        if cls._cjk_topic_match_score(topic, title, snippet) >= 2:
            return True
        if _is_cad_topic(f"{topic} {query}"):
            return topic_score >= 2 or query_score >= 2 or (
                cls._technical_signal_count(title, snippet) > 0 and max(topic_score, query_score) >= 1
            )
        return topic_score >= 2 or query_score >= 2 or cls._technical_signal_count(title, snippet) > 0

    @classmethod
    def _deterministic_source_prefilter_rejection(
        cls,
        url: str,
        title: str,
        snippet: str,
    ) -> tuple[str, str] | None:
        if not url:
            return "rejected_invalid_url", "missing or non-http URL"
        if not title and not snippet:
            return "rejected_empty_content", "title and snippet are both empty"
        if cls._is_search_page_dump(title, snippet):
            return "rejected_search_page_dump", "result looked like a search-result page dump"
        text = f"{title} {snippet}".lower()
        if any(marker in text for marker in _PROMPT_INJECTION_MARKERS):
            return "rejected_low_relevance", "result contained prompt-injection instructions"
        if cls._is_generic_reference_domain(url):
            return "rejected_generic_reference", "generic reference domain is not strong briefing evidence"
        parsed = urlparse(url)
        host = parsed.netloc.lower().removeprefix("www.")
        path = parsed.path.lower()
        if host in {"baidu.com", "m.baidu.com"}:
            return "rejected_search_page_dump", "Baidu wrapper URL is not an original public source"
        if any(marker in path for marker in _LOGIN_PATH_MARKERS):
            return "rejected_login_page", "login-gated URL path is outside public-source boundary"
        return None

    @classmethod
    def _source_rejection_reason(
        cls,
        topic: str,
        query: str,
        url: str,
        title: str,
        snippet: str,
        verdict: SourceQualityVerdict | None = None,
    ) -> tuple[str, str] | None:
        prefilter = cls._deterministic_source_prefilter_rejection(url, title, snippet)
        if prefilter is not None:
            return prefilter
        if verdict is not None:
            if verdict.keep_recommendation == "reject":
                return "rejected_low_relevance", f"source quality judge rejected source: {verdict.rationale}"
            if (
                verdict.technical_value == "low"
                and verdict.evidence_density == "low"
                and verdict.marketing_level == "dominant"
            ):
                return "rejected_low_relevance", f"source quality judge found dominant marketing: {verdict.rationale}"
            return None

        # Legacy fallback branch only for older direct callers/tests.
        text = f"{title} {snippet}".lower()
        if _is_cad_topic(f"{topic} {query}") and any(marker in text for marker in _CAD_MEDICAL_MARKERS):
            return "rejected_cad_medical", "CAD query matched medical CAD acronym content"
        if _is_cad_topic(f"{topic} {query}") and not _has_cad_anchor(title, snippet):
            return "rejected_cad_missing_anchor", "CAD query lacked engineering CAD anchors"
        if cls._requires_industrial_anchor(topic, query) and not cls._has_industrial_anchor(title, snippet):
            return "rejected_missing_industrial_anchor", "industrial query lacked manufacturing or industrial anchors"
        if not cls._is_report_source_candidate(topic, query, url, title, snippet):
            return "rejected_low_relevance", "insufficient topic, query, or technical signal"
        return None

    def _source_candidate(
        self,
        *,
        subscription: TopicSubscription,
        requested_platform: str,
        query: str,
        result: SearchResult,
        normalized_url: str,
        title: str,
        snippet: str,
        status: str,
        reason: str,
        relevance_score: float,
        importance_score: float,
    ) -> SourceCandidate:
        return SourceCandidate(
            id=f"cand_{uuid.uuid4().hex}",
            topic_id=subscription.id,
            user_id=subscription.user_id,
            title=title,
            url=normalized_url,
            snippet=snippet,
            platform=self._platform_from_url(normalized_url, requested_platform) if normalized_url else requested_platform,
            provider=result.provider,
            query=query,
            status=status,  # type: ignore[arg-type]
            reason=reason,
            relevance_score=relevance_score,
            importance_score=importance_score,
            retrieved_at=result.checked_at,
            created_at=datetime.now(UTC).isoformat(),
        )

    def _importance_score(
        self,
        topic: str,
        query: str,
        title: str,
        snippet: str,
        provider: str,
        verdict: SourceQualityVerdict | None = None,
    ) -> float:
        provider_bonus = 2.0 if provider.startswith("direct-") else 1.0 if provider.startswith("mcp:") else 0.0
        if verdict is not None:
            return (
                self._relevance_score(topic, title, snippet) * 2
                + self._query_relevance_score(query, title, snippet) * 2
                + self._source_quality_score(verdict)
                + provider_bonus
            )
        text = f"{title} {snippet}".lower()
        technical_hits = self._technical_signal_count(title, snippet)
        cad_hits = sum(self._matches_technical_marker(text, marker) for marker in _CAD_EVIDENCE_ANCHORS)
        evidence_hits = sum(marker in text for marker in ("paper", "论文", "official", "官方", "release", "发布", "benchmark", "案例", "架构"))
        return (
            self._relevance_score(topic, title, snippet) * 2
            + self._query_relevance_score(query, title, snippet) * 2
            + technical_hits * 3
            + cad_hits * 2
            + evidence_hits
            + provider_bonus
        )

    @staticmethod
    def _source_quality_score(verdict: SourceQualityVerdict) -> float:
        technical = {"high": 8.0, "medium": 4.0, "low": 0.0}[verdict.technical_value]
        evidence = {"high": 5.0, "medium": 2.0, "low": 0.0}[verdict.evidence_density]
        authority = {"primary": 6.0, "secondary": 3.0, "weak": 0.0}[verdict.authority]
        source_type = {
            "primary": 2.0,
            "secondary": 1.0,
            "community": 1.0,
            "unknown": 0.0,
            "marketing": -2.0,
            "aggregator": -3.0,
        }[verdict.source_type]
        marketing = {"none": 0.0, "mixed": -2.0, "dominant": -8.0}[verdict.marketing_level]
        keep = {"keep": 2.0, "borderline": 0.0, "reject": -10.0}[verdict.keep_recommendation]
        return technical + evidence + authority + source_type + marketing + keep

    @staticmethod
    def _evidence_workflow(sources: list[CollectedSource]) -> str:
        excerpts = [
            DailyBriefingService._short_evidence_excerpt(source.snippet)
            for source in sources
            if source.snippet
        ]
        excerpts = [excerpt for excerpt in excerpts if excerpt]
        if not excerpts:
            return " 来源未披露完整数据闭环；下一步应补查输入数据、系统接口、执行动作和人工接管点。"
        return (
            " 公开片段提供的可用线索包括："
            + "；".join(excerpts[:3])
            + "。但这些片段仍不足以完整证明输入、处理、接口写回和异常接管机制。"
        )

    @staticmethod
    def _evidence_summary(sources: list[CollectedSource]) -> str:
        titles: list[str] = []
        seen: set[str] = set()
        for source in sources:
            title = DailyBriefingService._trim_source_title(source.title)
            key = title.lower()
            if not title or key in seen:
                continue
            titles.append(title)
            seen.add(key)
            if len(titles) >= 4:
                break
        if not titles:
            return "本组暂无可读标题"
        suffix = "等" if len(sources) > len(titles) else ""
        return "主要证据包括：" + "；".join(f"《{title}》" for title in titles) + suffix + "。"

    @staticmethod
    def _short_evidence_excerpt(value: str, max_chars: int = 150) -> str:
        text = " ".join(value.split()).strip()
        if not text:
            return ""
        text = re.sub(r"作者：.*$", "", text).strip()
        text = re.sub(r"[-—]{2,}.*$", "", text).strip()
        candidates = [part.strip() for part in re.split(r"[。！？；\n]+", text) if part.strip()]
        if candidates:
            text = max(candidates[:3], key=len)
        if len(text) < 8 or not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", text):
            return ""
        if len(text) > max_chars:
            text = text[: max_chars - 1].rstrip("，,；;：:、 ") + "…"
        return text

    @staticmethod
    def _trim_source_title(value: str, max_chars: int = 46) -> str:
        title = " ".join(value.split()).strip()
        title = re.sub(r"[_\-—|].*$", "", title).strip() or title
        if len(title) > max_chars:
            return title[: max_chars - 1].rstrip("，,；;：:、 ") + "…"
        return title
