from __future__ import annotations

import re
import uuid
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from search_assistant.contracts import (
    DailyBriefing,
    DomainKnowledgeEntity,
    DomainKnowledgeEntityType,
    DomainKnowledgeGraph,
    DomainKnowledgeRelation,
)
from search_assistant.knowledge_graph.bm25 import tokenize
from search_assistant.memory.store import MemoryStore


# Terms that appear too often across topics to be meaningful graph entities.
_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "are",
    "was",
    "been",
    "into",
    "over",
    "than",
    "then",
    "they",
    "them",
    "their",
    "have",
    "has",
    "had",
    "will",
    "would",
    "could",
    "should",
    "about",
    "after",
    "before",
    "between",
    "through",
    "during",
    "using",
    "used",
    "use",
    "new",
    "also",
    "one",
    "two",
    "three",
    "based",
    "via",
    "yet",
    "its",
    "our",
    "your",
    "如何",
    "一个",
    "以及",
    "可以",
    "能够",
    "进行",
    "通过",
    "相关",
    "系统",
    "技术",
    "领域",
    "目前",
    "当前",
    "未来",
    "已经",
    "我们",
    "他们",
    "这些",
    "那些",
    "其中",
    "关于",
    "由于",
    "因为",
    "所以",
    "但是",
    "以及",
    "不是",
    "没有",
    "需要",
    "可能",
    "主要",
    "重要",
    "关键",
    "解决",
    "问题",
    "方案",
    "实现",
    "应用",
    "发展",
    "研究",
    "分析",
    "报告",
    "平台",
    "工具",
    "框架",
}

_MIN_TERM_LEN = 2
_MAX_TERM_LEN = 40
_MIN_OCCURRENCES = 3

# Upper bound for a single auto graph; further merges trim to the strongest
# entities by occurrence count so retrieval stays focused.
_MAX_AUTO_GRAPH_ENTITIES = 60

_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_MAX_CJK_RUN_LEN = 10

# Terms that are too generic to guide retrieval even when they recur.
_GENERIC_ENTITY_NAMES = frozenset(
    {
        "ai",
        "api",
        "data",
        "llm",
        "model",
        "models",
        "system",
        "systems",
        "use",
        "used",
        "using",
        "模型",
        "部署",
        "设计",
        "产品",
        "协议",
    }
    | _STOP_WORDS
)


def _entity_type(name: str) -> DomainKnowledgeEntityType:
    lowered = name.lower()
    if any(marker in lowered for marker in ("metric", "指标", "延迟", "吞吐", "成本", "精度", "召回")):
        return "metric"
    if any(marker in lowered for marker in ("risk", "风险", "威胁", "隐患", "安全")):
        return "risk"
    if any(marker in lowered for marker in ("system", "平台", "框架", "引擎", "库", "runtime", "sdk")):
        return "system"
    if any(marker in lowered for marker in ("data", "数据集", "语料", "benchmark", "基准")):
        return "data"
    if any(marker in lowered for marker in ("workflow", "pipeline", "流程", "管线", "链路")):
        return "workflow"
    return "concept"


def _meaningful_term(term: str) -> bool:
    if len(term) < _MIN_TERM_LEN or len(term) > _MAX_TERM_LEN:
        return False
    if re.fullmatch(r"\d+([.,]\d+)?%?", term):
        return False
    lowered = term.lower()
    if lowered in _STOP_WORDS:
        return False
    # A token must contain at least one letter or CJK character.
    if not re.search(r"[a-z]|[\u4e00-\u9fff]", lowered):
        return False
    return True


_CJK_BIGRAM_RE = re.compile(r"[\u4e00-\u9fff]{2}")


def is_cjk_fragment(term: str, cjk_runs: list[str]) -> bool:
    """True when a 2-char CJK term is a fragment of a longer CJK run.

    The shared tokenizer chunks CJK text into overlapping bigrams, so
    "智能体" produces both "智能" and "能体".  These fragments are poor graph
    entities; when the full run is present in the same text they are dropped
    while the run itself becomes the entity.
    """
    if not re.fullmatch(r"[\u4e00-\u9fff]{2}", term):
        return False
    return any(len(run) > 2 and term in run for run in cjk_runs)


def is_meaningful_entity_name(name: str, sibling_names: set[str] | None = None) -> bool:
    """Whether an entity name is specific enough to guide retrieval.

    Rejects single characters, generic terms, 2-char CJK fragments that are
    substrings of a longer sibling entity, and spaced names whose two parts are
    both 2-char CJK pieces (overlapping window artifacts such as "大模 模型").
    """
    stripped = name.strip()
    if len(stripped) < 2:
        return False
    if stripped.lower() in _GENERIC_ENTITY_NAMES:
        return False
    if " " in stripped:
        parts = [part for part in stripped.split() if part]
        if len(parts) == 2 and all(_CJK_BIGRAM_RE.fullmatch(part) for part in parts):
            return False
    if re.fullmatch(r"[\u4e00-\u9fff]{2}", stripped):
        if sibling_names is not None:
            for sibling in sibling_names:
                if sibling != stripped and stripped in sibling:
                    return False
    return True


def _candidate_terms(text: str) -> list[str]:
    """Candidate entity terms extracted from one text field.

    English words and word bigrams come from the shared tokenizer; CJK content
    is additionally matched as whole runs (2-10 characters) so that "智能体"
    survives as one entity while its overlapping bigram fragments ("智能",
    "能体") are dropped.
    """
    tokens = [token for token in tokenize(text) if _meaningful_term(token)]
    cjk_runs = [run for run in _CJK_RUN_RE.findall(text) if len(run) <= _MAX_CJK_RUN_LEN]
    kept_tokens = [token for token in tokens if not is_cjk_fragment(token, cjk_runs)]
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        if term not in seen:
            seen.add(term)
            terms.append(term)

    for token in kept_tokens:
        add(token)
    for left, right in zip(kept_tokens, kept_tokens[1:]):
        # Skip overlapping CJK window pairs ("大模 模型", "端视 视觉"): they
        # are artifacts of bigram chunking, not meaningful phrases.  Latin-Latin
        # ("persistent memory") and Latin-CJK ("a2a 协议") pairs stay.
        if _CJK_BIGRAM_RE.fullmatch(left) and _CJK_BIGRAM_RE.fullmatch(right):
            continue
        bigram = f"{left} {right}"
        if _meaningful_term(bigram):
            add(bigram)
    for run in cjk_runs:
        if _meaningful_term(run):
            add(run)
    return terms


def _collect_texts(briefing: DailyBriefing) -> list[str]:
    """Collect the free-text fields that should contribute candidate terms."""
    texts: list[str] = []
    synthesis = briefing.synthesis
    texts.extend(
        [
            synthesis.search_content_summary,
            synthesis.short_summary,
            synthesis.detailed_summary,
            synthesis.key_signal_interpretation,
            synthesis.analysis_judgment,
        ]
    )
    texts.extend(briefing.search_directions)
    texts.extend(briefing.keywords)
    for theme in synthesis.themes:
        texts.extend([theme.name, theme.analysis, theme.what_is_happening])
    for source in briefing.sources:
        texts.extend([source.title, source.snippet])
    return [text for text in texts if text]


def extract_entities(
    briefing: DailyBriefing,
    min_occurrences: int = _MIN_OCCURRENCES,
) -> list[DomainKnowledgeEntity]:
    """Extract candidate entities from a briefing trajectory by term frequency.

    English words and word bigrams are counted alongside whole CJK runs, so
    meaningful phrases such as "persistent memory" or "受控工单编排" surface as
    entities.  Overlapping CJK bigram fragments (e.g. "能体" inside "智能体")
    are dropped.  Only terms that appear in at least ``min_occurrences``
    different fields are kept, filtering out one-off vocabulary while retaining
    recurring domain concepts.
    """
    texts = _collect_texts(briefing)
    counter: Counter[str] = Counter()
    for text in texts:
        for term in set(_candidate_terms(text)):
            counter[term] += 1

    now = datetime.now(UTC).isoformat()
    entities: list[DomainKnowledgeEntity] = []
    for term, count in counter.most_common():
        if count < min_occurrences:
            break  # counter is sorted by frequency; once below threshold, stop.
        entities.append(
            DomainKnowledgeEntity(
                id=f"auto_entity_{uuid.uuid4().hex[:12]}",
                name=term,
                entity_type=_entity_type(term),
                aliases=[],
                summary=(
                    f"Auto-extracted from the '{briefing.topic}' briefing trajectory "
                    f"(appeared in {count} fields); needs human review before trusted use."
                ),
                evidence_refs=[f"briefing:{briefing.id}"],
                metadata={"auto_extracted": True, "occurrences": count, "source": "trajectory"},
            )
        )
    return entities[:50]


def extract_relations(
    entities: list[DomainKnowledgeEntity],
    briefing: DailyBriefing,
) -> list[DomainKnowledgeRelation]:
    """Create co-occurrence relations between entities found in the same source."""
    entity_by_name = {entity.name.lower(): entity for entity in entities}
    co_occurrences: Counter[tuple[str, str]] = Counter()

    # Co-occurrence within a single source (title + snippet) signals relatedness.
    for source in briefing.sources:
        window_text = f"{source.title} {source.snippet}".lower()
        present = [name for name in entity_by_name if name in window_text]
        for index, left in enumerate(present):
            for right in present[index + 1 :]:
                key = (left, right) if left < right else (right, left)
                co_occurrences[key] += 1

    now = datetime.now(UTC).isoformat()
    relations: list[DomainKnowledgeRelation] = []
    for (left_name, right_name), count in co_occurrences.most_common():
        left = entity_by_name[left_name]
        right = entity_by_name[right_name]
        relations.append(
            DomainKnowledgeRelation(
                id=f"auto_rel_{uuid.uuid4().hex[:12]}",
                source_entity_id=left.id,
                relation_type="co_occurs_with",
                target_entity_id=right.id,
                description=(
                    f"Co-occurred in {count} source(s) of the '{briefing.topic}' briefing; "
                    "relation is auto-extracted and needs human review."
                ),
                evidence_refs=[f"briefing:{briefing.id}"],
                weight=float(count),
            )
        )
    return relations[:50]


def _auto_graph_slug(value: str) -> str:
    """Normalize a topic or graph id into a safe auto-graph id suffix."""
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", value).strip("-")[:40]
    return slug or "misc"


def _effective_min_occurrences(briefing: DailyBriefing, min_occurrences: int) -> int:
    """Lower the occurrence threshold for short briefings.

    A daily briefing with fewer than 10 collected sources rarely repeats a term
    in 3 different fields, so extraction would come back empty.  Such briefings
    use a threshold of 2 instead of the configured default.
    """
    if len(briefing.sources) < 10 and min_occurrences >= 3:
        return 2
    return min_occurrences


def extend_graph_from_briefing(
    store: MemoryStore,
    briefing: DailyBriefing,
    graph_id: str | None = None,
    min_occurrences: int = _MIN_OCCURRENCES,
) -> dict[str, Any]:
    """Merge trajectory-extracted entities into a knowledge graph.

    A dedicated per-topic auto graph (``auto-<slug>``) is used so reviewed seed
    graphs stay untouched and different topics never pollute each other's auto
    graph.  The target id comes from ``graph_id`` when provided, otherwise from
    the briefing topic.  Repeated briefings on the same topic accumulate into
    the same auto graph; entities and co-occurrence relations are deduplicated
    by name / entity pair.  Auto entities carry ``metadata.auto_extracted=True``
    and are meant to be reviewed before being treated as trusted knowledge.
    """
    entities = extract_entities(briefing, min_occurrences=_effective_min_occurrences(briefing, min_occurrences))
    if not entities:
        return {"graph_id": None, "added_entities": 0, "added_relations": 0, "total_entities": 0}

    scope = str(graph_id).strip() if graph_id else briefing.topic
    target_id = f"auto-{_auto_graph_slug(scope)}"
    relations = extract_relations(entities, briefing)
    auto_graph = store.get_domain_knowledge_graph(target_id)
    if auto_graph is None:
        auto_graph = DomainKnowledgeGraph(
            id=target_id,
            name=f"自动提取图谱（{briefing.topic}）",
            description="Auto-extracted from briefing trajectories; review before trusted use.",
            overview=f"Entities and co-occurrence relations extracted from the '{briefing.topic}' briefing trajectory.",
            source="auto-extracted",
            version="1",
            entities=entities,
            relations=relations,
        )
        store.upsert_domain_knowledge_graph(auto_graph)
        return {
            "graph_id": auto_graph.id,
            "added_entities": len(entities),
            "added_relations": len(relations),
            "total_entities": len(entities),
            "min_occurrences_used": _effective_min_occurrences(briefing, min_occurrences),
        }

    # Merge into this topic's existing auto graph, deduplicating by name
    # (case-insensitive).
    existing_names = {entity.name.lower() for entity in auto_graph.entities}
    added_entities = [entity for entity in entities if entity.name.lower() not in existing_names]
    if added_entities:
        auto_graph.entities.extend(added_entities)

    # Relations referencing the same pair are deduplicated by (source,target).
    existing_pairs = {
        (relation.source_entity_id, relation.target_entity_id)
        for relation in auto_graph.relations
    }
    id_by_name = {entity.name.lower(): entity.id for entity in auto_graph.entities}
    added_relations: list[DomainKnowledgeRelation] = []
    for relation in relations:
        source_name = next(
            (name for name, entity_id in id_by_name.items() if entity_id == relation.source_entity_id),
            "",
        )
        target_name = next(
            (name for name, entity_id in id_by_name.items() if entity_id == relation.target_entity_id),
            "",
        )
        source_id = id_by_name.get(source_name)
        target_id = id_by_name.get(target_name)
        if source_id and target_id and (source_id, target_id) not in existing_pairs:
            relation.source_entity_id = source_id
            relation.target_entity_id = target_id
            added_relations.append(relation)
            existing_pairs.add((source_id, target_id))

    auto_graph.relations.extend(added_relations)
    # Capacity cap: an auto graph that keeps growing without bound pollutes
    # retrieval.  Trim to the strongest entities by occurrence count and drop
    # relations whose endpoints were trimmed away.
    if len(auto_graph.entities) > _MAX_AUTO_GRAPH_ENTITIES:
        auto_graph.entities.sort(
            key=lambda entity: int((entity.metadata or {}).get("occurrences", 0)),
            reverse=True,
        )
        auto_graph.entities = auto_graph.entities[:_MAX_AUTO_GRAPH_ENTITIES]
        kept_ids = {entity.id for entity in auto_graph.entities}
        auto_graph.relations = [
            relation
            for relation in auto_graph.relations
            if relation.source_entity_id in kept_ids and relation.target_entity_id in kept_ids
        ]
    store.upsert_domain_knowledge_graph(auto_graph)
    return {
        "graph_id": auto_graph.id,
        "added_entities": len(added_entities),
        "added_relations": len(added_relations),
        "total_entities": len(auto_graph.entities),
        "min_occurrences_used": _effective_min_occurrences(briefing, min_occurrences),
    }
