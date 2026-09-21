from __future__ import annotations

import re

from search_assistant.contracts import (
    DomainKnowledgeEntity,
    DomainKnowledgeGraph,
    DomainKnowledgeRelation,
    DomainKnowledgeSearchHit,
)
from search_assistant.knowledge_graph.bm25 import BM25Index, rrf_rank
from search_assistant.knowledge_graph.embedding import (
    EmbeddingProvider,
    build_embedding_provider,
    cosine_similarity,
    _embedding_cache_key,
)
from search_assistant.knowledge_graph.extractor import is_meaningful_entity_name
from search_assistant.knowledge_graph.seeds import default_domain_graphs
from search_assistant.memory.store import MemoryStore


_NEGATED_MENTION_PATTERN = re.compile(
    r"(?:"
    r"没有(?:讨论|涉及|提到|覆盖)|"
    r"未(?:讨论|涉及|提到|覆盖)|"
    r"不(?:讨论|涉及|覆盖)|"
    r"无(?:关|涉及|证据)|"
    r"no evidence for|without|does not discuss|do not discuss|not discuss"
    r")\s*([^。；;.!?\n]{0,80})",
    re.IGNORECASE,
)


class DomainKnowledgeGraphService:
    """Manage reviewed domain graphs used as a lightweight GraphRAG index."""

    def __init__(
        self,
        store: MemoryStore,
        embedding_provider: EmbeddingProvider | None = None,
        hybrid_retrieval: bool = True,
    ):
        self.store = store
        self.embedding_provider = embedding_provider or build_embedding_provider()
        self.hybrid_retrieval = hybrid_retrieval
        self._bm25_cache: dict[str, BM25Index] = {}

    def seed_default_graphs(self, domain_ids: list[str] | None = None) -> dict[str, object]:
        requested = {domain_id.strip() for domain_id in domain_ids or [] if domain_id.strip()}
        graphs = default_domain_graphs(requested or None)
        if requested:
            found = {graph.id for graph in graphs}
            missing = sorted(requested - found)
            if missing:
                raise ValueError(f"unknown default domain graph: {', '.join(missing)}")
        results = [self.store.upsert_domain_knowledge_graph(graph) for graph in graphs]
        return {
            "seeded_graphs": len(results),
            "graphs": results,
        }

    def list_graphs(self) -> list[dict[str, object]]:
        return self.store.list_domain_knowledge_graphs()

    def query(self, query: str, domain_id: str | None = None, limit: int = 10) -> list[DomainKnowledgeSearchHit]:
        normalized = " ".join(query.split()).strip()
        if not normalized:
            raise ValueError("query must not be empty")
        graphs = self._load_graphs(domain_id)
        query_embedding = self._query_embedding(normalized)
        hits: list[DomainKnowledgeSearchHit] = []
        for graph in graphs:
            entity_by_id = {entity.id: entity for entity in graph.entities}
            sibling_names = {entity.name for entity in graph.entities}
            bm25_scores = self._bm25_scores_for_graph(graph.id, normalized) if self.hybrid_retrieval else None
            bm25_max = max(bm25_scores) if bm25_scores else 0.0
            for index, entity in enumerate(graph.entities):
                # Skip entities that are too generic or are tokenizer fragments
                # (e.g. 2-char CJK substrings of a longer entity name).  This
                # keeps noisy auto-extracted vocabulary out of retrieval.
                if not is_meaningful_entity_name(entity.name, sibling_names):
                    continue
                score, matched_aliases = self._score_entity(normalized, entity)
                relation_bonus = self._relation_bonus(normalized, entity, graph.relations, entity_by_id)
                score += relation_bonus
                # Cross-lingual semantic boost: if the literal match is weak but the
                # query is semantically close to the entity, give it a small bonus.
                # Strong literal matches already dominate, so only boost weak ones.
                if score < 7.0 and query_embedding is not None:
                    semantic_similarity = self._semantic_similarity(query_embedding, entity)
                    if semantic_similarity >= 0.72:
                        score += semantic_similarity * 3.0
                # Sparse (BM25) bonus: helps when query terms are spread across the
                # entity text without forming a strong literal phrase.  The bonus is
                # capped so strong literal matches keep their dominance.
                if bm25_scores is not None and bm25_max > 0:
                    normalized_bm25 = bm25_scores[index] / bm25_max
                    score += min(2.0, normalized_bm25 * 2.0)
                if score <= 0:
                    continue
                hits.append(
                    DomainKnowledgeSearchHit(
                        graph_id=graph.id,
                        graph_name=graph.name,
                        entity_id=entity.id,
                        entity_name=entity.name,
                        entity_type=entity.entity_type,
                        score=round(score, 3),
                        summary=entity.summary,
                        matched_aliases=matched_aliases,
                        outgoing_relations=self._format_relations(
                            [relation for relation in graph.relations if relation.source_entity_id == entity.id],
                            entity_by_id,
                        ),
                        incoming_relations=self._format_relations(
                            [relation for relation in graph.relations if relation.target_entity_id == entity.id],
                            entity_by_id,
                        ),
                    )
                )
        hits.sort(key=lambda hit: (-hit.score, hit.graph_id, hit.entity_name))
        return hits[: max(1, limit)]

    def query_relevant(
        self,
        query: str,
        domain_id: str | None = None,
        limit: int = 10,
        min_score: float = 5.0,
    ) -> list[DomainKnowledgeSearchHit]:
        """Return graph hits strong enough to be used as planning context."""

        candidates = self.query(query, domain_id=domain_id, limit=max(limit * 4, 20))
        hits = [hit for hit in candidates if hit.score >= min_score]
        return hits[: max(1, limit)]

    def export_markdown(self, domain_id: str) -> str:
        graph = self.store.get_domain_knowledge_graph(domain_id)
        if graph is None:
            raise KeyError(f"domain knowledge graph not found: {domain_id}")
        entity_by_id = {entity.id: entity for entity in graph.entities}
        lines = [
            f"# {graph.name} 知识图谱",
            "",
            f"- 图谱 ID：`{graph.id}`",
            f"- 来源：`{graph.source}`",
            f"- 版本：`{graph.version}`",
            "",
            "## 概览",
            "",
            graph.overview,
            "",
            "## 实体",
            "",
        ]
        for entity in graph.entities:
            aliases = "、".join(entity.aliases) if entity.aliases else "无"
            lines.extend(
                [
                    f"### {entity.name}",
                    "",
                    f"- ID：`{entity.id}`",
                    f"- 类型：`{entity.entity_type}`",
                    f"- 别名：{aliases}",
                    f"- 证据引用：{', '.join(f'`{ref}`' for ref in entity.evidence_refs) or '无'}",
                    "",
                    entity.summary,
                    "",
                ]
            )
        lines.extend(["## 关系", ""])
        for relation in graph.relations:
            source = entity_by_id.get(relation.source_entity_id)
            target = entity_by_id.get(relation.target_entity_id)
            source_name = source.name if source else relation.source_entity_id
            target_name = target.name if target else relation.target_entity_id
            lines.extend(
                [
                    f"- `{relation.id}`：[{source_name}](#{_anchor(source_name)}) "
                    f"-- **{relation.relation_type}** --> "
                    f"[{target_name}](#{_anchor(target_name)})",
                    f"  - {relation.description}",
                ]
            )
        lines.extend(
            [
                "",
                "## 使用说明",
                "",
                "该图谱是经过审阅的领域语义骨架，适合作为日报检索、证据归类和后续增量知识沉淀的起点。",
                "新增知识时应保留原始自然语言证据，同时补充或更新实体-关系边，避免把孤立文本块直接平铺进知识库。",
                "",
            ]
        )
        return "\n".join(lines)

    def _load_graphs(self, domain_id: str | None) -> list[DomainKnowledgeGraph]:
        if domain_id is not None:
            graph = self.store.get_domain_knowledge_graph(domain_id)
            if graph is None:
                return []
            return [graph]
        return [
            graph
            for row in self.store.list_domain_knowledge_graphs()
            if (graph := self.store.get_domain_knowledge_graph(str(row["id"]))) is not None
        ]

    def _bm25_scores_for_graph(self, graph_id: str, query: str) -> list[float]:
        """Return BM25 scores of every entity in a graph for a query.

        The index is built lazily per graph id and cached in memory so repeated
        queries do not re-tokenize the entity corpus every time.
        """
        index = self._bm25_cache.get(graph_id)
        if index is None:
            graph = self.store.get_domain_knowledge_graph(graph_id)
            if graph is None:
                return []
            documents = [
                " ".join([entity.name, *entity.aliases, entity.summary]) for entity in graph.entities
            ]
            index = BM25Index(documents)
            self._bm25_cache[graph_id] = index
        return index.score_all(query)

    @staticmethod
    def _score_entity(query: str, entity: DomainKnowledgeEntity) -> tuple[float, list[str]]:
        query_lower = query.lower()
        positive_query_lower = _positive_query_text(query_lower)
        terms = _terms(query_lower)
        name_lower = entity.name.lower()
        alias_lowers = [alias.lower() for alias in entity.aliases]
        summary_lower = entity.summary.lower()
        metadata_lower = " ".join(str(value).lower() for value in entity.metadata.values())

        score = 0.0
        matched_aliases: list[str] = []
        if positive_query_lower and positive_query_lower == name_lower:
            score += 10.0
        elif positive_query_lower and (positive_query_lower in name_lower or name_lower in positive_query_lower):
            score += 7.0
        for alias, alias_lower in zip(entity.aliases, alias_lowers, strict=False):
            if positive_query_lower and positive_query_lower == alias_lower:
                score += 8.0
                matched_aliases.append(alias)
            elif positive_query_lower and (positive_query_lower in alias_lower or alias_lower in positive_query_lower):
                score += 5.0
                matched_aliases.append(alias)
        for term in terms:
            if term in name_lower:
                score += 3.0
            if any(term in alias for alias in alias_lowers):
                score += 2.0
            if term in summary_lower:
                score += 1.0
            if term in metadata_lower:
                score += 0.5
        return score, matched_aliases

    @staticmethod
    def _relation_bonus(
        query: str,
        entity: DomainKnowledgeEntity,
        relations: list[DomainKnowledgeRelation],
        entity_by_id: dict[str, DomainKnowledgeEntity],
    ) -> float:
        query_lower = _positive_query_text(query.lower())
        terms = _terms(query_lower)
        bonus = 0.0
        for relation in relations:
            if relation.source_entity_id != entity.id and relation.target_entity_id != entity.id:
                continue
            other_id = relation.target_entity_id if relation.source_entity_id == entity.id else relation.source_entity_id
            other = entity_by_id.get(other_id)
            haystack = " ".join(
                (
                    relation.relation_type,
                    relation.description,
                    other.name if other else other_id,
                    " ".join(other.aliases) if other else "",
                )
            ).lower()
            if query_lower in haystack:
                bonus += 2.0
            bonus += sum(0.4 for term in terms if term in haystack)
        return bonus

    @staticmethod
    def _format_relations(
        relations: list[DomainKnowledgeRelation],
        entity_by_id: dict[str, DomainKnowledgeEntity],
    ) -> list[str]:
        formatted: list[str] = []
        for relation in relations:
            source = entity_by_id.get(relation.source_entity_id)
            target = entity_by_id.get(relation.target_entity_id)
            source_name = source.name if source else relation.source_entity_id
            target_name = target.name if target else relation.target_entity_id
            formatted.append(f"{source_name} --{relation.relation_type}--> {target_name}: {relation.description}")
        return formatted

    def _query_embedding(self, query: str) -> list[float] | None:
        """Return the query embedding, using the SQLite cache when available.

        Returns ``None`` when no provider is configured or the provider fails,
        allowing the caller to fall back to literal matching only.
        """
        provider = self.embedding_provider
        if provider.__class__.__name__ == "NullEmbeddingProvider":
            return None
        cache_key = _embedding_cache_key(query)
        cached = self.store.get_entity_embedding(cache_key)
        if cached:
            return cached
        try:
            vectors = provider.embed([query])
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

    def _semantic_similarity(self, query_embedding: list[float], entity: DomainKnowledgeEntity) -> float:
        """Compute cosine similarity between the query and an entity text representation."""
        provider = self.embedding_provider
        if provider.__class__.__name__ == "NullEmbeddingProvider":
            return 0.0
        entity_text = " ".join([entity.name, *entity.aliases, entity.summary]).strip()
        if not entity_text:
            return 0.0
        cache_key = _embedding_cache_key(entity_text)
        cached = self.store.get_entity_embedding(cache_key)
        if cached:
            entity_embedding = cached
        else:
            try:
                vectors = provider.embed([entity_text])
            except Exception:
                return 0.0
            if not vectors or not vectors[0]:
                return 0.0
            entity_embedding = vectors[0]
            self.store.upsert_entity_embedding(
                cache_key,
                provider.__class__.__name__,
                getattr(provider, "model", "unknown"),
                entity_embedding,
            )
        return cosine_similarity(query_embedding, entity_embedding)


def _terms(value: str) -> list[str]:
    negated_terms: set[str] = set()
    for match in _NEGATED_MENTION_PATTERN.finditer(value):
        negated_terms.update(_extract_terms(match.group(1).lower()))
    terms: list[str] = []
    for chunk in re.split(r"[\s,，。；;:：/|()（）]+", value):
        normalized = chunk.strip()
        if not normalized:
            continue
        terms.extend(_extract_terms(normalized))
    return list(dict.fromkeys(term for term in terms if term and term not in negated_terms))


def _positive_query_text(value: str) -> str:
    return _NEGATED_MENTION_PATTERN.sub(" ", value).strip()


def _extract_terms(value: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+._-]{1,}|[\u4e00-\u9fff]{2,}", value)


def _anchor(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^\w\u4e00-\u9fff -]+", "", text)
    return re.sub(r"\s+", "-", text)
