"""
컨텍스트 직렬화 모듈
Context Serializer for GraphRAG System

서브그래프를 LLM이 이해할 수 있는 구조화된 텍스트로 변환
- 쿼리 타입별 최적화된 텍스트 구조
- 토큰 제한 고려한 스마트 압축
- 관련성 기반 우선순위 정렬
- 한국어/영어 혼용 지원
"""

import re
import json
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np

try:
    from .subgraph_extractor import SubgraphResult
    from ..query_analyzer import QueryAnalysisResult, QueryType, QueryComplexity
except ImportError as e:
    import warnings

    warnings.warn(f"Some GraphRAG components not available: {e}")

# 로깅 설정
logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    """직렬화 형태"""

    STRUCTURED = "structured"  # 구조화된 섹션별
    NARRATIVE = "narrative"  # 자연어 서술형
    LIST_BASED = "list_based"  # 리스트 기반
    HIERARCHICAL = "hierarchical"  # 계층적 구조
    COMPACT = "compact"  # 압축형


class ContextPriority(Enum):
    """컨텍스트 우선순위"""

    CRITICAL = "critical"  # 필수 정보
    HIGH = "high"  # 높은 관련성
    MEDIUM = "medium"  # 중간 관련성
    LOW = "low"  # 낮은 관련성
    SUPPLEMENTARY = "supplementary"  # 보조 정보


@dataclass
class SerializationConfig:
    """직렬화 설정"""

    # 출력 형태
    format_style: SerializationFormat = SerializationFormat.STRUCTURED
    language: str = "mixed"  # "ko", "en", "mixed"

    # 토큰 제한
    max_tokens: int = 8000
    max_nodes_detail: int = 50  # 상세 정보를 포함할 최대 노드 수
    max_edges_detail: int = 100  # 상세 정보를 포함할 최대 엣지 수

    # 내용 제어
    include_metadata: bool = True
    include_statistics: bool = True
    include_relationships: bool = True
    include_node_details: bool = True

    # 우선순위 제어
    priority_threshold: float = 0.5  # 포함할 최소 관련성 점수
    summarize_low_priority: bool = True

    # 언어별 설정
    use_english_terms: bool = True  # 전문용어는 영어 사용
    add_translations: bool = False  # 번역 추가

    # 압축 설정
    compress_similar_nodes: bool = True
    max_items_per_section: int = 20


@dataclass
class SerializedContext:
    """직렬화된 컨텍스트 결과"""

    # 메인 텍스트
    main_text: str

    # 섹션별 내용
    sections: Dict[str, str]

    # 메타데이터
    query: str
    total_tokens: int
    included_nodes: int
    included_edges: int
    compression_ratio: float

    # 우선순위별 통계
    priority_distribution: Dict[str, int]

    # 언어 정보
    language: str
    mixed_language_detected: bool

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "main_text": self.main_text,
            "sections": self.sections,
            "query": self.query,
            "total_tokens": self.total_tokens,
            "included_nodes": self.included_nodes,
            "included_edges": self.included_edges,
            "compression_ratio": self.compression_ratio,
            "priority_distribution": self.priority_distribution,
            "language": self.language,
            "mixed_language_detected": self.mixed_language_detected,
        }


class ContextSerializer:
    """컨텍스트 직렬화기"""

    def __init__(self, config: Optional[SerializationConfig] = None):
        """
        Args:
            config: 직렬화 설정
        """
        self.config = config or SerializationConfig()

        # 언어별 템플릿
        self._load_language_templates()

        # 토큰 추정기 (대략적)
        self._avg_chars_per_token = 4  # 한국어+영어 혼용 기준

        logger.info("✅ ContextSerializer initialized")
        logger.info(f"   📏 Max tokens: {self.config.max_tokens}")
        logger.info(f"   🌐 Language: {self.config.language}")
        logger.info(f"   📋 Format: {self.config.format_style.value}")

    def _load_language_templates(self) -> None:
        """언어별 템플릿 로드"""
        self.templates = {
            "ko": {
                "section_headers": {
                    "overview": "## 📊 개요",
                    "papers": "## 📄 관련 논문들",
                    "authors": "## 👥 연구자들",
                    "keywords": "## 🔤 주요 키워드",
                    "journals": "## 📰 저널/학회",
                    "relationships": "## 🔗 관계 정보",
                    "statistics": "## 📈 통계 정보",
                },
                "connection_words": {
                    "cited_by": "에서 인용됨",
                    "cites": "을/를 인용함",
                    "collaborated_with": "과/와 협업함",
                    "authored_by": "의 저자",
                    "published_in": "에 발표됨",
                    "has_keyword": "키워드 포함",
                    "similar_to": "와/과 유사함",
                },
                "summary_phrases": {
                    "total_papers": "총 {} 편의 논문",
                    "total_authors": "총 {} 명의 연구자",
                    "year_range": "{}년 ~ {}년 기간",
                    "main_topics": "주요 주제: {}",
                    "high_relevance": "높은 관련성",
                    "medium_relevance": "중간 관련성",
                    "additional_info": "추가 정보",
                },
            },
            "en": {
                "section_headers": {
                    "overview": "## 📊 Overview",
                    "papers": "## 📄 Related Papers",
                    "authors": "## 👥 Researchers",
                    "keywords": "## 🔤 Key Terms",
                    "journals": "## 📰 Journals/Venues",
                    "relationships": "## 🔗 Relationships",
                    "statistics": "## 📈 Statistics",
                },
                "connection_words": {
                    "cited_by": "cited by",
                    "cites": "cites",
                    "collaborated_with": "collaborated with",
                    "authored_by": "authored by",
                    "published_in": "published in",
                    "has_keyword": "includes keyword",
                    "similar_to": "similar to",
                },
                "summary_phrases": {
                    "total_papers": "{} papers total",
                    "total_authors": "{} researchers total",
                    "year_range": "{} - {} period",
                    "main_topics": "Main topics: {}",
                    "high_relevance": "High relevance",
                    "medium_relevance": "Medium relevance",
                    "additional_info": "Additional information",
                },
            },
        }

    def serialize(
        self,
        subgraph_result: SubgraphResult,
        query_analysis: Optional[QueryAnalysisResult] = None,
        custom_config: Optional[SerializationConfig] = None,
    ) -> SerializedContext:
        """메인 직렬화 함수"""

        config = custom_config or self.config

        logger.info(
            f"📝 Serializing subgraph with {subgraph_result.total_nodes} nodes..."
        )

        # 언어 감지
        detected_language = self._detect_content_language(
            subgraph_result, query_analysis
        )

        # 노드/엣지 우선순위 계산
        node_priorities = self._calculate_node_priorities(
            subgraph_result, query_analysis, config
        )
        edge_priorities = self._calculate_edge_priorities(
            subgraph_result, query_analysis, config
        )

        # 섹션별 내용 생성
        sections = self._generate_sections(
            subgraph_result,
            query_analysis,
            node_priorities,
            edge_priorities,
            config,
            detected_language,
        )

        # 메인 텍스트 조합
        main_text = self._assemble_main_text(sections, config, detected_language)

        # 토큰 수 추정 및 압축 (필요시)
        estimated_tokens = self._estimate_tokens(main_text)
        if estimated_tokens > config.max_tokens:
            main_text, sections = self._compress_content(
                main_text, sections, config, target_tokens=config.max_tokens
            )
            estimated_tokens = self._estimate_tokens(main_text)

        # 통계 계산
        included_nodes = len(
            [
                nid
                for nid, priority in node_priorities.items()
                if priority != ContextPriority.SUPPLEMENTARY
            ]
        )
        included_edges = len(
            [
                eid
                for eid, priority in edge_priorities.items()
                if priority != ContextPriority.SUPPLEMENTARY
            ]
        )

        compression_ratio = min(1.0, estimated_tokens / max(1, config.max_tokens))

        priority_dist = Counter([p.value for p in node_priorities.values()])

        # 결과 생성
        result = SerializedContext(
            main_text=main_text,
            sections=sections,
            query=subgraph_result.query,
            total_tokens=estimated_tokens,
            included_nodes=included_nodes,
            included_edges=included_edges,
            compression_ratio=compression_ratio,
            priority_distribution=dict(priority_dist),
            language=detected_language,
            mixed_language_detected=self._is_mixed_language(main_text),
        )

        logger.info(f"✅ Serialization completed:")
        logger.info(f"   📏 Tokens: {estimated_tokens:,}")
        logger.info(f"   📄 Nodes: {included_nodes}/{subgraph_result.total_nodes}")
        logger.info(f"   🔗 Edges: {included_edges}/{subgraph_result.total_edges}")
        logger.info(f"   🗜️ Compression: {compression_ratio:.1%}")

        return result

    def _detect_content_language(
        self,
        subgraph_result: SubgraphResult,
        query_analysis: Optional[QueryAnalysisResult],
    ) -> str:
        """컨텐츠 언어 감지"""

        # 쿼리 분석 결과가 있으면 우선 사용
        if query_analysis and query_analysis.language:
            return query_analysis.language

        # 노드 텍스트에서 언어 감지
        korean_chars = 0
        english_chars = 0

        # 샘플 노드들의 텍스트 분석
        sample_texts = []
        for node_id, node_data in list(subgraph_result.nodes.items())[:20]:
            title = node_data.get("title", "")
            if title:
                sample_texts.append(title)

        for text in sample_texts:
            korean_chars += len(re.findall(r"[가-힣]", text))
            english_chars += len(re.findall(r"[a-zA-Z]", text))

        total_chars = korean_chars + english_chars
        if total_chars == 0:
            return self.config.language

        korean_ratio = korean_chars / total_chars

        if korean_ratio > 0.6:
            return "ko"
        elif korean_ratio < 0.3:
            return "en"
        else:
            return "mixed"

    def _calculate_node_priorities(
        self,
        subgraph_result: SubgraphResult,
        query_analysis: Optional[QueryAnalysisResult],
        config: SerializationConfig,
    ) -> Dict[str, ContextPriority]:
        """노드별 우선순위 계산"""

        priorities = {}

        for node_id, node_data in subgraph_result.nodes.items():
            # 기본 관련성 점수
            relevance_score = subgraph_result.relevance_scores.get(node_id, 0.0)

            # 초기 검색 매치 보너스
            is_initial_match = any(
                result.node_id == node_id for result in subgraph_result.initial_matches
            )
            if is_initial_match:
                relevance_score += 0.3

            # 노드 타입별 중요도
            node_type = node_data.get("node_type", "unknown")
            type_bonus = {
                "paper": 0.2,
                "author": 0.15,
                "keyword": 0.1,
                "journal": 0.05,
            }.get(node_type, 0.0)
            relevance_score += type_bonus

            # 쿼리 분석 기반 보너스
            if query_analysis:
                if node_type in [nt.value for nt in query_analysis.required_node_types]:
                    relevance_score += 0.15

            # 우선순위 결정
            if relevance_score >= 0.8:
                priorities[node_id] = ContextPriority.CRITICAL
            elif relevance_score >= 0.6:
                priorities[node_id] = ContextPriority.HIGH
            elif relevance_score >= 0.4:
                priorities[node_id] = ContextPriority.MEDIUM
            elif relevance_score >= config.priority_threshold:
                priorities[node_id] = ContextPriority.LOW
            else:
                priorities[node_id] = ContextPriority.SUPPLEMENTARY

        return priorities

    def _calculate_edge_priorities(
        self,
        subgraph_result: SubgraphResult,
        query_analysis: Optional[QueryAnalysisResult],
        config: SerializationConfig,
    ) -> Dict[str, ContextPriority]:
        """엣지별 우선순위 계산"""

        priorities = {}

        for i, edge in enumerate(subgraph_result.edges):
            edge_id = f"edge_{i}"
            source = edge["source"]
            target = edge["target"]
            edge_type = edge.get("edge_type", "unknown")

            # 소스/타겟 노드의 관련성
            source_relevance = subgraph_result.relevance_scores.get(source, 0.0)
            target_relevance = subgraph_result.relevance_scores.get(target, 0.0)
            avg_relevance = (source_relevance + target_relevance) / 2

            # 엣지 타입별 중요도
            edge_importance = {
                "cites": 0.2,
                "authored_by": 0.15,
                "published_in": 0.1,
                "collaborates_with": 0.15,
                "has_keyword": 0.05,
                "similar_to": 0.1,
            }.get(edge_type, 0.05)

            # 종합 점수
            edge_score = avg_relevance + edge_importance

            # 우선순위 결정
            if edge_score >= 0.7:
                priorities[edge_id] = ContextPriority.CRITICAL
            elif edge_score >= 0.5:
                priorities[edge_id] = ContextPriority.HIGH
            elif edge_score >= 0.3:
                priorities[edge_id] = ContextPriority.MEDIUM
            else:
                priorities[edge_id] = ContextPriority.LOW

        return priorities

    def _generate_sections(
        self,
        subgraph_result: SubgraphResult,
        query_analysis: Optional[QueryAnalysisResult],
        node_priorities: Dict[str, ContextPriority],
        edge_priorities: Dict[str, ContextPriority],
        config: SerializationConfig,
        language: str,
    ) -> Dict[str, str]:
        """섹션별 내용 생성"""

        sections = {}
        templates = self.templates.get(language, self.templates["en"])

        # 1. 개요 섹션
        sections["overview"] = self._generate_overview_section(
            subgraph_result, query_analysis, templates, config
        )

        # 2. 노드 타입별 섹션
        if config.include_node_details:
            # 논문 섹션
            paper_nodes = {
                nid: data
                for nid, data in subgraph_result.nodes.items()
                if data.get("node_type") == "paper"
                and node_priorities.get(nid, ContextPriority.SUPPLEMENTARY)
                != ContextPriority.SUPPLEMENTARY
            }

            if paper_nodes:
                sections["papers"] = self._generate_papers_section(
                    paper_nodes, node_priorities, templates, config
                )

            # 저자 섹션
            author_nodes = {
                nid: data
                for nid, data in subgraph_result.nodes.items()
                if data.get("node_type") == "author"
                and node_priorities.get(nid, ContextPriority.SUPPLEMENTARY)
                != ContextPriority.SUPPLEMENTARY
            }

            if author_nodes:
                sections["authors"] = self._generate_authors_section(
                    author_nodes, node_priorities, templates, config
                )

            # 키워드 섹션
            keyword_nodes = {
                nid: data
                for nid, data in subgraph_result.nodes.items()
                if data.get("node_type") == "keyword"
                and node_priorities.get(nid, ContextPriority.SUPPLEMENTARY)
                != ContextPriority.SUPPLEMENTARY
            }

            if keyword_nodes:
                sections["keywords"] = self._generate_keywords_section(
                    keyword_nodes, node_priorities, templates, config
                )

        # 3. 관계 섹션
        if config.include_relationships:
            high_priority_edges = [
                edge
                for i, edge in enumerate(subgraph_result.edges)
                if edge_priorities.get(f"edge_{i}", ContextPriority.LOW)
                in [ContextPriority.CRITICAL, ContextPriority.HIGH]
            ]

            if high_priority_edges:
                sections["relationships"] = self._generate_relationships_section(
                    high_priority_edges, subgraph_result.nodes, templates, config
                )

        # 4. 통계 섹션
        if config.include_statistics:
            sections["statistics"] = self._generate_statistics_section(
                subgraph_result, templates, config
            )

        return sections

    def _generate_overview_section(
        self,
        subgraph_result: SubgraphResult,
        query_analysis: Optional[QueryAnalysisResult],
        templates: Dict[str, Any],
        config: SerializationConfig,
    ) -> str:
        """개요 섹션 생성"""

        lines = [templates["section_headers"]["overview"]]

        # 쿼리 정보
        lines.append(f"**Query:** {subgraph_result.query}")

        # 기본 통계
        summary = templates["summary_phrases"]

        paper_count = len(
            [n for n in subgraph_result.nodes.values() if n.get("node_type") == "paper"]
        )
        author_count = len(
            [
                n
                for n in subgraph_result.nodes.values()
                if n.get("node_type") == "author"
            ]
        )

        if paper_count > 0:
            lines.append(f"- {summary['total_papers'].format(paper_count)}")
        if author_count > 0:
            lines.append(f"- {summary['total_authors'].format(author_count)}")

        # 년도 범위
        years = []
        for node_data in subgraph_result.nodes.values():
            year = node_data.get("year", "")
            if year and str(year).isdigit():
                years.append(int(year))

        if years:
            min_year, max_year = min(years), max(years)
            if min_year != max_year:
                lines.append(f"- {summary['year_range'].format(min_year, max_year)}")

        # 주요 키워드 (상위 5개)
        keyword_freq = Counter()
        for node_data in subgraph_result.nodes.values():
            if node_data.get("node_type") == "keyword":
                keyword_freq[node_data.get("id", "")] += 1

        if keyword_freq:
            top_keywords = [kw for kw, _ in keyword_freq.most_common(5)]
            lines.append(f"- {summary['main_topics'].format(', '.join(top_keywords))}")

        # 신뢰도 정보
        confidence = subgraph_result.confidence_score
        confidence_text = (
            "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
        )
        lines.append(f"- **Confidence:** {confidence_text} ({confidence:.2f})")

        return "\n".join(lines) + "\n"

    def _generate_papers_section(
        self,
        paper_nodes: Dict[str, Dict[str, Any]],
        node_priorities: Dict[str, ContextPriority],
        templates: Dict[str, Any],
        config: SerializationConfig,
    ) -> str:
        """논문 섹션 생성"""

        lines = [templates["section_headers"]["papers"]]

        # 우선순위별로 정렬
        sorted_papers = sorted(
            paper_nodes.items(),
            key=lambda x: (
                node_priorities.get(x[0], ContextPriority.LOW).value,
                x[1].get("year", ""),
                x[1].get("title", ""),
            ),
        )

        # 제한된 수만 상세 표시
        detailed_count = min(len(sorted_papers), config.max_nodes_detail)

        for i, (paper_id, paper_data) in enumerate(sorted_papers):
            title = paper_data.get("title", "Unknown Title")
            year = paper_data.get("year", "")
            authors = paper_data.get("authors", [])
            journal = paper_data.get("journal", "")

            priority = node_priorities.get(paper_id, ContextPriority.LOW)

            if i < detailed_count:
                # 상세 정보
                lines.append(f"\n### 📄 {title}")
                if year:
                    lines.append(f"- **Year:** {year}")
                if authors:
                    author_list = authors if isinstance(authors, list) else [authors]
                    author_text = ", ".join(author_list[:3])  # 최대 3명
                    if len(author_list) > 3:
                        author_text += f" (+{len(author_list)-3} others)"
                    lines.append(f"- **Authors:** {author_text}")
                if journal:
                    lines.append(f"- **Journal:** {journal}")
                lines.append(f"- **Relevance:** {priority.value}")
            else:
                # 간단 표시
                summary = f"- {title}"
                if year:
                    summary += f" ({year})"
                lines.append(summary)

        # 요약 정보 (많은 경우)
        if len(sorted_papers) > detailed_count:
            remaining = len(sorted_papers) - detailed_count
            lines.append(f"\n*... and {remaining} more papers*")

        return "\n".join(lines) + "\n"

    def _generate_authors_section(
        self,
        author_nodes: Dict[str, Dict[str, Any]],
        node_priorities: Dict[str, ContextPriority],
        templates: Dict[str, Any],
        config: SerializationConfig,
    ) -> str:
        """저자 섹션 생성"""

        lines = [templates["section_headers"]["authors"]]

        # 우선순위별로 정렬
        sorted_authors = sorted(
            author_nodes.items(),
            key=lambda x: (
                node_priorities.get(x[0], ContextPriority.LOW).value,
                x[1].get("paper_count", 0),
            ),
            reverse=True,
        )

        for i, (author_id, author_data) in enumerate(
            sorted_authors[: config.max_nodes_detail]
        ):
            name = author_data.get("name", author_id)
            paper_count = author_data.get("paper_count", 0)
            productivity_type = author_data.get("productivity_type", "")

            lines.append(f"\n### 👤 {name}")
            if paper_count:
                lines.append(f"- **Papers:** {paper_count}")
            if productivity_type:
                lines.append(f"- **Type:** {productivity_type}")

            # 주요 키워드 (있다면)
            top_keywords = author_data.get("top_keywords", [])
            if top_keywords:
                if isinstance(top_keywords[0], (list, tuple)):
                    keyword_names = [kw[0] for kw in top_keywords[:3]]
                else:
                    keyword_names = top_keywords[:3]
                lines.append(f"- **Research Areas:** {', '.join(keyword_names)}")

        return "\n".join(lines) + "\n"

    def _generate_keywords_section(
        self,
        keyword_nodes: Dict[str, Dict[str, Any]],
        node_priorities: Dict[str, ContextPriority],
        templates: Dict[str, Any],
        config: SerializationConfig,
    ) -> str:
        """키워드 섹션 생성"""

        lines = [templates["section_headers"]["keywords"]]

        # 빈도별로 정렬
        sorted_keywords = sorted(
            keyword_nodes.items(),
            key=lambda x: (
                node_priorities.get(x[0], ContextPriority.LOW).value,
                x[1].get("frequency", 0),
            ),
            reverse=True,
        )

        # 그룹별로 표시
        critical_keywords = []
        high_keywords = []
        other_keywords = []

        for keyword_id, keyword_data in sorted_keywords:
            keyword = keyword_data.get("name", keyword_id)
            frequency = keyword_data.get("frequency", 0)
            priority = node_priorities.get(keyword_id, ContextPriority.LOW)

            keyword_info = f"{keyword}"
            if frequency:
                keyword_info += f" ({frequency})"

            if priority == ContextPriority.CRITICAL:
                critical_keywords.append(keyword_info)
            elif priority == ContextPriority.HIGH:
                high_keywords.append(keyword_info)
            else:
                other_keywords.append(keyword_info)

        # 우선순위별 출력
        if critical_keywords:
            lines.append(f"\n**{templates['summary_phrases']['high_relevance']}:**")
            lines.append(f"{', '.join(critical_keywords[:10])}")

        if high_keywords:
            lines.append(f"\n**{templates['summary_phrases']['medium_relevance']}:**")
            lines.append(f"{', '.join(high_keywords[:15])}")

        if other_keywords and not config.summarize_low_priority:
            lines.append(f"\n**{templates['summary_phrases']['additional_info']}:**")
            lines.append(f"{', '.join(other_keywords[:10])}")

        return "\n".join(lines) + "\n"

    def _generate_relationships_section(
        self,
        edges: List[Dict[str, Any]],
        nodes: Dict[str, Dict[str, Any]],
        templates: Dict[str, Any],
        config: SerializationConfig,
    ) -> str:
        """관계 섹션 생성"""

        lines = [templates["section_headers"]["relationships"]]

        # 엣지 타입별 그룹화
        edge_groups = defaultdict(list)
        for edge in edges:
            edge_type = edge.get("edge_type", "related_to")
            edge_groups[edge_type].append(edge)

        connection_words = templates["connection_words"]

        for edge_type, type_edges in edge_groups.items():
            if len(type_edges) == 0:
                continue

            connection_word = connection_words.get(edge_type, edge_type)
            lines.append(f"\n**{connection_word.title()} ({len(type_edges)}):**")

            # 샘플 관계들 표시
            sample_size = min(5, len(type_edges))
            for edge in type_edges[:sample_size]:
                source_id = edge["source"]
                target_id = edge["target"]

                source_data = nodes.get(source_id, {})
                target_data = nodes.get(target_id, {})

                source_name = self._get_node_display_name(source_data)
                target_name = self._get_node_display_name(target_data)

                lines.append(f"- {source_name} → {target_name}")

            # 더 있으면 요약
            if len(type_edges) > sample_size:
                lines.append(f"- *... and {len(type_edges) - sample_size} more*")

        return "\n".join(lines) + "\n"

    def _generate_statistics_section(
        self,
        subgraph_result: SubgraphResult,
        templates: Dict[str, Any],
        config: SerializationConfig,
    ) -> str:
        """통계 섹션 생성"""

        lines = [templates["section_headers"]["statistics"]]

        # 기본 통계
        lines.append(f"- **Total Nodes:** {subgraph_result.total_nodes}")
        lines.append(f"- **Total Edges:** {subgraph_result.total_edges}")
        lines.append(
            f"- **Extraction Strategy:** {subgraph_result.extraction_strategy.value}"
        )
        lines.append(f"- **Processing Time:** {subgraph_result.extraction_time:.2f}s")

        # 노드 타입별 분포
        if subgraph_result.nodes_by_type:
            lines.append(f"\n**Node Distribution:**")
            for node_type, count in subgraph_result.nodes_by_type.items():
                lines.append(f"- {node_type}: {count}")

        # 상위 관련 노드들
        top_relevant = sorted(
            subgraph_result.relevance_scores.items(), key=lambda x: x[1], reverse=True
        )[:5]

        if top_relevant:
            lines.append(f"\n**Most Relevant Nodes:**")
            for node_id, score in top_relevant:
                node_data = subgraph_result.nodes.get(node_id, {})
                name = self._get_node_display_name(node_data)
                lines.append(f"- {name}: {score:.3f}")

        return "\n".join(lines) + "\n"

    def _get_node_display_name(self, node_data: Dict[str, Any]) -> str:
        """노드 표시명 생성"""
        node_type = node_data.get("node_type", "unknown")

        if node_type == "paper":
            title = node_data.get("title", "Unknown Paper")
            return title[:50] + "..." if len(title) > 50 else title
        elif node_type == "author":
            return node_data.get("name", node_data.get("id", "Unknown Author"))
        elif node_type == "keyword":
            return node_data.get("name", node_data.get("id", "Unknown Keyword"))
        elif node_type == "journal":
            return node_data.get("name", node_data.get("id", "Unknown Journal"))
        else:
            return node_data.get("name", node_data.get("id", "Unknown"))

    def _assemble_main_text(
        self, sections: Dict[str, str], config: SerializationConfig, language: str
    ) -> str:
        """섹션들을 메인 텍스트로 조합"""

        if config.format_style == SerializationFormat.STRUCTURED:
            # 구조화된 형태
            section_order = [
                "overview",
                "papers",
                "authors",
                "keywords",
                "relationships",
                "statistics",
            ]
            text_parts = []

            for section_name in section_order:
                if section_name in sections and sections[section_name].strip():
                    text_parts.append(sections[section_name])

            return "\n".join(text_parts)

        elif config.format_style == SerializationFormat.NARRATIVE:
            # 자연어 서술형
            return self._create_narrative_text(sections, language)

        elif config.format_style == SerializationFormat.COMPACT:
            # 압축형
            return self._create_compact_text(sections, config)

        else:
            # 기본값: STRUCTURED
            return "\n".join(sections.values())

    def _create_narrative_text(self, sections: Dict[str, str], language: str) -> str:
        """자연어 서술형 텍스트 생성"""
        # 구현 간소화 - 구조화된 형태로 대체
        return "\n".join(sections.values())

    def _create_compact_text(
        self, sections: Dict[str, str], config: SerializationConfig
    ) -> str:
        """압축형 텍스트 생성"""
        # 헤더 제거 및 압축
        compact_lines = []

        for section_text in sections.values():
            lines = section_text.split("\n")
            # 헤더와 빈 줄 제거
            content_lines = [
                line for line in lines if line.strip() and not line.startswith("#")
            ]
            compact_lines.extend(content_lines[:5])  # 각 섹션에서 최대 5줄만

        return "\n".join(compact_lines)

    def _estimate_tokens(self, text: str) -> int:
        """토큰 수 추정 (대략적)"""
        # 한국어/영어 혼용 텍스트에 대한 대략적 추정
        char_count = len(text)
        return int(char_count / self._avg_chars_per_token)

    def _compress_content(
        self,
        main_text: str,
        sections: Dict[str, str],
        config: SerializationConfig,
        target_tokens: int,
    ) -> Tuple[str, Dict[str, str]]:
        """컨텐츠 압축"""

        logger.info(f"🗜️ Compressing content to fit {target_tokens} tokens...")

        # 우선순위 순서 (중요도 순)
        section_priorities = [
            "overview",
            "papers",
            "authors",
            "relationships",
            "keywords",
            "statistics",
        ]

        compressed_sections = {}
        current_tokens = 0

        for section_name in section_priorities:
            if section_name not in sections:
                continue

            section_text = sections[section_name]
            section_tokens = self._estimate_tokens(section_text)

            if current_tokens + section_tokens <= target_tokens:
                # 전체 포함
                compressed_sections[section_name] = section_text
                current_tokens += section_tokens
            else:
                # 부분 포함
                remaining_tokens = target_tokens - current_tokens
                if remaining_tokens > 100:  # 최소 100 토큰은 있어야 의미가 있음
                    # 섹션을 압축해서 포함
                    compressed_text = self._compress_section(
                        section_text, remaining_tokens
                    )
                    compressed_sections[section_name] = compressed_text
                break

        # 압축된 메인 텍스트 재조합
        compressed_main_text = self._assemble_main_text(
            compressed_sections, config, self.config.language
        )

        return compressed_main_text, compressed_sections

    def _compress_section(self, section_text: str, target_tokens: int) -> str:
        """개별 섹션 압축"""
        lines = section_text.split("\n")

        # 중요한 줄들 우선 선택 (헤더, 요약 정보 등)
        important_lines = []
        detail_lines = []

        for line in lines:
            if (
                line.startswith("#")
                or line.startswith("**")
                or "total" in line.lower()
                or "relevance" in line.lower()
            ):
                important_lines.append(line)
            else:
                detail_lines.append(line)

        # 중요한 줄들을 먼저 포함
        result_lines = important_lines[:]
        current_tokens = self._estimate_tokens("\n".join(result_lines))

        # 남은 토큰으로 상세 정보 추가
        for line in detail_lines:
            line_tokens = self._estimate_tokens(line)
            if current_tokens + line_tokens <= target_tokens:
                result_lines.append(line)
                current_tokens += line_tokens
            else:
                break

        # 잘렸다는 표시 추가
        if len(result_lines) < len(lines):
            result_lines.append("*[Content truncated due to length limits]*")

        return "\n".join(result_lines)

    def _is_mixed_language(self, text: str) -> bool:
        """혼용 언어 여부 확인"""
        korean_chars = len(re.findall(r"[가-힣]", text))
        english_chars = len(re.findall(r"[a-zA-Z]", text))

        total_chars = korean_chars + english_chars
        if total_chars == 0:
            return False

        korean_ratio = korean_chars / total_chars
        return 0.1 < korean_ratio < 0.9  # 10%-90% 사이면 혼용으로 판단


def main():
    """테스트 실행"""
    print("🧪 Testing ContextSerializer...")

    # 테스트용 더미 SubgraphResult 생성
    from dataclasses import dataclass
    from typing import Dict, List, Any

    # 더미 데이터 생성
    test_nodes = {
        "paper_1": {
            "id": "paper_1",
            "node_type": "paper",
            "title": "Deep Learning for Battery State of Charge Prediction using LSTM Networks",
            "year": "2023",
            "authors": ["김철수", "John Smith"],
            "journal": "IEEE Transactions on Power Electronics",
        },
        "author_1": {
            "id": "author_1",
            "node_type": "author",
            "name": "김철수",
            "paper_count": 15,
            "productivity_type": "Leading Researcher",
            "top_keywords": [("machine learning", 8), ("battery", 6)],
        },
        "keyword_1": {
            "id": "keyword_1",
            "node_type": "keyword",
            "name": "machine learning",
            "frequency": 25,
        },
    }

    test_edges = [
        {"source": "paper_1", "target": "author_1", "edge_type": "authored_by"},
        {"source": "paper_1", "target": "keyword_1", "edge_type": "has_keyword"},
    ]

    # 더미 SearchResult
    from collections import namedtuple

    SearchResult = namedtuple("SearchResult", ["node_id", "similarity_score"])

    test_initial_matches = [
        SearchResult("paper_1", 0.85),
        SearchResult("keyword_1", 0.75),
    ]

    # 더미 SubgraphResult
    from enum import Enum

    class SearchStrategy(Enum):
        HYBRID = "hybrid"

    class DummySubgraphResult:
        def __init__(self):
            self.nodes = test_nodes
            self.edges = test_edges
            self.query = "배터리 SoC 예측에 사용된 머신러닝 기법들은?"
            self.query_analysis = None
            self.extraction_strategy = SearchStrategy.HYBRID
            self.total_nodes = len(test_nodes)
            self.total_edges = len(test_edges)
            self.nodes_by_type = {"paper": 1, "author": 1, "keyword": 1}
            self.extraction_time = 1.5
            self.initial_matches = test_initial_matches
            self.expansion_path = []
            self.relevance_scores = {"paper_1": 0.9, "author_1": 0.7, "keyword_1": 0.8}
            self.confidence_score = 0.85

    # 테스트 실행
    test_result = DummySubgraphResult()

    # ContextSerializer 초기화
    config = SerializationConfig(
        format_style=SerializationFormat.STRUCTURED, max_tokens=2000, language="mixed"
    )

    serializer = ContextSerializer(config)

    # 직렬화 수행
    serialized = serializer.serialize(test_result)

    print(f"✅ Serialization completed:")
    print(f"   📏 Tokens: {serialized.total_tokens}")
    print(f"   🌐 Language: {serialized.language}")
    print(f"   📊 Compression: {serialized.compression_ratio:.1%}")

    print(f"\n📝 Generated Text:")
    print("=" * 60)
    print(serialized.main_text)
    print("=" * 60)

    print(f"\n✅ ContextSerializer test completed!")


if __name__ == "__main__":
    main()
