"""
통합 지식 그래프 구축 모듈
Unified Knowledge Graph Builder Module

6개의 개별 그래프를 하나의 통합된 지식 그래프로 결합
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import logging
from typing import Dict, List, Set, Tuple, Optional, Any

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedKnowledgeGraphBuilder:
    """6개의 개별 그래프를 통합된 지식 그래프로 구축하는 클래스"""

    def __init__(self, graphs_dir: Path):
        """
        Args:
            graphs_dir (Path): 개별 그래프 파일들이 저장된 디렉토리
        """
        self.graphs_dir = Path(graphs_dir)
        self.unified_graph = nx.MultiDiGraph()  # 다중 엣지 + 방향성 지원

        # 그래프별 파일 경로 정의
        self.graph_files = {
            "citation": "citation_network_graph.json",
            "keyword": "keyword_cooccurrence_graph.json",
            "semantic": "semantic_similarity_network_graph.json",
            "author_collab": "author_collaboration_graph.json",
            "author_paper": "author_paper_graph.json",
            "journal_paper": "journal_paper_graph.json",
        }

        # 노드 타입별 통계
        self.node_stats = defaultdict(int)
        self.edge_stats = defaultdict(int)

        # 통합 과정에서 발견된 문제들 추적
        self.integration_issues = {
            "missing_files": [],
            "node_conflicts": [],
            "data_inconsistencies": [],
        }

    def load_individual_graph(self, graph_name: str) -> Optional[nx.Graph]:
        """개별 그래프 파일 로드"""
        file_path = self.graphs_dir / self.graph_files[graph_name]

        if not file_path.exists():
            logger.warning(f"Graph file not found: {file_path}")
            self.integration_issues["missing_files"].append(str(file_path))
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)

            # NetworkX 그래프로 변환
            if graph_name in ["citation", "semantic"]:
                G = nx.DiGraph()  # 방향 그래프
            else:
                G = nx.Graph()  # 무방향 그래프

            # 노드 추가
            for node_data in graph_data.get("nodes", []):
                node_id = node_data["id"]
                attributes = {k: v for k, v in node_data.items() if k != "id"}
                G.add_node(node_id, **attributes)

            # 엣지 추가
            for edge_data in graph_data.get("edges", []):
                source = edge_data["source"]
                target = edge_data["target"]
                attributes = {
                    k: v for k, v in edge_data.items() if k not in ["source", "target"]
                }
                G.add_edge(source, target, **attributes)

            logger.info(
                f"✅ {graph_name} graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
            )
            return G

        except Exception as e:
            logger.error(f"❌ Error loading {graph_name} graph: {e}")
            return None

    def standardize_node_attributes(
        self, node_id: str, node_data: Dict[str, Any], source_graph: str
    ) -> Dict[str, Any]:
        """노드 속성 표준화 - Abstract 포함 버전"""

        standardized = node_data.copy()

        # 공통 속성 추가
        standardized["source_graphs"] = [source_graph]
        standardized["integration_timestamp"] = pd.Timestamp.now().isoformat()

        # node_type 표준화 (필수)
        if "node_type" not in standardized:
            # 노드 ID나 source_graph로부터 추론
            if "paper" in node_id.lower() or "paper" in source_graph:
                standardized["node_type"] = "paper"
            elif "author" in node_id.lower() or "author" in source_graph:
                standardized["node_type"] = "author"
            elif "keyword" in node_id.lower() or "keyword" in source_graph:
                standardized["node_type"] = "keyword"
            elif "journal" in node_id.lower() or "journal" in source_graph:
                standardized["node_type"] = "journal"
            else:
                standardized["node_type"] = "unknown"

        # 노드 타입별 특별 처리
        node_type = standardized.get("node_type", "unknown")

        if node_type == "paper":
            # ✅ 논문 노드에 Abstract 관련 처리 추가

            # 1. 기본 필드들 정리
            essential_fields = ["title", "authors", "year", "journal", "keywords"]
            for field in essential_fields:
                if field not in standardized:
                    standardized[field] = ""

            # 2. ✅ Abstract 처리 (핵심 추가)
            abstract_sources = [
                "abstract",  # 직접적인 abstract 필드
                "description",  # 일부 소스에서 사용
                "summary",  # 요약 필드
                "content",  # 일반적인 내용 필드
            ]

            abstract_content = ""
            for field in abstract_sources:
                if field in node_data and node_data[field]:
                    content = str(node_data[field]).strip()
                    if len(content) > len(abstract_content):
                        abstract_content = content

            standardized["abstract"] = abstract_content
            standardized["has_abstract"] = bool(abstract_content)

            # 3. ✅ Abstract 품질 분석
            if abstract_content:
                # Abstract 길이 분석
                standardized["abstract_length"] = len(abstract_content)
                standardized["abstract_word_count"] = len(abstract_content.split())

                # Abstract 품질 점수 (길이 기반)
                if len(abstract_content) > 100:
                    standardized["abstract_quality"] = "good"
                elif len(abstract_content) > 50:
                    standardized["abstract_quality"] = "fair"
                else:
                    standardized["abstract_quality"] = "poor"
            else:
                standardized["abstract_length"] = 0
                standardized["abstract_word_count"] = 0
                standardized["abstract_quality"] = "none"

            # 4. 키워드 처리 개선
            keywords = standardized.get("keywords", "")
            if isinstance(keywords, list):
                keywords = "; ".join(str(k) for k in keywords)
            elif not isinstance(keywords, str):
                keywords = str(keywords)

            # 키워드 정제
            if keywords:
                keyword_list = [kw.strip() for kw in keywords.split(";") if kw.strip()]
                standardized["keywords"] = "; ".join(keyword_list)
                standardized["keyword_count"] = len(keyword_list)
            else:
                standardized["keywords"] = ""
                standardized["keyword_count"] = 0

            # 5. 저자 처리 개선
            authors = standardized.get("authors", [])
            if isinstance(authors, str):
                authors = [a.strip() for a in authors.split(",") if a.strip()]
            elif not isinstance(authors, list):
                authors = [str(authors)]

            standardized["authors"] = authors
            standardized["author_count"] = len(authors)

            # 저자 관련 메타데이터
            if len(authors) == 1:
                standardized["collaboration_type"] = "Single Author"
            elif len(authors) <= 3:
                standardized["collaboration_type"] = "Small Team"
            else:
                standardized["collaboration_type"] = "Large Team"

            # 6. 연도 정규화
            year = standardized.get("year", "")
            if year:
                try:
                    year_int = int(str(year))
                    if 1900 <= year_int <= 2030:  # 합리적인 범위
                        standardized["year"] = year_int
                    else:
                        standardized["year"] = None
                except:
                    standardized["year"] = None
            else:
                standardized["year"] = None

            # 7. ✅ 논문 분류 및 특성 분석
            title = standardized.get("title", "").lower()
            abstract_lower = abstract_content.lower()

            # ML/AI 관련 키워드 탐지
            ml_keywords = [
                "machine learning",
                "deep learning",
                "neural network",
                "artificial intelligence",
                "reinforcement learning",
                "supervised learning",
                "unsupervised learning",
                "classification",
                "regression",
                "clustering",
                "algorithm",
            ]

            battery_keywords = [
                "battery",
                "lithium",
                "soc",
                "state of charge",
                "electric vehicle",
                "energy storage",
                "charging",
                "power management",
                "thermal management",
            ]

            ml_score = sum(
                1 for kw in ml_keywords if kw in title or kw in abstract_lower
            )
            battery_score = sum(
                1 for kw in battery_keywords if kw in title or kw in abstract_lower
            )

            standardized["ml_relevance_score"] = ml_score
            standardized["battery_relevance_score"] = battery_score
            standardized["is_interdisciplinary"] = ml_score > 0 and battery_score > 0

            # 경험있는 저자 여부 (휴리스틱)
            experienced_indicators = ["professor", "dr.", "phd", "senior", "lead"]
            author_text = " ".join(authors).lower()
            standardized["has_experienced_authors"] = any(
                indicator in author_text for indicator in experienced_indicators
            )

        elif node_type == "author":
            # 저자 노드 처리 (기존 유지)
            essential_fields = ["name", "paper_count", "collaborator_count"]
            for field in essential_fields:
                if field not in standardized:
                    if field == "name":
                        standardized[field] = node_id
                    else:
                        standardized[field] = 0

        elif node_type == "keyword":
            # 키워드 노드 처리 (기존 유지)
            if "name" not in standardized:
                standardized["name"] = node_id
            if "frequency" not in standardized:
                standardized["frequency"] = 1

        elif node_type == "journal":
            # 저널 노드 처리 (기존 유지)
            essential_fields = ["name", "paper_count"]
            for field in essential_fields:
                if field not in standardized:
                    if field == "name":
                        standardized[field] = node_id
                    else:
                        standardized[field] = 0

        # ID 정규화
        standardized["id"] = node_id

        return standardized

    def merge_duplicate_nodes(
        self,
        node_id: str,
        new_attributes: Dict[str, Any],
        existing_attributes: Dict[str, Any],
    ) -> Dict[str, Any]:
        """중복 노드 병합 - Abstract 고려 버전"""

        merged = existing_attributes.copy()

        # source_graphs 병합
        existing_sources = set(merged.get("source_graphs", []))
        new_sources = set(new_attributes.get("source_graphs", []))
        merged["source_graphs"] = list(existing_sources | new_sources)

        # 노드 타입별 특별 처리
        node_type = merged.get("node_type", "unknown")

        if node_type == "paper":
            # ✅ 논문 정보 병합 - Abstract 우선 처리

            # Abstract 병합 (더 긴 것 선택)
            existing_abstract = merged.get("abstract", "")
            new_abstract = new_attributes.get("abstract", "")

            if len(new_abstract) > len(existing_abstract):
                merged["abstract"] = new_abstract
                merged["has_abstract"] = bool(new_abstract)
                merged["abstract_length"] = len(new_abstract)
                merged["abstract_word_count"] = len(new_abstract.split())

                # Abstract 품질 재계산
                if len(new_abstract) > 100:
                    merged["abstract_quality"] = "good"
                elif len(new_abstract) > 50:
                    merged["abstract_quality"] = "fair"
                else:
                    merged["abstract_quality"] = "poor"

            # 다른 텍스트 필드들도 더 완전한 정보로 업데이트
            text_fields = ["title", "keywords"]
            for field in text_fields:
                if field in new_attributes and field in merged:
                    if len(str(new_attributes[field])) > len(str(merged[field])):
                        merged[field] = new_attributes[field]
                elif field in new_attributes:
                    merged[field] = new_attributes[field]

            # 저자 정보 병합 (더 많은 저자 정보 선택)
            if "authors" in new_attributes and "authors" in merged:
                existing_authors = (
                    merged["authors"] if isinstance(merged["authors"], list) else []
                )
                new_authors = (
                    new_attributes["authors"]
                    if isinstance(new_attributes["authors"], list)
                    else []
                )

                if len(new_authors) > len(existing_authors):
                    merged["authors"] = new_authors
                    merged["author_count"] = len(new_authors)

            # 수치형 필드들은 더 높은 값 선택
            numeric_fields = [
                "ml_relevance_score",
                "battery_relevance_score",
                "keyword_count",
            ]
            for field in numeric_fields:
                if field in new_attributes and field in merged:
                    merged[field] = max(
                        merged.get(field, 0), new_attributes.get(field, 0)
                    )
                elif field in new_attributes:
                    merged[field] = new_attributes[field]

            # Boolean 필드들은 OR 연산
            boolean_fields = ["is_interdisciplinary", "has_experienced_authors"]
            for field in boolean_fields:
                if field in new_attributes and field in merged:
                    merged[field] = merged.get(field, False) or new_attributes.get(
                        field, False
                    )
                elif field in new_attributes:
                    merged[field] = new_attributes[field]

        elif node_type == "author":
            # 저자 통계 정보 병합 (최대값 선택)
            numeric_fields = ["paper_count", "collaborator_count", "first_author_count"]
            for field in numeric_fields:
                if field in new_attributes and field in merged:
                    merged[field] = max(merged[field], new_attributes[field])
                elif field in new_attributes:
                    merged[field] = new_attributes[field]

        elif node_type == "keyword":
            # 키워드 빈도 합산
            if "frequency" in new_attributes and "frequency" in merged:
                merged["frequency"] = merged["frequency"] + new_attributes["frequency"]

        # 기타 새로운 속성 추가
        for key, value in new_attributes.items():
            if key not in merged and key not in ["source_graphs"]:
                merged[key] = value

        return merged

    def standardize_edge_attributes(
        self, edge_data: Dict[str, Any], source_graph: str
    ) -> Dict[str, Any]:
        """엣지 속성 표준화"""
        standardized = edge_data.copy()

        # 공통 속성 추가
        standardized["source_graph"] = source_graph
        standardized["integration_timestamp"] = pd.Timestamp.now().isoformat()

        # edge_type 표준화
        if "edge_type" not in standardized:
            # 소스 그래프에 따른 기본 edge_type 설정
            edge_type_mapping = {
                "citation": "cites",
                "keyword": "co_occurs_with",
                "semantic": "semantically_similar_to",
                "author_collab": "collaborates_with",
                "author_paper": "authored_by",
                "journal_paper": "published_in",
            }
            standardized["edge_type"] = edge_type_mapping.get(
                source_graph, "related_to"
            )

        # 가중치 표준화 (0-1 범위로)
        if "weight" in standardized:
            weight = standardized["weight"]
            if isinstance(weight, (int, float)) and weight > 1:
                # 1보다 큰 가중치는 정규화 (로그 스케일 고려)
                standardized["normalized_weight"] = min(1.0, weight / 100.0)
            else:
                standardized["normalized_weight"] = float(weight)
        else:
            standardized["normalized_weight"] = 1.0

        return standardized

    def add_cross_graph_edges(self):
        """그래프 간 추가 연결 엣지 생성"""
        logger.info("🔗 Creating cross-graph connections...")

        # 논문-키워드 연결 (keywords 속성 기반)
        self._connect_papers_to_keywords()

        # 저자-키워드 연결 (저자의 논문 키워드 기반)
        self._connect_authors_to_keywords()

        # 저널-키워드 연결 (저널 논문들의 키워드 기반)
        self._connect_journals_to_keywords()

        # 시간적 연결 (같은 연도 논문들)
        self._create_temporal_connections()

    def _connect_papers_to_keywords(self):
        """논문과 키워드 간 연결 생성"""
        paper_nodes = [
            n
            for n in self.unified_graph.nodes()
            if self.unified_graph.nodes[n].get("node_type") == "paper"
        ]
        keyword_nodes = [
            n
            for n in self.unified_graph.nodes()
            if self.unified_graph.nodes[n].get("node_type") == "keyword"
        ]

        connections_added = 0

        for paper_id in paper_nodes:
            paper_data = self.unified_graph.nodes[paper_id]
            paper_keywords = paper_data.get("keywords", [])

            if isinstance(paper_keywords, str):
                paper_keywords = [kw.strip() for kw in paper_keywords.split(";")]

            for keyword in paper_keywords:
                keyword_clean = keyword.lower().strip()

                # 정확히 일치하는 키워드 노드 찾기
                matching_keyword = None
                for kw_node in keyword_nodes:
                    if kw_node.lower() == keyword_clean:
                        matching_keyword = kw_node
                        break

                if matching_keyword and not self.unified_graph.has_edge(
                    paper_id, matching_keyword
                ):
                    self.unified_graph.add_edge(
                        paper_id,
                        matching_keyword,
                        edge_type="has_keyword",
                        source_graph="cross_connection",
                        weight=1.0,
                        normalized_weight=1.0,
                    )
                    connections_added += 1

        logger.info(f"📝 Added {connections_added} paper-keyword connections")

    def _connect_authors_to_keywords(self):
        """저자와 키워드 간 연결 생성 (저자 논문의 키워드 기반)"""
        author_nodes = [
            n
            for n in self.unified_graph.nodes()
            if self.unified_graph.nodes[n].get("node_type") == "author"
        ]

        connections_added = 0

        for author_id in author_nodes:
            # 저자의 모든 논문 찾기
            author_papers = []
            for edge in self.unified_graph.edges(data=True):
                if (
                    edge[2].get("edge_type") == "authored_by" and edge[1] == author_id
                ):  # target이 author
                    author_papers.append(edge[0])  # source는 paper

            # 저자 논문들의 키워드 수집
            author_keywords = Counter()
            for paper_id in author_papers:
                paper_data = self.unified_graph.nodes.get(paper_id, {})
                paper_keywords = paper_data.get("keywords", [])

                if isinstance(paper_keywords, str):
                    paper_keywords = [kw.strip() for kw in paper_keywords.split(";")]

                for keyword in paper_keywords:
                    if keyword.strip():
                        author_keywords[keyword.lower().strip()] += 1

            # 빈도 높은 키워드와 연결 (빈도 2 이상)
            for keyword, freq in author_keywords.items():
                if freq >= 2:  # 최소 2번 이상 사용한 키워드
                    # 해당 키워드 노드 찾기
                    keyword_nodes = [
                        n
                        for n in self.unified_graph.nodes()
                        if (
                            self.unified_graph.nodes[n].get("node_type") == "keyword"
                            and n.lower() == keyword
                        )
                    ]

                    for kw_node in keyword_nodes:
                        if not self.unified_graph.has_edge(author_id, kw_node):
                            self.unified_graph.add_edge(
                                author_id,
                                kw_node,
                                edge_type="specializes_in",
                                source_graph="cross_connection",
                                frequency=freq,
                                weight=min(1.0, freq / 10.0),  # 빈도 기반 가중치
                                normalized_weight=min(1.0, freq / 10.0),
                            )
                            connections_added += 1

        logger.info(f"👥 Added {connections_added} author-keyword connections")

    def _connect_journals_to_keywords(self):
        """저널과 키워드 간 연결 생성"""
        journal_nodes = [
            n
            for n in self.unified_graph.nodes()
            if self.unified_graph.nodes[n].get("node_type") == "journal"
        ]

        connections_added = 0

        for journal_id in journal_nodes:
            # 저널의 모든 논문 찾기
            journal_papers = []
            for edge in self.unified_graph.edges(data=True):
                if (
                    edge[2].get("edge_type") == "published_in" and edge[1] == journal_id
                ):  # target이 journal
                    journal_papers.append(edge[0])  # source는 paper

            # 저널 논문들의 키워드 수집
            journal_keywords = Counter()
            for paper_id in journal_papers:
                paper_data = self.unified_graph.nodes.get(paper_id, {})
                paper_keywords = paper_data.get("keywords", [])

                if isinstance(paper_keywords, str):
                    paper_keywords = [kw.strip() for kw in paper_keywords.split(";")]

                for keyword in paper_keywords:
                    if keyword.strip():
                        journal_keywords[keyword.lower().strip()] += 1

            # 상위 키워드들과 연결 (상위 20%만)
            if journal_keywords:
                top_keywords = journal_keywords.most_common(
                    max(5, len(journal_keywords) // 5)
                )

                for keyword, freq in top_keywords:
                    keyword_nodes = [
                        n
                        for n in self.unified_graph.nodes()
                        if (
                            self.unified_graph.nodes[n].get("node_type") == "keyword"
                            and n.lower() == keyword
                        )
                    ]

                    for kw_node in keyword_nodes:
                        if not self.unified_graph.has_edge(journal_id, kw_node):
                            self.unified_graph.add_edge(
                                journal_id,
                                kw_node,
                                edge_type="focuses_on",
                                source_graph="cross_connection",
                                frequency=freq,
                                weight=min(1.0, freq / max(journal_keywords.values())),
                                normalized_weight=min(
                                    1.0, freq / max(journal_keywords.values())
                                ),
                            )
                            connections_added += 1

        logger.info(f"📰 Added {connections_added} journal-keyword connections")

    def _create_temporal_connections(self):
        """시간적 근접성 기반 연결 생성"""
        paper_nodes = [
            n
            for n in self.unified_graph.nodes()
            if self.unified_graph.nodes[n].get("node_type") == "paper"
        ]

        # 연도별 논문 그룹화
        papers_by_year = defaultdict(list)
        for paper_id in paper_nodes:
            year = self.unified_graph.nodes[paper_id].get("year", "")
            if year and str(year).isdigit():
                papers_by_year[int(year)].append(paper_id)

        connections_added = 0

        # 같은 연도 논문들 간 약한 연결 (샘플링)
        for year, papers in papers_by_year.items():
            if len(papers) > 1:
                # 너무 많으면 샘플링 (최대 50개 논문만)
                if len(papers) > 50:
                    import random

                    papers = random.sample(papers, 50)

                # 모든 쌍이 아닌 일부만 연결 (computational cost 고려)
                for i in range(min(10, len(papers))):  # 각 논문당 최대 10개만 연결
                    for j in range(i + 1, min(i + 11, len(papers))):
                        paper1, paper2 = papers[i], papers[j]

                        if not self.unified_graph.has_edge(paper1, paper2):
                            self.unified_graph.add_edge(
                                paper1,
                                paper2,
                                edge_type="temporal_proximity",
                                source_graph="cross_connection",
                                year=year,
                                weight=0.1,  # 약한 연결
                                normalized_weight=0.1,
                            )
                            connections_added += 1

        logger.info(f"⏰ Added {connections_added} temporal proximity connections")

    def calculate_unified_statistics(self):
        """통합 그래프 통계 계산"""
        logger.info("📊 Calculating unified graph statistics...")

        # 노드 타입별 통계
        node_types = defaultdict(int)
        for node in self.unified_graph.nodes():
            node_type = self.unified_graph.nodes[node].get("node_type", "unknown")
            node_types[node_type] += 1
        # ✅ Abstract 관련 통계 추가
        abstract_stats = {
            "papers_with_abstract": 0,
            "papers_without_abstract": 0,
            "total_abstract_length": 0,
            "average_abstract_length": 0,
            "abstract_quality_distribution": {
                "good": 0,
                "fair": 0,
                "poor": 0,
                "none": 0,
            },
        }

        paper_nodes = [
            n
            for n in self.unified_graph.nodes()
            if self.unified_graph.nodes[n].get("node_type") == "paper"
        ]

        for paper_id in paper_nodes:
            paper_data = self.unified_graph.nodes[paper_id]
            has_abstract = paper_data.get("has_abstract", False)
            abstract_length = paper_data.get("abstract_length", 0)
            abstract_quality = paper_data.get("abstract_quality", "none")

            if has_abstract:
                abstract_stats["papers_with_abstract"] += 1
                abstract_stats["total_abstract_length"] += abstract_length
            else:
                abstract_stats["papers_without_abstract"] += 1

            abstract_stats["abstract_quality_distribution"][abstract_quality] += 1

        if abstract_stats["papers_with_abstract"] > 0:
            abstract_stats["average_abstract_length"] = (
                abstract_stats["total_abstract_length"]
                / abstract_stats["papers_with_abstract"]
            )

        # 엣지 타입별 통계
        edge_types = defaultdict(int)
        for edge in self.unified_graph.edges(data=True):
            edge_type = edge[2].get("edge_type", "unknown")
            edge_types[edge_type] += 1

        # 소스 그래프별 기여도
        source_contributions = defaultdict(lambda: {"nodes": 0, "edges": 0})

        for node in self.unified_graph.nodes():
            sources = self.unified_graph.nodes[node].get("source_graphs", ["unknown"])
            for source in sources:
                source_contributions[source]["nodes"] += 1

        for edge in self.unified_graph.edges(data=True):
            source = edge[2].get("source_graph", "unknown")
            source_contributions[source]["edges"] += 1

        # 연결성 분석
        if nx.is_connected(self.unified_graph.to_undirected()):
            connectivity = "fully_connected"
            largest_component_size = self.unified_graph.number_of_nodes()
        else:
            components = list(
                nx.connected_components(self.unified_graph.to_undirected())
            )
            connectivity = "disconnected"
            largest_component_size = len(max(components, key=len))

        stats = {
            "basic_info": {
                "total_nodes": self.unified_graph.number_of_nodes(),
                "total_edges": self.unified_graph.number_of_edges(),
                "density": nx.density(self.unified_graph),
                "connectivity": connectivity,
                "largest_component_size": largest_component_size,
            },
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "abstract_statistics": abstract_stats,  # ✅ 새로 추가
            "source_contributions": dict(source_contributions),
            "integration_issues": self.integration_issues,
        }
        # Abstract 통계 로깅
        logger.info(f"📄 Abstract Statistics:")
        logger.info(
            f"   Papers with abstract: {abstract_stats['papers_with_abstract']}"
        )
        logger.info(
            f"   Papers without abstract: {abstract_stats['papers_without_abstract']}"
        )
        logger.info(
            f"   Average abstract length: {abstract_stats['average_abstract_length']:.1f} chars"
        )
        return stats

    def save_unified_graph(
        self, output_dir: Path, save_formats: List[str] = ["json", "graphml"]
    ):
        """통합 그래프 저장"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        saved_files = []

        # JSON 형태로 저장 (GraphRAG에서 사용)
        if "json" in save_formats:
            graph_data = {"nodes": [], "edges": []}

            # 노드 정보
            for node in self.unified_graph.nodes():
                node_data = self.unified_graph.nodes[node].copy()
                node_data["id"] = node

                # 리스트/복잡한 타입 처리
                for key, value in node_data.items():
                    if isinstance(value, (list, set)):
                        node_data[key] = list(value)
                    elif not isinstance(value, (str, int, float, bool, type(None))):
                        node_data[key] = str(value)

                graph_data["nodes"].append(node_data)

            # 엣지 정보
            for edge in self.unified_graph.edges(data=True):
                edge_data = edge[2].copy()
                edge_data["source"] = edge[0]
                edge_data["target"] = edge[1]

                # 리스트/복잡한 타입 처리
                for key, value in edge_data.items():
                    if isinstance(value, (list, set)):
                        edge_data[key] = list(value)
                    elif not isinstance(value, (str, int, float, bool, type(None))):
                        edge_data[key] = str(value)

                graph_data["edges"].append(edge_data)

            json_file = output_dir / "unified_knowledge_graph.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)

            saved_files.append(json_file)
            logger.info(f"💾 JSON graph saved: {json_file}")

        # GraphML 형태로 저장 (Gephi, Cytoscape 등에서 사용)
        if "graphml" in save_formats:
            try:
                # GraphML 호환을 위해 복잡한 속성 문자열화
                G_graphml = self.unified_graph.copy()

                for node in G_graphml.nodes():
                    for attr_name, attr_value in G_graphml.nodes[node].items():
                        if isinstance(attr_value, (list, set)):
                            G_graphml.nodes[node][attr_name] = ";".join(
                                str(v) for v in attr_value
                            )
                        elif not isinstance(attr_value, (str, int, float, bool)):
                            G_graphml.nodes[node][attr_name] = str(attr_value)

                for edge in G_graphml.edges(data=True):
                    edge_data = edge[2]
                    for attr_name, attr_value in edge_data.items():
                        if isinstance(attr_value, (list, set)):
                            edge_data[attr_name] = ";".join(str(v) for v in attr_value)
                        elif not isinstance(attr_value, (str, int, float, bool)):
                            edge_data[attr_name] = str(attr_value)

                graphml_file = output_dir / "unified_knowledge_graph.graphml"
                nx.write_graphml(G_graphml, graphml_file)
                saved_files.append(graphml_file)
                logger.info(f"💾 GraphML graph saved: {graphml_file}")

            except Exception as e:
                logger.warning(f"⚠️ GraphML 저장 실패: {e}")

        return saved_files

    def build_unified_graph(
        self, save_output: bool = True, output_dir: Optional[Path] = None
    ) -> nx.MultiDiGraph:
        """전체 통합 그래프 구축 파이프라인"""
        logger.info("🚀 Starting unified knowledge graph construction...")

        # 1. 개별 그래프들 로드
        individual_graphs = {}
        for graph_name in self.graph_files.keys():
            logger.info(f"📂 Loading {graph_name} graph...")
            graph = self.load_individual_graph(graph_name)
            if graph:
                individual_graphs[graph_name] = graph

        if not individual_graphs:
            raise ValueError("❌ No graphs loaded successfully!")

        logger.info(f"✅ Loaded {len(individual_graphs)} graphs successfully")

        # 2. 노드 통합
        logger.info("🔄 Integrating nodes...")
        nodes_added = 0
        nodes_merged = 0

        for graph_name, graph in individual_graphs.items():
            for node_id in tqdm(graph.nodes(), desc=f"Processing {graph_name} nodes"):
                node_attrs = graph.nodes[node_id]
                standardized_attrs = self.standardize_node_attributes(
                    node_id, node_attrs, graph_name
                )

                if self.unified_graph.has_node(node_id):
                    # 기존 노드와 병합
                    existing_attrs = self.unified_graph.nodes[node_id]
                    merged_attrs = self.merge_duplicate_nodes(
                        node_id, standardized_attrs, existing_attrs
                    )
                    self.unified_graph.nodes[node_id].update(merged_attrs)
                    nodes_merged += 1
                else:
                    # 새 노드 추가
                    self.unified_graph.add_node(node_id, **standardized_attrs)
                    nodes_added += 1

        logger.info(f"📝 Nodes: {nodes_added} added, {nodes_merged} merged")

        # 3. 엣지 통합
        logger.info("🔗 Integrating edges...")
        edges_added = 0

        for graph_name, graph in individual_graphs.items():
            for source, target, edge_attrs in tqdm(
                graph.edges(data=True), desc=f"Processing {graph_name} edges"
            ):
                standardized_attrs = self.standardize_edge_attributes(
                    edge_attrs, graph_name
                )

                # MultiDiGraph이므로 중복 엣지도 추가 가능 (key로 구분)
                edge_key = (
                    f"{graph_name}_{standardized_attrs.get('edge_type', 'unknown')}"
                )
                self.unified_graph.add_edge(
                    source, target, key=edge_key, **standardized_attrs
                )
                edges_added += 1

        logger.info(f"🔗 Added {edges_added} edges")

        # 4. 그래프 간 추가 연결 생성
        self.add_cross_graph_edges()

        # 5. 통계 계산
        stats = self.calculate_unified_statistics()

        # 6. 결과 저장
        if save_output:
            if not output_dir:
                output_dir = self.graphs_dir / "unified"

            saved_files = self.save_unified_graph(output_dir)

            # 통계도 저장
            stats_file = output_dir / "unified_graph_statistics.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            logger.info(f"📊 Statistics saved: {stats_file}")

        # 7. 최종 요약
        logger.info("🎉 Unified Knowledge Graph Construction Complete!")
        logger.info(f"📊 Final Stats:")
        logger.info(f"   Total Nodes: {stats['basic_info']['total_nodes']:,}")
        logger.info(f"   Total Edges: {stats['basic_info']['total_edges']:,}")
        logger.info(f"   Graph Density: {stats['basic_info']['density']:.6f}")
        logger.info(f"   Node Types: {stats['node_types']}")
        logger.info(f"   Edge Types: {len(stats['edge_types'])} types")

        return self.unified_graph


def main():
    """메인 실행 함수"""
    from src import GRAPHS_DIR

    # Unified Knowledge Graph Builder 초기화
    builder = UnifiedKnowledgeGraphBuilder(GRAPHS_DIR)

    # 통합 그래프 구축
    unified_graph = builder.build_unified_graph(save_output=True)

    print(f"\n✅ Unified Knowledge Graph 구축 완료!")
    print(f"📁 출력 디렉토리: {GRAPHS_DIR / 'unified'}")
    print(f"🚀 다음 단계: Query Analyzer 구축")

    return unified_graph


if __name__ == "__main__":
    main()
