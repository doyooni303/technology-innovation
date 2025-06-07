"""
서브그래프 추출 모듈
Subgraph Extractor for GraphRAG System

벡터 검색과 그래프 탐색을 결합하여 쿼리 관련 서브그래프를 추출
- 다양한 검색 전략 지원 (Local, Global, Hybrid)
- 쿼리 타입별 최적화된 탐색
- 효율적인 그래프 탐색 알고리즘
- LLM 토큰 제한 고려한 크기 조절
"""

import json
import logging
import warnings
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import numpy as np
import networkx as nx
from tqdm import tqdm

# GraphRAG imports
try:
    from ..embeddings.vector_store_manager import VectorStoreManager, SearchResult
    from ..embeddings.embedding_models import create_embedding_model
    from ..query_analyzer import (
        QueryAnalysisResult,
        QueryComplexity,
        QueryType,
        SearchMode,
    )
except ImportError as e:
    warnings.warn(f"Some GraphRAG components not available: {e}")

# 로깅 설정
logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """검색 전략"""

    LOCAL = "local"  # 특정 노드 중심 확장
    GLOBAL = "global"  # 전역 구조 기반
    HYBRID = "hybrid"  # 혼합 전략
    COMMUNITY = "community"  # 커뮤니티 기반
    TEMPORAL = "temporal"  # 시간적 연관성
    SEMANTIC = "semantic"  # 의미적 유사도 우선


@dataclass
class ExtractionConfig:
    """서브그래프 추출 설정"""

    # 크기 제한
    max_nodes: int = 200
    max_edges: int = 500
    max_hops: int = 3

    # 검색 설정
    initial_top_k: int = 20  # 초기 벡터 검색 결과 수
    expansion_factor: float = 2.0  # 확장 비율
    similarity_threshold: float = 0.5

    # 전략별 가중치
    vector_weight: float = 0.4
    graph_weight: float = 0.3
    metadata_weight: float = 0.3

    # 노드 타입별 중요도
    node_type_weights: Dict[str, float] = None

    # 성능 설정
    enable_caching: bool = True
    parallel_processing: bool = False

    def __post_init__(self):
        if self.node_type_weights is None:
            self.node_type_weights = {
                "paper": 1.0,
                "author": 0.8,
                "keyword": 0.6,
                "journal": 0.5,
            }


@dataclass
class SubgraphResult:
    """서브그래프 추출 결과"""

    # 그래프 데이터
    nodes: Dict[str, Dict[str, Any]]  # node_id -> node_data
    edges: List[Dict[str, Any]]  # edge 리스트

    # 메타데이터
    query: str
    query_analysis: Optional[QueryAnalysisResult]
    extraction_strategy: SearchStrategy

    # 통계
    total_nodes: int
    total_edges: int
    nodes_by_type: Dict[str, int]
    extraction_time: float

    # 검색 결과
    initial_matches: List[SearchResult]
    expansion_path: List[Dict[str, Any]]

    # 스코어링
    relevance_scores: Dict[str, float]  # node_id -> relevance_score
    confidence_score: float

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "query": self.query,
            "extraction_strategy": self.extraction_strategy.value,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "nodes_by_type": self.nodes_by_type,
            "extraction_time": self.extraction_time,
            "confidence_score": self.confidence_score,
            "relevance_scores": self.relevance_scores,
        }

    def get_networkx_graph(self) -> nx.Graph:
        """NetworkX 그래프로 변환"""
        G = nx.Graph()

        # 노드 추가
        for node_id, node_data in self.nodes.items():
            G.add_node(node_id, **node_data)

        # 엣지 추가
        for edge in self.edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and target and G.has_node(source) and G.has_node(target):
                edge_attrs = {
                    k: v for k, v in edge.items() if k not in ["source", "target"]
                }
                G.add_edge(source, target, **edge_attrs)

        return G


class SubgraphExtractor:
    """서브그래프 추출기"""

    def __init__(
        self,
        unified_graph_path: str,
        vector_store_path: str,
        embedding_model: str = "auto",
        config: Optional[ExtractionConfig] = None,
        device: str = "auto",
    ):
        """
        Args:
            unified_graph_path: 통합 그래프 파일 경로
            vector_store_path: 벡터 저장소 경로
            embedding_model: 임베딩 모델명
            config: 추출 설정
            device: 디바이스 설정
        """
        self.unified_graph_path = Path(unified_graph_path)
        self.vector_store_path = Path(vector_store_path)
        self.config = config or ExtractionConfig()

        # 임베딩 모델 초기화
        self.embedding_model = create_embedding_model(
            model_name=embedding_model, device=device
        )

        # 데이터 저장
        self.graph_data = None
        self.networkx_graph = None
        self.vector_store = None

        # 캐시
        self.query_cache = {} if self.config.enable_caching else None
        self.node_neighbors_cache = {}

        logger.info("✅ SubgraphExtractor initialized")
        logger.info(f"   📁 Graph: {self.unified_graph_path}")
        logger.info(f"   🗄️ Vector Store: {self.vector_store_path}")

    def load_unified_graph(self) -> Dict[str, Any]:
        """통합 그래프 로드"""
        if self.graph_data is not None:
            return self.graph_data

        if not self.unified_graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {self.unified_graph_path}")

        logger.info(f"📂 Loading unified graph...")

        with open(self.unified_graph_path, "r", encoding="utf-8") as f:
            self.graph_data = json.load(f)

        # NetworkX 그래프 생성
        self._create_networkx_graph()

        logger.info(
            f"✅ Graph loaded: {len(self.graph_data['nodes'])} nodes, {len(self.graph_data['edges'])} edges"
        )
        return self.graph_data

    def _create_networkx_graph(self) -> None:
        """NetworkX 그래프 생성 (탐색 최적화용)"""
        logger.info("🔧 Creating NetworkX graph for efficient traversal...")

        # 방향 그래프로 생성 (엣지 타입 고려)
        self.networkx_graph = nx.MultiDiGraph()

        # 노드 추가
        for node in self.graph_data["nodes"]:
            node_id = node["id"]
            node_attrs = {k: v for k, v in node.items() if k != "id"}
            self.networkx_graph.add_node(node_id, **node_attrs)

        # 엣지 추가
        for edge in self.graph_data["edges"]:
            source = edge["source"]
            target = edge["target"]
            edge_attrs = {
                k: v for k, v in edge.items() if k not in ["source", "target"]
            }

            # 키 생성 (multiple edges 지원)
            edge_type = edge_attrs.get("edge_type", "default")
            self.networkx_graph.add_edge(source, target, key=edge_type, **edge_attrs)

        logger.info(f"✅ NetworkX graph created")

    def load_vector_store(self) -> VectorStoreManager:
        """벡터 저장소 로드 - 개선된 버전"""
        if self.vector_store is not None:
            return self.vector_store

        if not self.vector_store_path.exists():
            raise FileNotFoundError(f"Vector store not found: {self.vector_store_path}")

        logger.info(f"📂 Loading vector store from: {self.vector_store_path}")

        try:
            # 벡터 저장소 타입 자동 감지
            store_type = "auto"

            # FAISS 파일 확인
            if (self.vector_store_path / "faiss_index.bin").exists():
                store_type = "faiss"
                actual_path = self.vector_store_path
            # ChromaDB 파일 확인
            elif (self.vector_store_path / "chroma.sqlite3").exists():
                store_type = "chroma"
                actual_path = self.vector_store_path
            # 서브폴더에서 찾기
            elif (self.vector_store_path / "faiss" / "faiss_index.bin").exists():
                store_type = "faiss"
                actual_path = self.vector_store_path / "faiss"
            elif (self.vector_store_path / "chromadb" / "chroma.sqlite3").exists():
                store_type = "chroma"
                actual_path = self.vector_store_path / "chromadb"
            else:
                # 임베딩 파일에서 로드 시도
                embeddings_dir = self.vector_store_path / "embeddings"
                if (
                    embeddings_dir.exists()
                    and (embeddings_dir / "embeddings.npy").exists()
                ):
                    logger.info("📥 Building vector store from embeddings...")

                    # 기본 FAISS로 구축
                    store_type = "faiss"
                    actual_path = self.vector_store_path / "faiss"
                    actual_path.mkdir(exist_ok=True)

                    from .vector_store_manager import VectorStoreManager

                    self.vector_store = VectorStoreManager(
                        store_type="faiss", persist_directory=str(actual_path)
                    )

                    self.vector_store.load_from_saved_embeddings(
                        str(self.vector_store_path), embeddings_subdir="embeddings"
                    )

                    logger.info("✅ Vector store built from embeddings")
                    return self.vector_store
                else:
                    raise FileNotFoundError(
                        f"No vector store or embeddings found in: {self.vector_store_path}"
                    )

            # 벡터 저장소 로드
            from .vector_store_manager import VectorStoreManager

            self.vector_store = VectorStoreManager(
                store_type=store_type, persist_directory=str(actual_path)
            )

            logger.info(f"✅ Vector store loaded: {store_type} from {actual_path}")
            return self.vector_store

        except Exception as e:
            logger.error(f"❌ Vector store loading failed: {e}")
            raise

    def extract_subgraph(
        self,
        query: str,
        query_analysis: Optional[QueryAnalysisResult] = None,
        strategy: Optional[SearchStrategy] = None,
        custom_config: Optional[ExtractionConfig] = None,
    ) -> SubgraphResult:
        """메인 서브그래프 추출 함수"""

        # 설정 결정
        config = custom_config or self.config

        # 캐시 확인
        cache_key = (
            f"{query}_{strategy}_{hash(str(config.__dict__))}"
            if self.query_cache
            else None
        )
        if cache_key and cache_key in self.query_cache:
            logger.info("✅ Using cached result")
            return self.query_cache[cache_key]

        logger.info(f"🔍 Extracting subgraph for query: '{query[:50]}...'")

        import time

        start_time = time.time()

        # 데이터 로드
        self.load_unified_graph()
        self.load_vector_store()

        # 검색 전략 결정
        if strategy is None:
            strategy = self._determine_strategy(query_analysis)

        logger.info(f"🎯 Using strategy: {strategy.value}")

        try:
            # 1. 초기 벡터 검색
            initial_matches = self._initial_vector_search(query, query_analysis, config)

            # 2. 그래프 확장
            expanded_nodes, expansion_path = self._expand_from_initial_matches(
                initial_matches, strategy, query_analysis, config
            )

            # 3. 서브그래프 구성
            subgraph_nodes, subgraph_edges = self._build_subgraph(
                expanded_nodes, config
            )

            # 4. 관련성 스코어링
            relevance_scores = self._calculate_relevance_scores(
                subgraph_nodes, query, initial_matches, config
            )

            # 5. 신뢰도 계산
            confidence_score = self._calculate_confidence_score(
                initial_matches, len(subgraph_nodes), relevance_scores
            )

            # 6. 통계 생성
            nodes_by_type = defaultdict(int)
            for node_data in subgraph_nodes.values():
                node_type = node_data.get("node_type", "unknown")
                nodes_by_type[node_type] += 1

            extraction_time = time.time() - start_time

            # 결과 생성
            result = SubgraphResult(
                nodes=subgraph_nodes,
                edges=subgraph_edges,
                query=query,
                query_analysis=query_analysis,
                extraction_strategy=strategy,
                total_nodes=len(subgraph_nodes),
                total_edges=len(subgraph_edges),
                nodes_by_type=dict(nodes_by_type),
                extraction_time=extraction_time,
                initial_matches=initial_matches,
                expansion_path=expansion_path,
                relevance_scores=relevance_scores,
                confidence_score=confidence_score,
            )

            # 캐시 저장
            if cache_key:
                self.query_cache[cache_key] = result

            logger.info(
                f"✅ Subgraph extracted: {result.total_nodes} nodes, {result.total_edges} edges"
            )
            logger.info(f"   ⏱️ Time: {extraction_time:.2f}s")
            logger.info(f"   🎯 Confidence: {confidence_score:.3f}")

            return result

        except Exception as e:
            logger.error(f"❌ Subgraph extraction failed: {e}")
            raise

    def _determine_strategy(
        self, query_analysis: Optional[QueryAnalysisResult]
    ) -> SearchStrategy:
        """쿼리 분석 결과로부터 최적 전략 결정"""

        if not query_analysis:
            return SearchStrategy.HYBRID

        # 복잡도 기반 전략
        if query_analysis.complexity in [QueryComplexity.SIMPLE]:
            return SearchStrategy.LOCAL
        elif query_analysis.complexity == QueryComplexity.EXPLORATORY:
            return SearchStrategy.GLOBAL

        # 쿼리 타입 기반 전략
        if query_analysis.query_type == QueryType.CITATION_ANALYSIS:
            return SearchStrategy.LOCAL
        elif query_analysis.query_type == QueryType.COLLABORATION_ANALYSIS:
            return SearchStrategy.COMMUNITY
        elif query_analysis.query_type == QueryType.TREND_ANALYSIS:
            return SearchStrategy.TEMPORAL
        elif query_analysis.query_type == QueryType.SIMILARITY_ANALYSIS:
            return SearchStrategy.SEMANTIC
        elif query_analysis.query_type == QueryType.COMPREHENSIVE_ANALYSIS:
            return SearchStrategy.GLOBAL

        # 검색 모드 기반 전략
        if query_analysis.search_mode == SearchMode.LOCAL:
            return SearchStrategy.LOCAL
        elif query_analysis.search_mode == SearchMode.GLOBAL:
            return SearchStrategy.GLOBAL
        else:
            return SearchStrategy.HYBRID

    def _initial_vector_search(
        self,
        query: str,
        query_analysis: Optional[QueryAnalysisResult],
        config: ExtractionConfig,
    ) -> List[SearchResult]:
        """초기 벡터 검색"""

        logger.info(f"🔍 Initial vector search (top_k={config.initial_top_k})...")

        # 쿼리 임베딩 생성
        if not self.embedding_model.is_loaded():
            self.embedding_model.load_model()

        query_embedding = self.embedding_model.encode([query])[0]

        # 노드 타입 필터링 (필요시)
        node_types = None
        if query_analysis and query_analysis.required_node_types:
            node_types = [nt.value for nt in query_analysis.required_node_types]

        # 벡터 검색
        search_results = self.vector_store.search_similar_nodes(
            query_embedding=query_embedding,
            top_k=config.initial_top_k,
            node_types=node_types,
        )

        # 유사도 임계값 필터링
        filtered_results = [
            result
            for result in search_results
            if result.similarity_score >= config.similarity_threshold
        ]

        logger.info(
            f"✅ Found {len(filtered_results)} initial matches (threshold: {config.similarity_threshold})"
        )

        return filtered_results

    def _expand_from_initial_matches(
        self,
        initial_matches: List[SearchResult],
        strategy: SearchStrategy,
        query_analysis: Optional[QueryAnalysisResult],
        config: ExtractionConfig,
    ) -> Tuple[Set[str], List[Dict[str, Any]]]:
        """초기 매치로부터 그래프 확장"""

        logger.info(f"🚀 Expanding graph using {strategy.value} strategy...")

        # 초기 노드들
        seed_nodes = {result.node_id for result in initial_matches}
        expansion_path = []

        # 전략별 확장
        if strategy == SearchStrategy.LOCAL:
            expanded_nodes = self._local_expansion(seed_nodes, config, expansion_path)
        elif strategy == SearchStrategy.GLOBAL:
            expanded_nodes = self._global_expansion(seed_nodes, config, expansion_path)
        elif strategy == SearchStrategy.COMMUNITY:
            expanded_nodes = self._community_expansion(
                seed_nodes, config, expansion_path
            )
        elif strategy == SearchStrategy.TEMPORAL:
            expanded_nodes = self._temporal_expansion(
                seed_nodes, query_analysis, config, expansion_path
            )
        elif strategy == SearchStrategy.SEMANTIC:
            expanded_nodes = self._semantic_expansion(
                seed_nodes, config, expansion_path
            )
        else:  # HYBRID
            expanded_nodes = self._hybrid_expansion(
                seed_nodes, query_analysis, config, expansion_path
            )

        logger.info(f"✅ Expanded to {len(expanded_nodes)} nodes")

        return expanded_nodes, expansion_path

    def _local_expansion(
        self,
        seed_nodes: Set[str],
        config: ExtractionConfig,
        expansion_path: List[Dict[str, Any]],
    ) -> Set[str]:
        """지역적 확장 (BFS)"""

        expanded = set(seed_nodes)
        current_level = seed_nodes

        for hop in range(config.max_hops):
            if len(expanded) >= config.max_nodes:
                break

            next_level = set()

            for node_id in current_level:
                if len(expanded) >= config.max_nodes:
                    break

                # 이웃 노드들 찾기
                neighbors = self._get_node_neighbors(node_id)

                for neighbor_id, edge_data in neighbors:
                    if neighbor_id not in expanded:
                        next_level.add(neighbor_id)

                        if len(expanded) + len(next_level) >= config.max_nodes:
                            break

            # 다음 레벨 추가
            expanded.update(next_level)
            current_level = next_level

            expansion_path.append(
                {
                    "hop": hop + 1,
                    "added_nodes": len(next_level),
                    "total_nodes": len(expanded),
                    "strategy": "local_bfs",
                }
            )

            if not next_level:  # 더 이상 확장할 노드가 없음
                break

        return expanded

    def _global_expansion(
        self,
        seed_nodes: Set[str],
        config: ExtractionConfig,
        expansion_path: List[Dict[str, Any]],
    ) -> Set[str]:
        """전역적 확장 (중심성 기반)"""

        # 중심성 계산 (미리 계산되어 있지 않다면)
        if not hasattr(self, "_centrality_scores"):
            logger.info("📊 Computing centrality scores...")

            # 효율을 위해 Degree Centrality 사용
            self._centrality_scores = nx.degree_centrality(
                self.networkx_graph.to_undirected()
            )

        # 높은 중심성을 가진 노드들 우선 선택
        all_nodes = list(self.networkx_graph.nodes())

        # 중심성 점수로 정렬
        sorted_nodes = sorted(
            all_nodes, key=lambda x: self._centrality_scores.get(x, 0), reverse=True
        )

        # 시드 노드들과 연결성이 있는 고중심성 노드들 선택
        expanded = set(seed_nodes)

        for node_id in sorted_nodes:
            if len(expanded) >= config.max_nodes:
                break

            if node_id in expanded:
                continue

            # 기존 노드들과의 연결성 확인
            has_connection = False
            for existing_node in expanded:
                if self.networkx_graph.has_edge(
                    node_id, existing_node
                ) or self.networkx_graph.has_edge(existing_node, node_id):
                    has_connection = True
                    break

            # 직접 연결이 없어도 중심성이 높으면 포함 (Global 전략)
            centrality = self._centrality_scores.get(node_id, 0)
            if has_connection or centrality > 0.1:  # 임계값
                expanded.add(node_id)

        expansion_path.append(
            {
                "strategy": "global_centrality",
                "total_nodes": len(expanded),
                "centrality_threshold": 0.1,
            }
        )

        return expanded

    def _community_expansion(
        self,
        seed_nodes: Set[str],
        config: ExtractionConfig,
        expansion_path: List[Dict[str, Any]],
    ) -> Set[str]:
        """커뮤니티 기반 확장"""

        # 커뮤니티 감지 (미리 계산되어 있지 않다면)
        if not hasattr(self, "_communities"):
            logger.info("🏘️ Detecting communities...")

            # Louvain 커뮤니티 감지
            undirected_graph = self.networkx_graph.to_undirected()
            try:
                import community as community_louvain

                self._communities = community_louvain.best_partition(undirected_graph)
            except ImportError:
                # 대안: 간단한 연결 성분 사용
                components = list(nx.connected_components(undirected_graph))
                self._communities = {}
                for i, component in enumerate(components):
                    for node in component:
                        self._communities[node] = i

        # 시드 노드들이 속한 커뮤니티들 찾기
        seed_communities = set()
        for node_id in seed_nodes:
            if node_id in self._communities:
                seed_communities.add(self._communities[node_id])

        # 해당 커뮤니티의 모든 노드들 추가
        expanded = set(seed_nodes)

        for node_id, community_id in self._communities.items():
            if len(expanded) >= config.max_nodes:
                break

            if community_id in seed_communities:
                expanded.add(node_id)

        expansion_path.append(
            {
                "strategy": "community_based",
                "seed_communities": list(seed_communities),
                "total_nodes": len(expanded),
            }
        )

        return expanded

    def _temporal_expansion(
        self,
        seed_nodes: Set[str],
        query_analysis: Optional[QueryAnalysisResult],
        config: ExtractionConfig,
        expansion_path: List[Dict[str, Any]],
    ) -> Set[str]:
        """시간적 확장 (년도 기반)"""

        # 시드 노드들의 년도 분포 파악
        seed_years = []
        for node_id in seed_nodes:
            node_data = self.networkx_graph.nodes.get(node_id, {})
            year = node_data.get("year", "")
            if year and str(year).isdigit():
                seed_years.append(int(year))

        if not seed_years:
            # 년도 정보가 없으면 지역 확장으로 대체
            return self._local_expansion(seed_nodes, config, expansion_path)

        # 년도 범위 결정
        min_year, max_year = min(seed_years), max(seed_years)
        year_window = max(3, max_year - min_year + 2)  # 최소 3년 윈도우

        # 시간적 근접성 기반 노드 선택
        expanded = set(seed_nodes)

        for node_id in self.networkx_graph.nodes():
            if len(expanded) >= config.max_nodes:
                break

            if node_id in expanded:
                continue

            node_data = self.networkx_graph.nodes[node_id]
            year = node_data.get("year", "")

            if year and str(year).isdigit():
                node_year = int(year)

                # 시간 윈도우 내의 노드들 포함
                if min_year - year_window <= node_year <= max_year + year_window:
                    expanded.add(node_id)

        expansion_path.append(
            {
                "strategy": "temporal",
                "year_range": [min_year, max_year],
                "year_window": year_window,
                "total_nodes": len(expanded),
            }
        )

        return expanded

    def _semantic_expansion(
        self,
        seed_nodes: Set[str],
        config: ExtractionConfig,
        expansion_path: List[Dict[str, Any]],
    ) -> Set[str]:
        """의미적 확장 (임베딩 유사도 기반)"""

        # 시드 노드들의 평균 임베딩 계산
        seed_embeddings = []
        for node_id in seed_nodes:
            embedding = self.vector_store.get_node_embedding(node_id)
            if embedding is not None:
                seed_embeddings.append(embedding)

        if not seed_embeddings:
            # 임베딩이 없으면 지역 확장으로 대체
            return self._local_expansion(seed_nodes, config, expansion_path)

        # 평균 임베딩
        avg_embedding = np.mean(seed_embeddings, axis=0)

        # 유사한 노드들 검색
        similar_results = self.vector_store.search_similar_nodes(
            query_embedding=avg_embedding, top_k=config.max_nodes, node_types=None
        )

        # 유사도 임계값 적용
        expanded = set(seed_nodes)
        for result in similar_results:
            if len(expanded) >= config.max_nodes:
                break

            if (
                result.similarity_score >= config.similarity_threshold
                and result.node_id not in expanded
            ):
                expanded.add(result.node_id)

        expansion_path.append(
            {
                "strategy": "semantic",
                "similarity_threshold": config.similarity_threshold,
                "total_nodes": len(expanded),
            }
        )

        return expanded

    def _hybrid_expansion(
        self,
        seed_nodes: Set[str],
        query_analysis: Optional[QueryAnalysisResult],
        config: ExtractionConfig,
        expansion_path: List[Dict[str, Any]],
    ) -> Set[str]:
        """하이브리드 확장 (여러 전략 조합)"""

        # 각 전략으로 일부씩 확장
        total_budget = config.max_nodes
        current_expanded = set(seed_nodes)

        # 전략별 예산 할당
        strategies = [
            (SearchStrategy.LOCAL, 0.4),
            (SearchStrategy.SEMANTIC, 0.3),
            (SearchStrategy.GLOBAL, 0.3),
        ]

        for strategy, weight in strategies:
            if len(current_expanded) >= total_budget:
                break

            # 해당 전략의 예산
            strategy_budget = int((total_budget - len(seed_nodes)) * weight)
            if strategy_budget < 1:
                continue

            # 임시 설정으로 해당 전략 실행
            temp_config = ExtractionConfig()
            temp_config.max_nodes = len(current_expanded) + strategy_budget
            temp_config.max_hops = config.max_hops
            temp_config.similarity_threshold = config.similarity_threshold

            temp_expansion_path = []

            if strategy == SearchStrategy.LOCAL:
                strategy_result = self._local_expansion(
                    current_expanded, temp_config, temp_expansion_path
                )
            elif strategy == SearchStrategy.SEMANTIC:
                strategy_result = self._semantic_expansion(
                    current_expanded, temp_config, temp_expansion_path
                )
            elif strategy == SearchStrategy.GLOBAL:
                strategy_result = self._global_expansion(
                    current_expanded, temp_config, temp_expansion_path
                )
            else:
                strategy_result = current_expanded

            # 새로 추가된 노드들만 선택
            new_nodes = strategy_result - current_expanded

            # 예산 내에서만 추가
            if len(new_nodes) > strategy_budget:
                new_nodes = set(list(new_nodes)[:strategy_budget])

            current_expanded.update(new_nodes)

            expansion_path.append(
                {
                    "strategy": f"hybrid_{strategy.value}",
                    "budget": strategy_budget,
                    "added_nodes": len(new_nodes),
                    "total_nodes": len(current_expanded),
                }
            )

        return current_expanded

    def _get_node_neighbors(self, node_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        """노드의 이웃들 조회 (캐싱 지원)"""

        if node_id in self.node_neighbors_cache:
            return self.node_neighbors_cache[node_id]

        neighbors = []

        # 나가는 엣지
        for neighbor in self.networkx_graph.successors(node_id):
            edge_data = self.networkx_graph.get_edge_data(node_id, neighbor)
            if edge_data:
                # Multiple edges인 경우 첫 번째 엣지 사용
                first_edge = next(iter(edge_data.values()))
                neighbors.append((neighbor, first_edge))

        # 들어오는 엣지
        for neighbor in self.networkx_graph.predecessors(node_id):
            edge_data = self.networkx_graph.get_edge_data(neighbor, node_id)
            if edge_data:
                first_edge = next(iter(edge_data.values()))
                neighbors.append((neighbor, first_edge))

        # 캐시 저장
        if self.config.enable_caching:
            self.node_neighbors_cache[node_id] = neighbors

        return neighbors

    def _build_subgraph(
        self, node_ids: Set[str], config: ExtractionConfig
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        """선택된 노드들로 서브그래프 구성"""

        logger.info(f"🔨 Building subgraph from {len(node_ids)} nodes...")

        # 노드 데이터 수집
        subgraph_nodes = {}
        for node_id in node_ids:
            if node_id in self.networkx_graph:
                node_data = dict(self.networkx_graph.nodes[node_id])
                node_data["id"] = node_id  # ID 추가
                subgraph_nodes[node_id] = node_data

        # 엣지 데이터 수집
        subgraph_edges = []
        edge_count = 0

        for source in node_ids:
            if edge_count >= config.max_edges:
                break

            for target in node_ids:
                if source == target or edge_count >= config.max_edges:
                    continue

                # 엣지 존재 확인
                if self.networkx_graph.has_edge(source, target):
                    edge_data = self.networkx_graph.get_edge_data(source, target)

                    # Multiple edges 처리
                    for key, attrs in edge_data.items():
                        if edge_count >= config.max_edges:
                            break

                        edge_dict = dict(attrs)
                        edge_dict.update(
                            {"source": source, "target": target, "key": key}
                        )
                        subgraph_edges.append(edge_dict)
                        edge_count += 1

        logger.info(
            f"✅ Subgraph built: {len(subgraph_nodes)} nodes, {len(subgraph_edges)} edges"
        )

        return subgraph_nodes, subgraph_edges

    def _calculate_relevance_scores(
        self,
        nodes: Dict[str, Dict[str, Any]],
        query: str,
        initial_matches: List[SearchResult],
        config: ExtractionConfig,
    ) -> Dict[str, float]:
        """노드별 관련성 점수 계산"""

        relevance_scores = {}

        # 초기 매치들의 점수
        initial_scores = {
            result.node_id: result.similarity_score for result in initial_matches
        }

        for node_id, node_data in nodes.items():
            score = 0.0

            # 1. 벡터 유사도 점수
            if node_id in initial_scores:
                score += initial_scores[node_id] * config.vector_weight

            # 2. 노드 타입 가중치
            node_type = node_data.get("node_type", "unknown")
            type_weight = config.node_type_weights.get(node_type, 0.5)
            score += type_weight * config.metadata_weight

            # 3. 그래프 중심성 (있다면)
            if hasattr(self, "_centrality_scores"):
                centrality = self._centrality_scores.get(node_id, 0)
                score += centrality * config.graph_weight

            # 정규화
            relevance_scores[node_id] = min(1.0, score)

        return relevance_scores

    def _calculate_confidence_score(
        self,
        initial_matches: List[SearchResult],
        total_nodes: int,
        relevance_scores: Dict[str, float],
    ) -> float:
        """전체 신뢰도 점수 계산"""

        if not initial_matches or not relevance_scores:
            return 0.0

        # 초기 매치들의 평균 유사도
        avg_similarity = np.mean([m.similarity_score for m in initial_matches])

        # 관련성 점수들의 평균
        avg_relevance = np.mean(list(relevance_scores.values()))

        # 노드 수 대비 적절성 (너무 많거나 적으면 신뢰도 감소)
        size_factor = 1.0
        if total_nodes < 5:
            size_factor = 0.7  # 너무 적음
        elif total_nodes > 100:
            size_factor = 0.8  # 너무 많음

        # 종합 신뢰도
        confidence = avg_similarity * 0.4 + avg_relevance * 0.4 + size_factor * 0.2

        return min(1.0, confidence)


def main():
    """테스트 실행"""
    # 기본 import 경로 설정
    import sys
    from pathlib import Path

    # src 디렉토리를 Python path에 추가
    src_dir = Path(__file__).parent.parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        from src import GRAPHS_DIR

        print("🧪 Testing SubgraphExtractor...")

        # 파일 경로 확인
        unified_graph_file = GRAPHS_DIR / "unified" / "unified_knowledge_graph.json"
        vector_store_dir = GRAPHS_DIR / "embeddings"

        if not unified_graph_file.exists():
            print(f"❌ Unified graph not found: {unified_graph_file}")
            return

        if not vector_store_dir.exists():
            print(f"❌ Vector store not found: {vector_store_dir}")
            return

        # SubgraphExtractor 초기화
        extractor = SubgraphExtractor(
            unified_graph_path=str(unified_graph_file),
            vector_store_path=str(vector_store_dir),
            embedding_model="auto",
        )

        # 테스트 쿼리들
        test_queries = [
            "배터리 SoC 예측에 사용된 머신러닝 기법들은?",
            "김철수 교수의 연구 네트워크",
            "전기차 충전 관련 최신 연구 동향",
        ]

        print(f"\n🔍 Testing subgraph extraction...")

        for i, query in enumerate(test_queries[:1]):  # 첫 번째만 테스트
            print(f"\n📝 Query {i+1}: {query}")

            try:
                # 서브그래프 추출
                result = extractor.extract_subgraph(
                    query=query, strategy=SearchStrategy.HYBRID
                )

                print(f"✅ Extraction successful:")
                print(f"   📄 Nodes: {result.total_nodes}")
                print(f"   🔗 Edges: {result.total_edges}")
                print(f"   📊 Node types: {result.nodes_by_type}")
                print(f"   🎯 Confidence: {result.confidence_score:.3f}")
                print(f"   ⏱️ Time: {result.extraction_time:.2f}s")

                # 상위 관련 노드들
                top_relevant = sorted(
                    result.relevance_scores.items(), key=lambda x: x[1], reverse=True
                )[:5]

                print(f"   🏆 Top relevant nodes:")
                for node_id, score in top_relevant:
                    node_type = result.nodes[node_id].get("node_type", "unknown")
                    print(f"      {node_id} ({node_type}): {score:.3f}")

            except Exception as e:
                print(f"❌ Query {i+1} failed: {e}")
                import traceback

                traceback.print_exc()

        print(f"\n✅ SubgraphExtractor test completed!")

    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
