"""
GraphRAG LangChain 커스텀 리트리버
Custom GraphRAG Retriever for LangChain Integration

LangChain BaseRetriever를 상속받아 GraphRAG 시스템과 완전 통합
- 쿼리 분석 → 서브그래프 추출 → 컨텍스트 직렬화 파이프라인
- LangChain Document 형태로 결과 반환
- 비동기 처리 지원
- 메타데이터 및 소스 추적
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from pydantic import Field
import warnings

# LangChain imports
try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import (
        CallbackManagerForRetrieverRun,
        AsyncCallbackManagerForRetrieverRun,
    )

    _langchain_available = True
except ImportError:
    _langchain_available = False
    warnings.warn(
        "LangChain not available. Install with: pip install langchain langchain-core"
    )

    # Placeholder classes for type hints
    class BaseRetriever:
        def __init__(self, *args, **kwargs):
            raise ImportError("LangChain is required but not installed")

    class Document:
        def __init__(self, *args, **kwargs):
            pass

    class CallbackManagerForRetrieverRun:
        pass

    class AsyncCallbackManagerForRetrieverRun:
        pass


# GraphRAG imports
try:
    from ..query_analyzer import QueryAnalyzer, QueryAnalysisResult
    from ..embeddings.subgraph_extractor import SubgraphExtractor, SubgraphResult
    from ..embeddings.context_serializer import ContextSerializer, SerializedContext
    from ..embeddings.vector_store_manager import VectorStoreManager
except ImportError as e:
    warnings.warn(f"Some GraphRAG components not available: {e}")

# 로깅 설정
logger = logging.getLogger(__name__)


class GraphRAGRetriever(BaseRetriever):
    """GraphRAG 커스텀 LangChain 리트리버

    LangChain의 BaseRetriever를 상속받아 GraphRAG 시스템을
    LangChain 체인에서 직접 사용할 수 있도록 하는 어댑터 클래스
    """

    # Pydantic 필드 정의 (LangChain v0.1+ 호환)
    unified_graph_path: str = Field(description="통합 그래프 파일 경로")
    vector_store_path: str = Field(description="벡터 저장소 경로")
    embedding_model: str = Field(default="auto", description="임베딩 모델명")

    # 구성요소 설정
    max_docs: int = Field(default=10, description="반환할 최대 문서 수")
    min_relevance_score: float = Field(default=0.3, description="최소 관련성 점수")
    enable_query_analysis: bool = Field(default=True, description="쿼리 분석 활성화")

    # 내부 컴포넌트들 (초기화 후 설정)
    query_analyzer: Optional[QueryAnalyzer] = Field(default=None, exclude=True)
    subgraph_extractor: Optional[SubgraphExtractor] = Field(default=None, exclude=True)
    context_serializer: Optional[ContextSerializer] = Field(default=None, exclude=True)

    # 캐시 및 성능 설정
    enable_caching: bool = Field(default=True, description="결과 캐싱 활성화")
    cache_ttl_seconds: int = Field(default=3600, description="캐시 유지 시간(초)")

    # 내부 캐시 저장소
    query_cache: Dict[str, Document] = Field(default_factory=dict, exclude=True)
    cache_timestamps: Dict[str, float] = Field(default_factory=dict, exclude=True)

    class Config:
        """Pydantic 설정"""

        arbitrary_types_allowed = True
        exclude = {
            "query_analyzer",
            "subgraph_extractor",
            "context_serializer",
            "query_cache",
            "cache_timestamps",
        }

    def __init__(self, **kwargs):
        """GraphRAGRetriever 초기화"""
        super().__init__(**kwargs)

        if not _langchain_available:
            raise ImportError(
                "LangChain is required for GraphRAGRetriever.\n"
                "Install with: pip install langchain langchain-core"
            )

        # 컴포넌트 초기화 플래그
        self._initialized = False

        logger.info("✅ GraphRAGRetriever created")
        logger.info(f"   📁 Graph: {self.unified_graph_path}")
        logger.info(f"   🗄️ Vector Store: {self.vector_store_path}")
        logger.info(f"   🤖 Model: {self.embedding_model}")

    def _lazy_init(self) -> None:
        """지연 초기화 - 첫 번째 쿼리 시 실행"""
        if self._initialized:
            return

        logger.info("🔧 Initializing GraphRAG components...")

        try:
            # 1. Query Analyzer 초기화
            if self.enable_query_analysis:
                self.query_analyzer = QueryAnalyzer()
                logger.info("✅ QueryAnalyzer initialized")

            # 2. SubgraphExtractor 초기화
            self.subgraph_extractor = SubgraphExtractor(
                unified_graph_path=self.unified_graph_path,
                vector_store_path=self.vector_store_path,
                embedding_model=self.embedding_model,
            )
            logger.info("✅ SubgraphExtractor initialized")

            # 3. ContextSerializer 초기화
            self.context_serializer = ContextSerializer()
            logger.info("✅ ContextSerializer initialized")

            self._initialized = True
            logger.info("🚀 GraphRAG components ready!")

        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise

    def _get_docs(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """동기 문서 검색 (LangChain BaseRetriever 인터페이스)"""

        # 지연 초기화
        self._lazy_init()

        # 캐시 확인
        if self.enable_caching:
            cached_result = self._get_from_cache(query)
            if cached_result:
                logger.info("✅ Using cached result")
                return cached_result

        try:
            logger.info(f"🔍 Processing query: '{query[:50]}...'")

            # 1. 쿼리 분석 (선택적)
            query_analysis = None
            if self.enable_query_analysis and self.query_analyzer:
                logger.info("📊 Analyzing query...")
                query_analysis = self.query_analyzer.analyze(query)
                logger.info(f"   Complexity: {query_analysis.complexity.value}")
                logger.info(f"   Type: {query_analysis.query_type.value}")

            # 2. 서브그래프 추출
            logger.info("🔍 Extracting relevant subgraph...")
            subgraph_result = self.subgraph_extractor.extract_subgraph(
                query=query, query_analysis=query_analysis
            )

            # 3. 컨텍스트 직렬화
            logger.info("📝 Serializing context...")
            serialized_context = self.context_serializer.serialize(
                subgraph_result=subgraph_result, query_analysis=query_analysis
            )

            # 4. LangChain Document 형태로 변환
            documents = self._convert_to_langchain_documents(
                serialized_context, subgraph_result, query_analysis
            )

            # 5. 관련성 점수로 필터링 및 정렬
            filtered_docs = self._filter_and_rank_documents(documents)

            # 6. 캐시 저장
            if self.enable_caching:
                self._save_to_cache(query, filtered_docs)

            logger.info(f"✅ Retrieved {len(filtered_docs)} documents")
            return filtered_docs

        except Exception as e:
            logger.error(f"❌ Document retrieval failed: {e}")
            # 에러 시 빈 결과 반환 (체인이 중단되지 않도록)
            return []

    async def _aget_docs(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """비동기 문서 검색"""

        # 동기 메서드를 비동기로 래핑
        loop = asyncio.get_event_loop()

        # CallbackManager 타입 변환 (동기 버전으로)
        sync_run_manager = None  # 필요시 구현

        return await loop.run_in_executor(None, self._get_docs, query, sync_run_manager)

    def _convert_to_langchain_documents(
        self,
        serialized_context: SerializedContext,
        subgraph_result: SubgraphResult,
        query_analysis: Optional[QueryAnalysisResult],
    ) -> List[Document]:
        """SerializedContext를 LangChain Document 리스트로 변환"""

        documents = []

        # 1. 메인 컨텍스트 문서
        main_doc = Document(
            page_content=serialized_context.main_text,
            metadata={
                "source": "graphrag_main_context",
                "query": serialized_context.query,
                "total_nodes": serialized_context.included_nodes,
                "total_edges": serialized_context.included_edges,
                "confidence_score": serialized_context.confidence_score,
                "language": serialized_context.language,
                "extraction_strategy": subgraph_result.extraction_strategy.value,
                "processing_time": subgraph_result.extraction_time,
                "document_type": "main_context",
            },
        )
        documents.append(main_doc)

        # 2. 개별 노드 문서들 (상위 관련성만)
        top_relevant_nodes = sorted(
            subgraph_result.relevance_scores.items(), key=lambda x: x[1], reverse=True
        )[: self.max_docs]

        for node_id, relevance_score in top_relevant_nodes:
            if relevance_score < self.min_relevance_score:
                continue

            node_data = subgraph_result.nodes.get(node_id, {})
            node_type = node_data.get("node_type", "unknown")

            # 노드별 텍스트 생성
            content = self._generate_node_content(node_data)

            if content:
                node_doc = Document(
                    page_content=content,
                    metadata={
                        "source": f"graphrag_node_{node_id}",
                        "node_id": node_id,
                        "node_type": node_type,
                        "relevance_score": relevance_score,
                        "document_type": "individual_node",
                        "query": serialized_context.query,
                    },
                )
                documents.append(node_doc)

        return documents

    def _generate_node_content(self, node_data: Dict[str, Any]) -> str:
        """개별 노드용 컨텐츠 생성"""
        node_type = node_data.get("node_type", "unknown")
        node_id = node_data.get("id", "unknown")

        if node_type == "paper":
            title = node_data.get("title", "Unknown Title")
            abstract = node_data.get("abstract", "")
            authors = node_data.get("authors", [])
            year = node_data.get("year", "")

            content = f"Paper: {title}"
            if year:
                content += f" ({year})"
            if authors:
                author_list = authors if isinstance(authors, list) else [authors]
                content += f"\nAuthors: {', '.join(author_list[:3])}"
            if abstract:
                content += f"\nAbstract: {abstract[:300]}..."

        elif node_type == "author":
            name = node_data.get("name", node_id)
            paper_count = node_data.get("paper_count", 0)
            top_keywords = node_data.get("top_keywords", [])

            content = f"Author: {name}"
            if paper_count:
                content += f"\nPublications: {paper_count} papers"
            if top_keywords:
                if isinstance(top_keywords[0], (list, tuple)):
                    keywords = [kw[0] for kw in top_keywords[:5]]
                else:
                    keywords = top_keywords[:5]
                content += f"\nResearch Areas: {', '.join(keywords)}"

        elif node_type == "keyword":
            keyword = node_data.get("name", node_id)
            frequency = node_data.get("frequency", 0)

            content = f"Keyword: {keyword}"
            if frequency:
                content += f"\nFrequency: {frequency} papers"

        elif node_type == "journal":
            name = node_data.get("name", node_id)
            paper_count = node_data.get("paper_count", 0)

            content = f"Journal: {name}"
            if paper_count:
                content += f"\nPublications: {paper_count} papers"
        else:
            content = f"{node_type}: {node_id}"

        return content

    def _filter_and_rank_documents(self, documents: List[Document]) -> List[Document]:
        """문서 필터링 및 랭킹"""

        # 관련성 점수로 필터링
        filtered_docs = []
        for doc in documents:
            relevance_score = doc.metadata.get("relevance_score", 1.0)

            # 메인 컨텍스트는 항상 포함
            if doc.metadata.get("document_type") == "main_context":
                filtered_docs.append(doc)
            # 개별 노드는 임계값 적용
            elif relevance_score >= self.min_relevance_score:
                filtered_docs.append(doc)

        # 관련성 점수로 정렬 (메인 컨텍스트는 맨 앞)
        def sort_key(doc):
            doc_type = doc.metadata.get("document_type", "unknown")
            relevance = doc.metadata.get("relevance_score", 0.0)

            if doc_type == "main_context":
                return (2, relevance)  # 높은 우선순위
            else:
                return (1, relevance)  # 관련성 점수 순

        filtered_docs.sort(key=sort_key, reverse=True)

        # 최대 문서 수 제한
        return filtered_docs[: self.max_docs]

    def _get_cache_key(self, query: str) -> str:
        """캐시 키 생성"""
        import hashlib

        key_data = f"{query}_{self.max_docs}_{self.min_relevance_score}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_from_cache(self, query: str) -> Optional[List[Document]]:
        """캐시에서 결과 조회"""
        if not self.enable_caching:
            return None

        cache_key = self._get_cache_key(query)

        if cache_key in self.query_cache:
            # TTL 확인
            import time

            cache_time = self.cache_timestamps.get(cache_key, 0)

            if time.time() - cache_time < self.cache_ttl_seconds:
                return self.query_cache[cache_key]
            else:
                # 만료된 캐시 제거
                del self.query_cache[cache_key]
                del self.cache_timestamps[cache_key]

        return None

    def _save_to_cache(self, query: str, documents: List[Document]) -> None:
        """캐시에 결과 저장"""
        if not self.enable_caching:
            return

        cache_key = self._get_cache_key(query)

        import time

        self.query_cache[cache_key] = documents
        self.cache_timestamps[cache_key] = time.time()

        # 캐시 크기 제한 (최대 100개)
        if len(self.query_cache) > 100:
            # 가장 오래된 캐시 제거
            oldest_key = min(self.cache_timestamps, key=self.cache_timestamps.get)
            del self.query_cache[oldest_key]
            del self.cache_timestamps[oldest_key]

    def clear_cache(self) -> None:
        """캐시 초기화"""
        self.query_cache.clear()
        self.cache_timestamps.clear()
        logger.info("🗑️ Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        return {
            "cached_queries": len(self.query_cache),
            "cache_size_mb": sum(len(str(docs)) for docs in self.query_cache.values())
            / 1024
            / 1024,
            "cache_hit_ratio": getattr(self, "_cache_hits", 0)
            / max(1, getattr(self, "_total_queries", 1)),
        }

    def update_config(self, **kwargs) -> None:
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"📝 Updated {key} = {value}")

        # 컴포넌트 재초기화가 필요한 경우
        if any(
            key in kwargs
            for key in ["unified_graph_path", "vector_store_path", "embedding_model"]
        ):
            self._initialized = False
            logger.info("🔄 Components will be reinitialized on next query")


def create_graphrag_retriever(
    unified_graph_path: str,
    vector_store_path: str,
    embedding_model: str = "auto",
    **kwargs,
) -> GraphRAGRetriever:
    """GraphRAGRetriever 팩토리 함수

    Args:
        unified_graph_path: 통합 그래프 파일 경로
        vector_store_path: 벡터 저장소 경로
        embedding_model: 임베딩 모델명
        **kwargs: 추가 설정

    Returns:
        GraphRAGRetriever 인스턴스
    """
    return GraphRAGRetriever(
        unified_graph_path=unified_graph_path,
        vector_store_path=vector_store_path,
        embedding_model=embedding_model,
        **kwargs,
    )


# 사용 예시
def main():
    """GraphRAGRetriever 사용 예시"""

    if not _langchain_available:
        print("❌ LangChain not available for testing")
        return

    print("🧪 Testing GraphRAGRetriever...")

    # 기본 경로 설정
    from pathlib import Path

    base_dir = Path(__file__).parent.parent.parent.parent

    try:
        # GraphRAGRetriever 생성
        retriever = create_graphrag_retriever(
            unified_graph_path=str(
                base_dir / "graphs" / "unified" / "unified_knowledge_graph.json"
            ),
            vector_store_path=str(base_dir / "graphs" / "embeddings"),
            embedding_model="auto",
            max_docs=5,
            min_relevance_score=0.3,
        )

        # 테스트 쿼리
        test_queries = [
            "배터리 SoC 예측에 사용된 머신러닝 기법들은?",
            "김철수 교수의 연구 분야는?",
            "전기차 충전 관련 최신 연구 동향은?",
        ]

        for query in test_queries[:1]:  # 첫 번째만 테스트
            print(f"\n📝 Query: {query}")

            try:
                # 문서 검색
                documents = retriever._get_docs(query, run_manager=None)

                print(f"✅ Retrieved {len(documents)} documents:")
                for i, doc in enumerate(documents):
                    doc_type = doc.metadata.get("document_type", "unknown")
                    relevance = doc.metadata.get("relevance_score", "N/A")
                    print(f"   📄 Doc {i+1} ({doc_type}): relevance={relevance}")
                    print(f"      Content: {doc.page_content[:100]}...")

                # 캐시 통계
                cache_stats = retriever.get_cache_stats()
                print(f"📊 Cache stats: {cache_stats}")

            except Exception as e:
                print(f"❌ Query failed: {e}")

        print(f"\n✅ GraphRAGRetriever test completed!")

    except Exception as e:
        print(f"❌ Test setup failed: {e}")


if __name__ == "__main__":
    main()
