"""
간단한 메모리 벡터 저장소
Simple Memory Vector Store

FAISS/ChromaDB 호환성 문제 해결을 위한 임시 솔루션
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SearchResult:
    """검색 결과 클래스"""

    node_id: str
    node_type: str
    similarity_score: float
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "similarity_score": self.similarity_score,
            "text": self.text,
            "metadata": self.metadata,
        }


class SimpleVectorStore:
    """간단한 메모리 기반 벡터 저장소"""

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Args:
            persist_directory: 저장 디렉토리
        """
        self.persist_directory = Path(persist_directory) if persist_directory else None

        # 메모리 저장소
        self.embeddings = []  # List[np.ndarray]
        self.node_ids = []  # List[str]
        self.node_types = []  # List[str]
        self.texts = []  # List[str]
        self.metadatas = []  # List[Dict]

        self.is_initialized = False
        self.dimension = None
        self.total_vectors = 0

    def initialize(self, dimension: int) -> None:
        """벡터 저장소 초기화"""
        self.dimension = dimension
        self.is_initialized = True

        # 기존 데이터 로드 시도
        self.load()

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        node_ids: List[str],
        node_types: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """임베딩 추가"""
        if not self.is_initialized:
            raise ValueError("Vector store not initialized")

        # 데이터 추가
        for i in range(len(embeddings)):
            self.embeddings.append(embeddings[i])
            self.node_ids.append(node_ids[i])
            self.node_types.append(node_types[i])
            self.texts.append(texts[i])
            self.metadatas.append(metadatas[i])

        self.total_vectors = len(self.embeddings)
        print(f"✅ Added {len(embeddings)} embeddings to SimpleVectorStore")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        node_types: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """유사도 검색"""
        if not self.embeddings:
            return []

        # 코사인 유사도 계산
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        similarities = []
        valid_indices = []

        for i, embedding in enumerate(self.embeddings):
            # 노드 타입 필터링
            if node_types and self.node_types[i] not in node_types:
                continue

            # 추가 필터 적용
            if filters and not self._apply_filters(self.metadatas[i], filters):
                continue

            # 코사인 유사도 계산
            embedding_norm = np.linalg.norm(embedding)
            if embedding_norm == 0:
                similarity = 0.0
            else:
                similarity = np.dot(query_embedding, embedding) / (
                    query_norm * embedding_norm
                )

            similarities.append(similarity)
            valid_indices.append(i)

        # 상위 k개 선택
        if not similarities:
            return []

        # 유사도 기준 정렬
        sorted_pairs = sorted(zip(similarities, valid_indices), reverse=True)
        top_pairs = sorted_pairs[:top_k]

        # 결과 생성
        results = []
        for similarity, idx in top_pairs:
            result = SearchResult(
                node_id=self.node_ids[idx],
                node_type=self.node_types[idx],
                similarity_score=float(similarity),
                text=self.texts[idx],
                metadata=self.metadatas[idx],
                embedding=self.embeddings[idx],
            )
            results.append(result)

        return results

    def _apply_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """메타데이터 필터 적용"""
        for key, value in filters.items():
            if key not in metadata:
                return False

            metadata_value = metadata[key]

            if isinstance(value, list):
                if metadata_value not in value:
                    return False
            else:
                if metadata_value != value:
                    return False

        return True

    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """특정 노드 임베딩 조회"""
        try:
            idx = self.node_ids.index(node_id)
            return self.embeddings[idx]
        except ValueError:
            return None

    def save(self) -> None:
        """저장소 영구 저장"""
        if not self.persist_directory:
            print("⚠️ No persist directory configured")
            return

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        try:
            # 데이터 저장
            data = {
                "embeddings": self.embeddings,
                "node_ids": self.node_ids,
                "node_types": self.node_types,
                "texts": self.texts,
                "metadatas": self.metadatas,
                "dimension": self.dimension,
                "total_vectors": self.total_vectors,
            }

            save_file = self.persist_directory / "simple_vector_store.pkl"
            with open(save_file, "wb") as f:
                pickle.dump(data, f)

            print(f"✅ SimpleVectorStore saved to {save_file}")

        except Exception as e:
            print(f"❌ Failed to save SimpleVectorStore: {e}")

    def load(self) -> None:
        """저장소 로드"""
        if not self.persist_directory:
            return

        load_file = self.persist_directory / "simple_vector_store.pkl"
        if not load_file.exists():
            return

        try:
            with open(load_file, "rb") as f:
                data = pickle.load(f)

            self.embeddings = data["embeddings"]
            self.node_ids = data["node_ids"]
            self.node_types = data["node_types"]
            self.texts = data["texts"]
            self.metadatas = data["metadatas"]
            self.dimension = data["dimension"]
            self.total_vectors = data["total_vectors"]

            print(f"✅ SimpleVectorStore loaded: {self.total_vectors} vectors")

        except Exception as e:
            print(f"❌ Failed to load SimpleVectorStore: {e}")


class VectorStoreManager:
    """벡터 저장소 통합 관리자 (간소화 버전)"""

    def __init__(
        self,
        store_type: str = "simple",
        persist_directory: Optional[str] = None,
        collection_name: str = "graphrag_embeddings",
        **kwargs,
    ):
        """
        Args:
            store_type: 저장소 타입 ("simple" 고정)
            persist_directory: 영구 저장 디렉토리
            collection_name: 컬렉션/인덱스 이름
            **kwargs: 추가 설정
        """
        self.store_type = store_type
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # 간단한 벡터 저장소 사용
        self.store = SimpleVectorStore(persist_directory)

        print(f"✅ VectorStoreManager initialized: {store_type}")

    def load_from_embeddings(self, embedding_results) -> None:
        """EmbeddingResult로부터 벡터 저장소 구축"""

        # 차원 결정
        first_result = next(iter(next(iter(embedding_results.values()))))
        dimension = len(first_result.embedding)

        # 저장소 초기화
        self.store.initialize(dimension)

        print(f"📚 Loading embeddings into vector store...")

        # 타입별로 처리
        for node_type, results in embedding_results.items():
            if not results:
                continue

            print(f"📝 Processing {node_type}: {len(results)} embeddings")

            # 배치로 변환
            embeddings = np.array([result.embedding for result in results])
            node_ids = [result.node_id for result in results]
            node_types = [result.node_type for result in results]
            texts = [result.text for result in results]
            metadatas = [result.metadata for result in results]

            # 저장소에 추가
            self.store.add_embeddings(
                embeddings=embeddings,
                node_ids=node_ids,
                node_types=node_types,
                texts=texts,
                metadatas=metadatas,
            )

        # 저장
        self.store.save()

        total_vectors = sum(len(results) for results in embedding_results.values())
        print(f"✅ Loaded {total_vectors} embeddings into vector store")

    def search_similar_nodes(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        node_types: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """유사 노드 검색"""
        return self.store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            node_types=node_types,
            filters=filters,
        )

    def get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """특정 노드 임베딩 조회"""
        return self.store.get_embedding(node_id)

    def get_store_info(self) -> Dict[str, Any]:
        """저장소 정보 반환"""
        return {
            "store_type": self.store_type,
            "total_vectors": self.store.total_vectors,
            "dimension": self.store.dimension,
            "is_initialized": self.store.is_initialized,
            "persist_directory": self.persist_directory,
            "collection_name": self.collection_name,
        }


def create_vector_store(
    store_type: str = "simple", persist_directory: Optional[str] = None, **kwargs
) -> VectorStoreManager:
    """벡터 저장소 팩토리 함수"""
    return VectorStoreManager(
        store_type=store_type, persist_directory=persist_directory, **kwargs
    )
