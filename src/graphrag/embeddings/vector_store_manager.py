"""
벡터 저장소 관리 모듈
Vector Store Manager for GraphRAG Embeddings

다양한 벡터 저장소 지원 및 효율적인 유사도 검색
- ChromaDB: 영구 저장소 지원
- FAISS: 고속 검색 특화
- 하이브리드 검색: 벡터 + 메타데이터 필터링
- 배치 연산 및 캐싱 최적화
"""

import os
import json
import pickle
import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import defaultdict

# 벡터 저장소 의존성 체크
try:
    import chromadb
    from chromadb.config import Settings

    _chromadb_available = True
except (ImportError, RuntimeError) as e:
    _chromadb_available = False
    if "sqlite3" in str(e):
        warnings.warn(
            "ChromaDB not available due to SQLite compatibility. Using FAISS instead."
        )
    else:
        warnings.warn("ChromaDB not available. Install with: pip install chromadb")

try:
    import faiss

    _faiss_available = True
except (ImportError, AttributeError) as e:
    _faiss_available = False
    if "numpy" in str(e).lower() or "_array_api" in str(e):
        warnings.warn(
            "FAISS not available due to NumPy compatibility. Try: pip install 'numpy<2.0' faiss-cpu"
        )
    else:
        warnings.warn("FAISS not available. Install with: pip install faiss-cpu")

from .multi_node_embedder import EmbeddingResult

# 로깅 설정
logger = logging.getLogger(__name__)


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


@dataclass
class VectorStoreConfig:
    """벡터 저장소 설정"""

    store_type: str  # "chroma", "faiss", "hybrid"
    persist_directory: Optional[str] = None
    collection_name: str = "graphrag_embeddings"
    distance_metric: str = "cosine"  # "cosine", "l2", "ip"
    index_type: str = "flat"  # "flat", "ivf", "hnsw"
    batch_size: int = 1000
    cache_size: int = 10000


class BaseVectorStore(ABC):
    """벡터 저장소 기본 추상 클래스"""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.is_initialized = False
        self.dimension = None
        self.total_vectors = 0

    @abstractmethod
    def initialize(self, dimension: int) -> None:
        """벡터 저장소 초기화"""
        pass

    @abstractmethod
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        node_ids: List[str],
        node_types: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """임베딩 추가"""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        node_types: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """유사도 검색"""
        pass

    @abstractmethod
    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """특정 노드의 임베딩 반환"""
        pass

    @abstractmethod
    def save(self) -> None:
        """저장소 영구 저장"""
        pass

    @abstractmethod
    def load(self) -> None:
        """저장소 로드"""
        pass


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB 기반 벡터 저장소"""

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)

        if not _chromadb_available:
            raise ImportError("ChromaDB is required but not installed")

        self.client = None
        self.collection = None

    def initialize(self, dimension: int) -> None:
        """ChromaDB 초기화"""
        self.dimension = dimension

        # ChromaDB 클라이언트 설정
        if self.config.persist_directory:
            # 영구 저장소
            persist_path = Path(self.config.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=Settings(anonymized_telemetry=False, is_persistent=True),
            )
        else:
            # 메모리 저장소
            self.client = chromadb.Client()

        # 거리 메트릭 설정
        distance_function = {"cosine": "cosine", "l2": "l2", "ip": "ip"}.get(
            self.config.distance_metric, "cosine"
        )

        # 컬렉션 생성/로드
        try:
            self.collection = self.client.get_collection(
                name=self.config.collection_name
            )
            logger.info(
                f"✅ Loaded existing ChromaDB collection: {self.config.collection_name}"
            )
        except:
            self.collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": distance_function},
            )
            logger.info(
                f"✅ Created new ChromaDB collection: {self.config.collection_name}"
            )

        self.is_initialized = True

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        node_ids: List[str],
        node_types: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """ChromaDB에 임베딩 추가"""
        if not self.is_initialized:
            raise ValueError("Vector store not initialized")

        # ChromaDB 메타데이터 형태로 변환
        chroma_metadatas = []
        for i, (node_id, node_type, metadata) in enumerate(
            zip(node_ids, node_types, metadatas)
        ):
            chroma_meta = {
                "node_type": node_type,
                "text_length": len(texts[i]),
                "word_count": len(texts[i].split()),
            }

            # 기본 메타데이터 추가 (ChromaDB는 간단한 타입만 지원)
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    chroma_meta[key] = value
                elif isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], (str, int, float)):
                        chroma_meta[f"{key}_count"] = len(value)
                        if isinstance(value[0], str):
                            chroma_meta[f"{key}_first"] = value[0][:100]  # 첫 번째 값만

            chroma_metadatas.append(chroma_meta)

        # 배치 처리
        batch_size = self.config.batch_size
        for i in range(0, len(embeddings), batch_size):
            end_idx = min(i + batch_size, len(embeddings))

            batch_embeddings = embeddings[i:end_idx].tolist()
            batch_ids = node_ids[i:end_idx]
            batch_texts = texts[i:end_idx]
            batch_metadatas = chroma_metadatas[i:end_idx]

            try:
                self.collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                )
            except Exception as e:
                logger.warning(f"Failed to add batch {i}-{end_idx}: {e}")

        self.total_vectors += len(embeddings)
        logger.info(f"✅ Added {len(embeddings)} embeddings to ChromaDB")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        node_types: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """ChromaDB 검색"""
        if not self.is_initialized:
            raise ValueError("Vector store not initialized")

        # 필터 구성
        where_filter = {}
        if node_types:
            where_filter["node_type"] = {"$in": node_types}

        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    where_filter[key] = {"$in": value}
                else:
                    where_filter[key] = value

        # 검색 실행
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"],
            )

            # 결과 변환
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, node_id in enumerate(results["ids"][0]):
                    similarity = (
                        1.0 - results["distances"][0][i]
                    )  # ChromaDB는 거리를 반환

                    search_result = SearchResult(
                        node_id=node_id,
                        node_type=results["metadatas"][0][i].get(
                            "node_type", "unknown"
                        ),
                        similarity_score=similarity,
                        text=results["documents"][0][i],
                        metadata=results["metadatas"][0][i],
                    )
                    search_results.append(search_result)

            return search_results

        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []

    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """특정 노드 임베딩 조회"""
        try:
            result = self.collection.get(ids=[node_id], include=["embeddings"])

            if result["embeddings"] and result["embeddings"][0]:
                return np.array(result["embeddings"][0])
            return None

        except Exception as e:
            logger.warning(f"Failed to get embedding for {node_id}: {e}")
            return None

    def save(self) -> None:
        """ChromaDB 저장 (영구 저장소인 경우 자동)"""
        if self.config.persist_directory:
            logger.info("✅ ChromaDB automatically persisted")
        else:
            logger.warning("⚠️ In-memory ChromaDB cannot be persisted")

    def load(self) -> None:
        """ChromaDB 로드 (초기화 시 자동)"""
        logger.info("✅ ChromaDB loaded during initialization")


class FAISSVectorStore(BaseVectorStore):
    """FAISS 기반 벡터 저장소"""

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)

        if not _faiss_available:
            raise ImportError("FAISS is required but not installed")

        self.index = None
        self.node_id_to_idx = {}
        self.idx_to_node_id = {}
        self.node_metadatas = {}
        self.node_texts = {}
        self.node_types = {}

        # GPU 리소스 설정
        self.gpu_resources = None
        self.use_gpu = False

    def initialize(self, dimension: int) -> None:
        """FAISS 인덱스 초기화"""
        self.dimension = dimension

        # GPU 사용 가능 여부 확인
        try:
            import faiss

            ngpus = faiss.get_num_gpus()
            self.use_gpu = ngpus > 0

            if self.use_gpu:
                self.gpu_resources = faiss.StandardGpuResources()
                print(f"✅ FAISS will use GPU ({ngpus} GPUs available)")
        except:
            self.use_gpu = False
            print("⚠️ GPU not available, using CPU")

        # 인덱스 타입에 따른 생성
        if self.config.index_type == "flat":
            if self.config.distance_metric == "cosine":
                # 코사인 유사도용 (내적)
                cpu_index = faiss.IndexFlatIP(dimension)
            else:
                # L2 거리
                cpu_index = faiss.IndexFlatL2(dimension)

        elif self.config.index_type == "ivf":
            # IVF (Inverted File) 인덱스
            nlist = 100  # 클러스터 수
            quantizer = faiss.IndexFlatL2(dimension)
            cpu_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

        elif self.config.index_type == "hnsw":
            # HNSW (Hierarchical Navigable Small World)
            cpu_index = faiss.IndexHNSWFlat(dimension, 32)
            cpu_index.hnsw.efConstruction = 40
            cpu_index.hnsw.efSearch = 16

        else:
            # 기본값: Flat 인덱스
            cpu_index = faiss.IndexFlatIP(dimension)

        # GPU로 전송 (가능한 경우)
        if self.use_gpu and cpu_index:
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)
        else:
            self.index = cpu_index

        self.is_initialized = True
        gpu_status = "GPU" if self.use_gpu else "CPU"
        logger.info(
            f"✅ FAISS index initialized: {self.config.index_type}, {gpu_status}, dim={dimension}"
        )

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        node_ids: List[str],
        node_types: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """FAISS에 임베딩 추가"""
        if not self.is_initialized:
            raise ValueError("Vector store not initialized")

        # 코사인 유사도용 정규화 (필요시)
        if self.config.distance_metric == "cosine":
            # 임베딩이 이미 정규화되었는지 확인
            norms = np.linalg.norm(embeddings, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-6):
                embeddings = embeddings / norms[:, np.newaxis]

        # 인덱스 훈련 (IVF의 경우)
        if self.config.index_type == "ivf" and not self.index.is_trained:
            logger.info("🔧 Training FAISS IVF index...")
            self.index.train(embeddings.astype(np.float32))

        # 현재 인덱스 크기
        current_size = self.index.ntotal

        # 인덱스에 추가
        self.index.add(embeddings.astype(np.float32))

        # 메타데이터 저장
        for i, (node_id, node_type, text, metadata) in enumerate(
            zip(node_ids, node_types, texts, metadatas)
        ):
            idx = current_size + i
            self.node_id_to_idx[node_id] = idx
            self.idx_to_node_id[idx] = node_id
            self.node_metadatas[node_id] = metadata
            self.node_texts[node_id] = text
            self.node_types[node_id] = node_type

        self.total_vectors += len(embeddings)
        logger.info(f"✅ Added {len(embeddings)} embeddings to FAISS index")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        node_types: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """FAISS 검색"""
        if not self.is_initialized:
            raise ValueError("Vector store not initialized")

        # 쿼리 정규화 (코사인 유사도용)
        if self.config.distance_metric == "cosine":
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm

        # 검색 실행
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        # 더 많이 검색한 후 필터링 (FAISS는 메타데이터 필터링을 지원하지 않음)
        search_k = (
            min(top_k * 10, self.total_vectors) if node_types or filters else top_k
        )

        try:
            scores, indices = self.index.search(query_embedding, search_k)

            # 결과 변환 및 필터링
            search_results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # 유효하지 않은 인덱스
                    continue

                node_id = self.idx_to_node_id.get(idx)
                if not node_id:
                    continue

                node_type = self.node_types.get(node_id, "unknown")

                # 노드 타입 필터링
                if node_types and node_type not in node_types:
                    continue

                # 추가 필터 적용
                if filters:
                    metadata = self.node_metadatas.get(node_id, {})
                    if not self._apply_filters(metadata, filters):
                        continue

                # 유사도 점수 변환
                if self.config.distance_metric == "cosine":
                    similarity = float(score)  # 내적 값 (이미 정규화됨)
                else:
                    similarity = 1.0 / (1.0 + float(score))  # L2 거리를 유사도로 변환

                search_result = SearchResult(
                    node_id=node_id,
                    node_type=node_type,
                    similarity_score=similarity,
                    text=self.node_texts.get(node_id, ""),
                    metadata=self.node_metadatas.get(node_id, {}),
                )
                search_results.append(search_result)

                if len(search_results) >= top_k:
                    break

            return search_results

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

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
        idx = self.node_id_to_idx.get(node_id)
        if idx is None:
            return None

        try:
            # FAISS에서 특정 벡터 추출
            embedding = self.index.reconstruct(idx)
            return embedding
        except Exception as e:
            logger.warning(f"Failed to get embedding for {node_id}: {e}")
            return None

    def save(self) -> None:
        """FAISS 인덱스 저장"""
        if not self.config.persist_directory:
            logger.warning("⚠️ No persist directory configured")
            return

        persist_path = Path(self.config.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        try:
            # FAISS 인덱스 저장
            index_file = persist_path / "faiss_index.bin"
            faiss.write_index(self.index, str(index_file))

            # 메타데이터 저장
            metadata_file = persist_path / "faiss_metadata.pkl"
            metadata = {
                "node_id_to_idx": self.node_id_to_idx,
                "idx_to_node_id": self.idx_to_node_id,
                "node_metadatas": self.node_metadatas,
                "node_texts": self.node_texts,
                "node_types": self.node_types,
                "total_vectors": self.total_vectors,
                "dimension": self.dimension,
                "config": self.config,
            }

            with open(metadata_file, "wb") as f:
                pickle.dump(metadata, f)

            logger.info(f"✅ FAISS index saved to {persist_path}")

        except Exception as e:
            logger.error(f"❌ Failed to save FAISS index: {e}")

    def load(self) -> None:
        """FAISS 인덱스 로드"""
        if not self.config.persist_directory:
            logger.warning("⚠️ No persist directory configured")
            return

        persist_path = Path(self.config.persist_directory)
        index_file = persist_path / "faiss_index.bin"
        metadata_file = persist_path / "faiss_metadata.pkl"

        if not index_file.exists() or not metadata_file.exists():
            logger.info("📂 No existing FAISS index found")
            return

        try:
            # FAISS 인덱스 로드
            self.index = faiss.read_index(str(index_file))

            # 메타데이터 로드
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)

            self.node_id_to_idx = metadata["node_id_to_idx"]
            self.idx_to_node_id = metadata["idx_to_node_id"]
            self.node_metadatas = metadata["node_metadatas"]
            self.node_texts = metadata["node_texts"]
            self.node_types = metadata["node_types"]
            self.total_vectors = metadata["total_vectors"]
            self.dimension = metadata["dimension"]

            self.is_initialized = True
            logger.info(f"✅ FAISS index loaded: {self.total_vectors} vectors")

        except Exception as e:
            logger.error(f"❌ Failed to load FAISS index: {e}")


class VectorStoreManager:
    """벡터 저장소 통합 관리자"""

    def __init__(
        self,
        store_type: str = "auto",
        persist_directory: Optional[str] = None,
        collection_name: str = "graphrag_embeddings",
        **kwargs,
    ):
        """
        Args:
            store_type: 저장소 타입 ("auto", "chroma", "faiss", "hybrid")
            persist_directory: 영구 저장 디렉토리
            collection_name: 컬렉션/인덱스 이름
            **kwargs: 추가 설정
        """

        # 자동 저장소 선택
        if store_type == "auto":
            if _chromadb_available:
                store_type = "chroma"
            elif _faiss_available:
                store_type = "faiss"
            else:
                raise ImportError(
                    "No vector store library available. Install chromadb or faiss-cpu"
                )

        # 설정 생성
        self.config = VectorStoreConfig(
            store_type=store_type,
            persist_directory=persist_directory,
            collection_name=collection_name,
            **kwargs,
        )

        # 벡터 저장소 초기화
        if store_type == "chroma":
            self.store = ChromaVectorStore(self.config)
        elif store_type == "faiss":
            self.store = FAISSVectorStore(self.config)
        else:
            raise ValueError(f"Unsupported store type: {store_type}")

        logger.info(f"✅ VectorStoreManager initialized: {store_type}")

    def load_from_embeddings(
        self, embedding_results: Dict[str, List[EmbeddingResult]]
    ) -> None:
        """EmbeddingResult로부터 벡터 저장소 구축"""

        # 차원 결정
        first_result = next(iter(next(iter(embedding_results.values()))))
        dimension = len(first_result.embedding)

        # 저장소 초기화
        self.store.initialize(dimension)

        # 기존 데이터 로드 시도
        self.store.load()

        logger.info(f"📚 Loading embeddings into vector store...")

        # 타입별로 처리
        for node_type, results in embedding_results.items():
            if not results:
                continue

            logger.info(f"📝 Processing {node_type}: {len(results)} embeddings")

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
        logger.info(f"✅ Loaded {total_vectors} embeddings into vector store")

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
            "store_type": self.config.store_type,
            "total_vectors": self.store.total_vectors,
            "dimension": self.store.dimension,
            "is_initialized": self.store.is_initialized,
            "persist_directory": self.config.persist_directory,
            "collection_name": self.config.collection_name,
        }


def create_vector_store(
    store_type: str = "auto", persist_directory: Optional[str] = None, **kwargs
) -> VectorStoreManager:
    """벡터 저장소 팩토리 함수"""
    return VectorStoreManager(
        store_type=store_type, persist_directory=persist_directory, **kwargs
    )


def list_available_stores() -> Dict[str, bool]:
    """사용 가능한 벡터 저장소 목록"""
    return {"chromadb": _chromadb_available, "faiss": _faiss_available}


if __name__ == "__main__":
    # 테스트 실행
    print("🧪 Testing Vector Store Manager...")

    # 사용 가능한 저장소 확인
    available = list_available_stores()
    print(f"📋 Available stores: {available}")

    if not any(available.values()):
        print("❌ No vector stores available for testing")
        print("Install with: pip install chromadb faiss-cpu")
        exit(1)

    # 테스트 데이터 생성
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"🔧 Testing with temp directory: {temp_dir}")

        # 테스트 임베딩 데이터
        test_embeddings = {
            "paper": [
                EmbeddingResult(
                    node_id="paper_1",
                    node_type="paper",
                    text="Machine learning for battery optimization",
                    embedding=np.random.random(384),
                    metadata={"year": "2023", "has_abstract": True},
                ),
                EmbeddingResult(
                    node_id="paper_2",
                    node_type="paper",
                    text="Deep learning approaches to energy management",
                    embedding=np.random.random(384),
                    metadata={"year": "2022", "has_abstract": False},
                ),
            ],
            "author": [
                EmbeddingResult(
                    node_id="author_1",
                    node_type="author",
                    text="김철수 machine learning researcher",
                    embedding=np.random.random(384),
                    metadata={
                        "paper_count": 15,
                        "productivity_type": "Leading Researcher",
                    },
                )
            ],
        }

        # 각 저장소 타입 테스트
        for store_type, available in available.items():
            if not available:
                continue

            print(f"\n🔧 Testing {store_type.upper()}...")

            try:
                # 벡터 저장소 생성
                manager = create_vector_store(
                    store_type=store_type, persist_directory=f"{temp_dir}/{store_type}"
                )

                # 임베딩 로드
                manager.load_from_embeddings(test_embeddings)

                # 검색 테스트
                query_embedding = np.random.random(384)
                results = manager.search_similar_nodes(
                    query_embedding=query_embedding, top_k=5
                )

                print(f"✅ {store_type}: {len(results)} search results")
                for result in results[:2]:
                    print(f"   📄 {result.node_id}: {result.similarity_score:.3f}")

                # 정보 출력
                info = manager.get_store_info()
                print(f"📊 Store info: {info}")

            except Exception as e:
                print(f"❌ {store_type} test failed: {e}")

    print(f"\n✅ Vector Store Manager tests completed!")
