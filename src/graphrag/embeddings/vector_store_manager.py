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
from .simple_vector_store import SimpleVectorStore

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
    """벡터 저장소 설정 - 확장된 버전"""

    store_type: str  # "chroma", "faiss", "simple"
    persist_directory: Optional[str] = None
    collection_name: str = "graphrag_embeddings"
    distance_metric: str = "cosine"  # "cosine", "l2", "ip"
    index_type: str = "flat"  # "flat", "ivf", "hnsw"
    batch_size: int = 1000
    cache_size: int = 10000

    # 새로운 서브폴더 지원 속성들
    faiss_directory: str = ""
    chromadb_directory: str = ""
    simple_directory: str = ""

    # FAISS 관련 설정들 (누락된 필드들 추가)
    use_gpu: bool = False
    gpu_id: int = 0
    gpu_memory_fraction: float = 0.5

    def __post_init__(self):
        """서브 디렉토리 자동 설정"""
        if self.persist_directory:
            if not self.faiss_directory:
                self.faiss_directory = f"{self.persist_directory}/faiss"
            if not self.chromadb_directory:
                self.chromadb_directory = f"{self.persist_directory}/chromadb"
            if not self.simple_directory:
                self.simple_directory = f"{self.persist_directory}/simple"


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
        """FAISS 인덱스 저장 (GPU → CPU 변환 포함)"""
        if not self.config.persist_directory:
            logger.warning("⚠️ No persist directory configured")
            return

        persist_path = Path(self.config.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        try:
            # GPU 인덱스를 CPU로 변환하여 저장
            if self.use_gpu and self.index:
                logger.info("🔄 Converting GPU index to CPU for saving...")
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            else:
                cpu_index = self.index

            # 인덱스 저장 가능 여부 확인
            if cpu_index is None:
                raise ValueError("No index to save")

            # FAISS 인덱스 저장
            index_file = persist_path / "faiss_index.bin"
            logger.info(f"💾 Saving FAISS index to {index_file}")
            faiss.write_index(cpu_index, str(index_file))

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
                "use_gpu": self.use_gpu,  # GPU 사용 여부 저장
                "index_type": self.config.index_type,
            }

            with open(metadata_file, "wb") as f:
                pickle.dump(metadata, f)

            logger.info(f"✅ FAISS index saved to {persist_path}")

        except Exception as e:
            logger.error(f"❌ Failed to save FAISS index: {e}")
            logger.error(f"   Index type: {type(self.index)}")
            logger.error(f"   Use GPU: {self.use_gpu}")

    def load(self) -> None:
        """FAISS 인덱스 로드 (GPU 변환 포함)"""
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
            # 메타데이터 먼저 로드
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)

            self.node_id_to_idx = metadata["node_id_to_idx"]
            self.idx_to_node_id = metadata["idx_to_node_id"]
            self.node_metadatas = metadata["node_metadatas"]
            self.node_texts = metadata["node_texts"]
            self.node_types = metadata["node_types"]
            self.total_vectors = metadata["total_vectors"]
            self.dimension = metadata["dimension"]

            # FAISS 인덱스 로드 (CPU로 먼저 로드)
            logger.info(f"📂 Loading FAISS index from {index_file}")
            cpu_index = faiss.read_index(str(index_file))

            # GPU 사용이 설정되어 있고 사용 가능하면 GPU로 전송
            if self.use_gpu and cpu_index:
                try:
                    logger.info("🚀 Converting loaded index to GPU...")

                    # GPU 리소스 재초기화 (필요시)
                    if not self.gpu_resources:
                        self.gpu_resources = faiss.StandardGpuResources()

                    self.index = faiss.index_cpu_to_gpu(
                        self.gpu_resources, 0, cpu_index
                    )
                    logger.info("✅ Index successfully moved to GPU")
                except Exception as gpu_error:
                    logger.warning(f"⚠️ Failed to move index to GPU: {gpu_error}")
                    logger.warning("📱 Using CPU index instead")
                    self.index = cpu_index
                    self.use_gpu = False
            else:
                self.index = cpu_index

            self.is_initialized = True
            logger.info(f"✅ FAISS index loaded: {self.total_vectors} vectors")

        except Exception as e:
            logger.error(f"❌ Failed to load FAISS index: {e}")

            # 손상된 파일이 있으면 제거
            try:
                if index_file.exists():
                    index_file.unlink()
                    logger.info("🗑️ Removed corrupted index file")
                if metadata_file.exists():
                    metadata_file.unlink()
                    logger.info("🗑️ Removed corrupted metadata file")
            except:
                pass


class VectorStoreManager:
    """벡터 저장소 통합 관리자 - 새로운 경로 구조 지원"""

    def __init__(
        self,
        store_type: str = "auto",
        persist_directory: Optional[str] = None,
        collection_name: str = "graphrag_embeddings",
        config_manager: Optional["GraphRAGConfigManager"] = None,
        **kwargs,
    ):
        """
        Args:
            store_type: 저장소 타입 ("auto", "chroma", "faiss", "simple")
            persist_directory: 영구 저장 디렉토리 (None이면 config에서 가져옴)
            collection_name: 컬렉션/인덱스 이름
            config_manager: 설정 관리자 (제공시 설정 자동 적용)
            **kwargs: 추가 설정
        """
        self.config_manager = config_manager

        # 설정 관리자가 있으면 설정을 가져옴
        if config_manager:
            vs_config = config_manager.get_vector_store_config()
            store_type = vs_config["store_type"]
            persist_directory = vs_config["persist_directory"]

            # # 설정 관리자의 값을 우선 사용
            # if store_type == "auto":
            #     store_type = vs_config["store_type"]
            # if persist_directory is None:
            #     persist_directory = vs_config["persist_directory"]

            # 추가 설정 병합
            kwargs.update(
                {
                    k: v
                    for k, v in vs_config.items()
                    if k not in ["store_type", "persist_directory"] and k not in kwargs
                }
            )

        # 자동 저장소 선택
        if store_type == "auto":
            if _faiss_available:
                store_type = "faiss"
            elif _chromadb_available:
                store_type = "chroma"
            else:
                raise ImportError(
                    "No vector store library available. Install chromadb or faiss-cpu"
                )

        # 기본 디렉토리 설정
        if persist_directory is None:
            persist_directory = "./data/processed/vector_store"

        # 설정 생성 - 저장소별 서브폴더 자동 생성
        base_config = VectorStoreConfig(
            store_type=store_type,
            persist_directory=persist_directory,
            collection_name=collection_name,
            **kwargs,
        )

        # 저장소별 전용 디렉토리 사용
        if store_type == "faiss":
            actual_persist_dir = base_config.faiss_directory
        elif store_type == "chroma":
            actual_persist_dir = base_config.chromadb_directory
        elif store_type == "simple":
            actual_persist_dir = base_config.simple_directory
        else:
            actual_persist_dir = persist_directory

        # 실제 설정 객체 생성
        self.config = VectorStoreConfig(
            store_type=store_type,
            persist_directory=actual_persist_dir,
            collection_name=collection_name,
            **kwargs,
        )

        # 디렉토리 생성
        Path(actual_persist_dir).mkdir(parents=True, exist_ok=True)

        # 벡터 저장소 초기화
        if store_type == "chroma":
            self.store = ChromaVectorStore(self.config)
        elif store_type == "faiss":
            self.store = FAISSVectorStore(self.config)
        elif store_type == "simple":
            self.store = SimpleVectorStore(self.config)
        else:
            raise ValueError(f"Unsupported store type: {store_type}")

        logger.info(f"✅ VectorStoreManager initialized: {store_type}")
        logger.info(f"   📁 Directory: {actual_persist_dir}")

    def load_from_embeddings(
        self,
        embedding_results: Dict[str, List[EmbeddingResult]],
        embeddings_dir: Optional[str] = None,
    ) -> None:
        """EmbeddingResult로부터 벡터 저장소 구축 - 새로운 경로 구조 지원"""

        # 차원 결정
        first_result = next(iter(next(iter(embedding_results.values()))))
        dimension = len(first_result.embedding)

        # 저장소 초기화
        self.store.initialize(dimension)

        # 기존 데이터 로드 시도
        self.store.load()

        logger.info(f"📚 Loading embeddings into vector store...")
        logger.info(f"   📂 Store type: {self.config.store_type}")
        logger.info(f"   📁 Directory: {self.config.persist_directory}")

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

        # 임베딩 디렉토리 정보 기록 (참조용)
        if embeddings_dir:
            self._save_embeddings_reference(embeddings_dir)

    def load_from_saved_embeddings(
        self,
        embeddings_root_dir: str,
        embeddings_subdir: str = "embeddings",
    ) -> None:
        """저장된 임베딩 파일로부터 벡터 저장소 구축"""

        embeddings_dir = Path(embeddings_root_dir) / embeddings_subdir

        if not embeddings_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

        logger.info(f"📂 Loading embeddings from saved files: {embeddings_dir}")

        try:
            # NumPy 파일들 로드
            embeddings = np.load(embeddings_dir / "embeddings.npy")
            node_ids = np.load(embeddings_dir / "node_ids.npy")
            node_types = np.load(embeddings_dir / "node_types.npy")

            # 메타데이터 로드
            with open(
                embeddings_dir / "embeddings_metadata.json", "r", encoding="utf-8"
            ) as f:
                metadata_by_type = json.load(f)

            # 인덱스 파일 로드
            with open(embeddings_dir / "node_index.json", "r", encoding="utf-8") as f:
                index_data = json.load(f)

            dimension = embeddings.shape[1]
            logger.info(f"📏 Embedding dimension: {dimension}")

            # 저장소 초기화
            self.store.initialize(dimension)

            # 기존 데이터 로드 시도
            self.store.load()

            # 메타데이터 재구성
            texts = []
            metadatas = []

            # 인덱스 순서대로 메타데이터 정렬
            for i, node_id in enumerate(node_ids):
                node_type = node_types[i]

                # 해당 노드의 메타데이터 찾기
                found_metadata = None
                found_text = ""

                if node_type in metadata_by_type:
                    for item in metadata_by_type[node_type]:
                        if item["node_id"] == node_id:
                            found_metadata = item["metadata"]
                            found_text = item["text"]
                            break

                if found_metadata is None:
                    found_metadata = {"node_type": node_type}

                metadatas.append(found_metadata)
                texts.append(found_text)

            # 저장소에 추가
            self.store.add_embeddings(
                embeddings=embeddings,
                node_ids=node_ids.tolist(),
                node_types=node_types.tolist(),
                texts=texts,
                metadatas=metadatas,
            )

            # 저장
            self.store.save()

            logger.info(f"✅ Loaded {len(embeddings)} embeddings from saved files")

        except Exception as e:
            logger.error(f"❌ Failed to load embeddings from files: {e}")
            raise

    def _save_embeddings_reference(self, embeddings_dir: str) -> None:
        """임베딩 디렉토리 참조 정보 저장"""

        reference_file = (
            Path(self.config.persist_directory) / "embeddings_reference.json"
        )

        reference_data = {
            "embeddings_directory": str(embeddings_dir),
            "vector_store_type": self.config.store_type,
            "created_at": pd.Timestamp.now().isoformat(),
            "store_info": self.get_store_info(),
        }

        try:
            with open(reference_file, "w", encoding="utf-8") as f:
                json.dump(reference_data, f, ensure_ascii=False, indent=2)

            logger.debug(f"📝 Saved embeddings reference: {reference_file}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to save embeddings reference: {e}")

    def get_store_info(self) -> Dict[str, Any]:
        """저장소 정보 반환 - 경로 정보 포함"""

        base_info = {
            "store_type": self.config.store_type,
            "total_vectors": self.store.total_vectors,
            "dimension": self.store.dimension,
            "is_initialized": self.store.is_initialized,
            "persist_directory": self.config.persist_directory,
            "collection_name": self.config.collection_name,
        }

        # 설정 관리자가 있으면 전체 경로 구조 정보 추가
        if self.config_manager:
            paths_config = self.config_manager.config.paths
            base_info["path_structure"] = {
                "vector_store_root": paths_config.vector_store_root,
                "embeddings_dir": paths_config.vector_store.embeddings,  # ✅
                "faiss_dir": paths_config.vector_store.faiss,  # ✅
                "chromadb_dir": paths_config.vector_store.chromadb,  # ✅
                "simple_dir": paths_config.vector_store.simple,  # ✅
            }

        return base_info

    def migrate_store_type(
        self,
        new_store_type: str,
        keep_original: bool = True,
    ) -> "VectorStoreManager":
        """벡터 저장소 타입 마이그레이션"""

        logger.info(f"🔄 Migrating from {self.config.store_type} to {new_store_type}")

        # 현재 데이터 추출
        if not self.store.is_initialized or self.store.total_vectors == 0:
            raise ValueError("No data to migrate")

        # 새로운 매니저 생성
        new_manager = VectorStoreManager(
            store_type=new_store_type,
            persist_directory=self.config.persist_directory.replace(
                self.config.store_type, new_store_type
            ),
            collection_name=self.config.collection_name,
            config_manager=self.config_manager,
        )

        # 데이터 복사 (임베딩을 직접 추출하여 전송)
        logger.info("📦 Extracting data from current store...")

        # 모든 노드 ID 수집
        all_node_ids = list(getattr(self.store, "node_id_to_idx", {}).keys()) or list(
            getattr(self.store, "node_ids", [])
        )

        if not all_node_ids:
            raise ValueError("Cannot extract node IDs from current store")

        # 배치로 데이터 추출 및 이전
        batch_size = 100
        total_batches = (len(all_node_ids) + batch_size - 1) // batch_size

        for i in range(0, len(all_node_ids), batch_size):
            batch_ids = all_node_ids[i : i + batch_size]

            # 현재 저장소에서 데이터 추출
            batch_embeddings = []
            batch_texts = []
            batch_types = []
            batch_metadatas = []

            for node_id in batch_ids:
                # 각 저장소별 데이터 추출 방법
                if hasattr(self.store, "node_texts"):
                    embedding = self.store.get_embedding(node_id)
                    text = self.store.node_texts.get(node_id, "")
                    node_type = self.store.node_types.get(node_id, "unknown")
                    metadata = self.store.node_metadatas.get(node_id, {})

                    if embedding is not None:
                        batch_embeddings.append(embedding)
                        batch_texts.append(text)
                        batch_types.append(node_type)
                        batch_metadatas.append(metadata)

            if batch_embeddings:
                # 새로운 저장소에 추가
                new_manager.store.add_embeddings(
                    embeddings=np.array(batch_embeddings),
                    node_ids=batch_ids[: len(batch_embeddings)],
                    node_types=batch_types,
                    texts=batch_texts,
                    metadatas=batch_metadatas,
                )

            logger.info(f"📦 Migrated batch {i//batch_size + 1}/{total_batches}")

        # 새로운 저장소 저장
        new_manager.store.save()

        logger.info(
            f"✅ Migration completed: {new_manager.store.total_vectors} vectors"
        )

        return new_manager

    def search_similar_nodes(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        node_types: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """유사 노드 검색 (기존 search 메서드 래퍼)"""
        return self.store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            node_types=node_types,
            filters=filters,
        )

    def get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """특정 노드 임베딩 조회 (기존 get_embedding 메서드 래퍼)"""
        return self.store.get_embedding(node_id)


def create_vector_store(
    store_type: str = "auto",
    persist_directory: Optional[str] = None,
    config_manager: Optional["GraphRAGConfigManager"] = None,
    **kwargs,
) -> VectorStoreManager:
    """벡터 저장소 팩토리 함수 - 설정 관리자 지원"""
    return VectorStoreManager(
        store_type=store_type,
        persist_directory=persist_directory,
        config_manager=config_manager,
        **kwargs,
    )


def create_vector_store_from_config(
    config_manager: "GraphRAGConfigManager", store_type: Optional[str] = None, **kwargs
) -> VectorStoreManager:
    """설정 관리자로부터 벡터 저장소 생성"""

    if store_type is None:
        store_type = config_manager.config.vector_store.store_type

    return VectorStoreManager(
        store_type=store_type, config_manager=config_manager, **kwargs
    )


def setup_vector_store_from_embeddings(
    embeddings_root_dir: str,
    config_manager: "GraphRAGConfigManager",
    store_type: Optional[str] = None,
    force_rebuild: bool = False,
) -> VectorStoreManager:
    """임베딩 파일로부터 벡터 저장소 완전 자동 설정"""

    logger.info("🏗️ Setting up vector store from embeddings...")

    # 벡터 저장소 생성
    vector_store = create_vector_store_from_config(
        config_manager=config_manager,
        store_type=store_type,
    )

    # 기존 저장소 확인
    if not force_rebuild and vector_store.store.total_vectors > 0:
        logger.info(
            f"✅ Existing vector store found with {vector_store.store.total_vectors} vectors"
        )
        return vector_store

    # 임베딩 파일에서 로드
    try:
        vector_store.load_from_saved_embeddings(embeddings_root_dir)
        logger.info("✅ Vector store setup completed from saved embeddings")

    except Exception as e:
        logger.error(f"❌ Failed to setup vector store: {e}")
        raise

    return vector_store


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
