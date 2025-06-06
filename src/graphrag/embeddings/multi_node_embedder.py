"""
다중 노드 임베딩 생성기
MultiNodeEmbedder for GraphRAG System

통합 지식 그래프의 모든 노드 타입에 대해 최적화된 임베딩 생성
- 노드 타입별 텍스트 처리 최적화
- 배치 처리를 통한 메모리 효율성
- 진행률 추적 및 캐싱 지원
- 다양한 임베딩 모델 지원
"""

import os
import json
import pickle
import hashlib
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

# GraphRAG imports
from .embedding_models import BaseEmbeddingModel, create_embedding_model
from .node_text_processors import BaseNodeTextProcessor, create_text_processor

# 로깅 설정
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """임베딩 결과 클래스"""

    node_id: str
    node_type: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (임베딩 제외)"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "text": self.text,
            "embedding_shape": self.embedding.shape,
            "metadata": self.metadata,
        }


@dataclass
class EmbeddingStats:
    """임베딩 통계 정보"""

    total_nodes: int
    nodes_by_type: Dict[str, int]
    embedding_dimension: int
    total_size_mb: float
    processing_time_seconds: float
    model_info: Dict[str, Any]
    failed_nodes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)


class MultiNodeEmbedder:
    """다중 노드 임베딩 생성기"""

    def __init__(
        self,
        unified_graph_path: str,
        embedding_model: Union[str, BaseEmbeddingModel] = "auto",
        text_processors: Optional[Dict[str, BaseNodeTextProcessor]] = None,
        batch_size: int = 32,
        max_text_length: int = 512,
        language: str = "mixed",
        cache_dir: Optional[str] = None,
        device: str = "auto",
        **kwargs,
    ):
        """
        Args:
            unified_graph_path: 통합 그래프 JSON 파일 경로
            embedding_model: 임베딩 모델 (문자열 또는 모델 인스턴스)
            text_processors: 노드 타입별 텍스트 프로세서 딕셔너리
            batch_size: 배치 크기
            max_text_length: 최대 텍스트 길이
            language: 주요 언어 ("ko", "en", "mixed")
            cache_dir: 캐시 디렉토리
            device: 디바이스 설정
            **kwargs: 추가 설정
        """
        self.unified_graph_path = Path(unified_graph_path)
        self.batch_size = batch_size
        self.max_text_length = max_text_length
        self.language = language
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # 임베딩 모델 초기화
        if isinstance(embedding_model, str):
            self.embedding_model = create_embedding_model(
                model_name=embedding_model,
                device=device,
                batch_size=batch_size,
                **kwargs,
            )
        else:
            self.embedding_model = embedding_model

        # 텍스트 프로세서 초기화
        if text_processors:
            self.text_processors = text_processors
        else:
            self.text_processors = self._create_default_processors()

        # 데이터 저장
        self.graph_data = None
        self.embeddings_cache = {}
        self.node_index = {}  # node_id -> index 매핑

        # 캐시 설정
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_key = self._generate_cache_key()

        logger.info("✅ MultiNodeEmbedder initialized")
        logger.info(f"   📁 Graph file: {self.unified_graph_path}")
        logger.info(f"   🤖 Model: {self.embedding_model.config.model_name}")
        logger.info(f"   📏 Batch size: {batch_size}")
        logger.info(f"   🌐 Language: {language}")

    def _create_default_processors(self) -> Dict[str, BaseNodeTextProcessor]:
        """기본 텍스트 프로세서 생성"""
        processors = {}

        supported_types = ["paper", "author", "keyword", "journal"]

        for node_type in supported_types:
            try:
                processors[node_type] = create_text_processor(
                    node_type=node_type,
                    max_length=self.max_text_length,
                    language=self.language,
                )
            except Exception as e:
                logger.warning(f"Failed to create processor for {node_type}: {e}")

        return processors

    def _generate_cache_key(self) -> str:
        """캐시 키 생성 (설정 기반)"""
        key_data = {
            "graph_file": str(self.unified_graph_path),
            "model_name": self.embedding_model.config.model_name,
            "model_type": self.embedding_model.config.model_type,
            "max_length": self.max_text_length,
            "language": self.language,
            "file_mtime": (
                self.unified_graph_path.stat().st_mtime
                if self.unified_graph_path.exists()
                else 0
            ),
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def load_unified_graph(self) -> Dict[str, Any]:
        """통합 그래프 데이터 로드"""
        if self.graph_data is not None:
            return self.graph_data

        if not self.unified_graph_path.exists():
            raise FileNotFoundError(
                f"Unified graph file not found: {self.unified_graph_path}"
            )

        logger.info(f"📂 Loading unified graph from {self.unified_graph_path}")

        try:
            with open(self.unified_graph_path, "r", encoding="utf-8") as f:
                self.graph_data = json.load(f)

            # 기본 구조 검증
            if "nodes" not in self.graph_data or "edges" not in self.graph_data:
                raise ValueError("Invalid graph format: missing 'nodes' or 'edges'")

            # 노드 인덱스 생성
            self.node_index = {
                node["id"]: idx for idx, node in enumerate(self.graph_data["nodes"])
            }

            logger.info(f"✅ Graph loaded successfully")
            logger.info(f"   📄 Nodes: {len(self.graph_data['nodes']):,}")
            logger.info(f"   🔗 Edges: {len(self.graph_data['edges']):,}")

            return self.graph_data

        except Exception as e:
            logger.error(f"❌ Failed to load graph: {e}")
            raise

    def analyze_graph_structure(self) -> Dict[str, Any]:
        """그래프 구조 분석"""
        if self.graph_data is None:
            self.load_unified_graph()

        logger.info("📊 Analyzing graph structure...")

        # 노드 타입별 통계
        node_types = Counter()
        node_type_samples = defaultdict(list)

        for node in self.graph_data["nodes"]:
            node_type = node.get("node_type", "unknown")
            node_types[node_type] += 1

            # 각 타입별 샘플 수집 (처음 3개)
            if len(node_type_samples[node_type]) < 3:
                node_type_samples[node_type].append(
                    {"id": node["id"], "sample_keys": list(node.keys())}
                )

        # 엣지 타입별 통계
        edge_types = Counter()
        for edge in self.graph_data["edges"]:
            edge_type = edge.get("edge_type", "unknown")
            edge_types[edge_type] += 1

        analysis = {
            "total_nodes": len(self.graph_data["nodes"]),
            "total_edges": len(self.graph_data["edges"]),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "node_type_samples": dict(node_type_samples),
            "supported_processors": list(self.text_processors.keys()),
        }

        logger.info(f"📋 Graph Analysis:")
        logger.info(f"   📄 Total nodes: {analysis['total_nodes']:,}")
        for ntype, count in node_types.most_common():
            supported = "✅" if ntype in self.text_processors else "❌"
            logger.info(f"   {supported} {ntype}: {count:,}")

        return analysis

    def process_nodes_to_text(
        self, node_types: Optional[List[str]] = None, show_progress: bool = True
    ) -> Dict[str, List[Tuple[str, str, Dict[str, Any]]]]:
        """노드들을 타입별로 텍스트 처리

        Returns:
            {node_type: [(node_id, processed_text, metadata), ...]}
        """
        if self.graph_data is None:
            self.load_unified_graph()

        # 처리할 노드 타입 결정
        if node_types is None:
            node_types = list(self.text_processors.keys())

        # 노드 타입별로 그룹화
        nodes_by_type = defaultdict(list)
        for node in self.graph_data["nodes"]:
            node_type = node.get("node_type", "unknown")
            if node_type in node_types and node_type in self.text_processors:
                nodes_by_type[node_type].append(node)

        logger.info(f"🔤 Processing nodes to text...")
        for ntype, nodes in nodes_by_type.items():
            logger.info(f"   📝 {ntype}: {len(nodes):,} nodes")

        processed_data = {}
        failed_nodes = []

        # 타입별로 처리
        for node_type, nodes in nodes_by_type.items():
            logger.info(f"📝 Processing {node_type} nodes...")

            processor = self.text_processors[node_type]
            type_results = []

            for node in tqdm(
                nodes, desc=f"Processing {node_type}", disable=not show_progress
            ):
                try:
                    # 텍스트 생성
                    processed_text = processor.process_node(node)

                    # 메타데이터 수집
                    metadata = {
                        "node_type": node_type,
                        "original_keys": list(node.keys()),
                        "text_length": len(processed_text),
                        "word_count": len(processed_text.split()),
                    }

                    # 특별 메타데이터 (노드 타입별)
                    if node_type == "paper":
                        metadata.update(
                            {
                                "has_abstract": bool(node.get("abstract", "")),
                                "keyword_count": len(node.get("keywords", [])),
                                "author_count": len(node.get("authors", [])),
                            }
                        )
                    elif node_type == "author":
                        metadata.update(
                            {
                                "paper_count": node.get("paper_count", 0),
                                "productivity_type": node.get("productivity_type", ""),
                            }
                        )
                    elif node_type == "keyword":
                        metadata.update({"frequency": node.get("frequency", 0)})
                    elif node_type == "journal":
                        metadata.update(
                            {
                                "paper_count": node.get("paper_count", 0),
                                "journal_type": node.get("journal_type", ""),
                            }
                        )

                    type_results.append((node["id"], processed_text, metadata))

                except Exception as e:
                    logger.warning(f"Failed to process node {node['id']}: {e}")
                    failed_nodes.append(node["id"])

            processed_data[node_type] = type_results
            logger.info(
                f"✅ {node_type}: {len(type_results)} processed, {len(nodes) - len(type_results)} failed"
            )

        if failed_nodes:
            logger.warning(f"⚠️ {len(failed_nodes)} nodes failed processing")

        return processed_data

    def generate_embeddings(
        self,
        node_types: Optional[List[str]] = None,
        use_cache: bool = True,
        save_cache: bool = True,
        show_progress: bool = True,
    ) -> Dict[str, List[EmbeddingResult]]:
        """전체 임베딩 생성 파이프라인"""

        # 캐시 확인
        if use_cache and self.cache_dir:
            cached_results = self._load_from_cache()
            if cached_results:
                logger.info("✅ Loaded embeddings from cache")
                return cached_results

        # 텍스트 처리
        processed_data = self.process_nodes_to_text(node_types, show_progress)

        # 모델 로드 (지연 로딩)
        if not self.embedding_model.is_loaded():
            logger.info("📥 Loading embedding model...")
            self.embedding_model.load_model()

        embedding_results = {}
        total_nodes = sum(len(data) for data in processed_data.values())

        logger.info(f"🚀 Generating embeddings for {total_nodes:,} nodes...")

        with tqdm(
            total=total_nodes, desc="Generating embeddings", disable=not show_progress
        ) as pbar:

            for node_type, type_data in processed_data.items():
                logger.info(f"🤖 Embedding {node_type} nodes ({len(type_data):,})...")

                type_results = []

                # 배치 처리
                for i in range(0, len(type_data), self.batch_size):
                    batch_data = type_data[i : i + self.batch_size]

                    # 배치 텍스트 추출
                    batch_texts = [item[1] for item in batch_data]

                    try:
                        # 배치 임베딩 생성
                        batch_embeddings = self.embedding_model.encode(
                            batch_texts, batch_size=self.batch_size, show_progress=False
                        )

                        # 결과 객체 생성
                        for j, (node_id, text, metadata) in enumerate(batch_data):
                            embedding_result = EmbeddingResult(
                                node_id=node_id,
                                node_type=node_type,
                                text=text,
                                embedding=batch_embeddings[j],
                                metadata=metadata,
                            )
                            type_results.append(embedding_result)

                        pbar.update(len(batch_data))

                    except Exception as e:
                        logger.error(f"❌ Batch embedding failed: {e}")
                        # 개별 처리로 폴백
                        for node_id, text, metadata in batch_data:
                            try:
                                embedding = self.embedding_model.encode([text])[0]
                                embedding_result = EmbeddingResult(
                                    node_id=node_id,
                                    node_type=node_type,
                                    text=text,
                                    embedding=embedding,
                                    metadata=metadata,
                                )
                                type_results.append(embedding_result)
                            except Exception as e2:
                                logger.warning(f"Failed to embed node {node_id}: {e2}")

                            pbar.update(1)

                embedding_results[node_type] = type_results
                logger.info(f"✅ {node_type}: {len(type_results)} embeddings generated")

        # 캐시 저장
        if save_cache and self.cache_dir:
            self._save_to_cache(embedding_results)

        return embedding_results

    def _load_from_cache(self) -> Optional[Dict[str, List[EmbeddingResult]]]:
        """캐시에서 임베딩 로드"""
        if not self.cache_dir or not self._cache_key:
            return None

        cache_file = self.cache_dir / f"embeddings_{self._cache_key}.pkl"

        if not cache_file.exists():
            return None

        try:
            logger.info(f"📂 Loading from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)

            # 캐시 유효성 검증
            if "embeddings" in cached_data and "metadata" in cached_data:
                cache_meta = cached_data["metadata"]

                # 파일 수정 시간 확인
                current_mtime = self.unified_graph_path.stat().st_mtime
                if (
                    abs(cache_meta.get("file_mtime", 0) - current_mtime) < 1
                ):  # 1초 오차 허용
                    return cached_data["embeddings"]
                else:
                    logger.info("🔄 Cache outdated due to file modification")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load cache: {e}")

        return None

    def _save_to_cache(
        self, embedding_results: Dict[str, List[EmbeddingResult]]
    ) -> None:
        """캐시에 임베딩 저장"""
        if not self.cache_dir or not self._cache_key:
            return

        cache_file = self.cache_dir / f"embeddings_{self._cache_key}.pkl"

        try:
            cache_data = {
                "embeddings": embedding_results,
                "metadata": {
                    "cache_key": self._cache_key,
                    "file_mtime": self.unified_graph_path.stat().st_mtime,
                    "model_name": self.embedding_model.config.model_name,
                    "embedding_dim": self.embedding_model.get_embedding_dimension(),
                    "total_nodes": sum(
                        len(results) for results in embedding_results.values()
                    ),
                    "created_at": pd.Timestamp.now().isoformat(),
                },
            }

            logger.info(f"💾 Saving to cache: {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 캐시 파일 크기 확인
            cache_size_mb = cache_file.stat().st_size / 1024 / 1024
            logger.info(f"✅ Cache saved ({cache_size_mb:.1f} MB)")

        except Exception as e:
            logger.warning(f"⚠️ Failed to save cache: {e}")

    def save_embeddings(
        self,
        embedding_results: Dict[str, List[EmbeddingResult]],
        output_dir: str,
        formats: List[str] = ["numpy", "json"],
    ) -> Dict[str, Path]:
        """임베딩 결과를 다양한 형태로 저장"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # 전체 통계 계산
        total_nodes = sum(len(results) for results in embedding_results.values())
        if total_nodes == 0:
            logger.warning("⚠️ No embeddings to save")
            return saved_files

        first_embedding = next(iter(next(iter(embedding_results.values()))))
        embedding_dim = len(first_embedding.embedding)

        # 통계 정보 생성
        stats = EmbeddingStats(
            total_nodes=total_nodes,
            nodes_by_type={
                ntype: len(results) for ntype, results in embedding_results.items()
            },
            embedding_dimension=embedding_dim,
            total_size_mb=(total_nodes * embedding_dim * 4)
            / 1024
            / 1024,  # float32 기준
            processing_time_seconds=0,  # 실제로는 측정 필요
            model_info={
                "model_name": self.embedding_model.config.model_name,
                "model_type": self.embedding_model.config.model_type,
                "dimension": embedding_dim,
            },
            failed_nodes=[],
        )

        logger.info(f"💾 Saving embeddings to {output_dir}")

        # 1. NumPy 형태 저장 (벡터 검색용)
        if "numpy" in formats:
            embeddings_array = []
            node_ids = []
            node_types = []

            for node_type, results in embedding_results.items():
                for result in results:
                    embeddings_array.append(result.embedding)
                    node_ids.append(result.node_id)
                    node_types.append(result.node_type)

            embeddings_array = np.array(embeddings_array)

            # 배열 저장
            np.save(output_dir / "embeddings.npy", embeddings_array)
            np.save(output_dir / "node_ids.npy", np.array(node_ids))
            np.save(output_dir / "node_types.npy", np.array(node_types))

            saved_files["numpy_embeddings"] = output_dir / "embeddings.npy"
            saved_files["numpy_node_ids"] = output_dir / "node_ids.npy"
            saved_files["numpy_node_types"] = output_dir / "node_types.npy"

        # 2. JSON 메타데이터 저장
        if "json" in formats:
            # 메타데이터만 (임베딩 제외)
            metadata_dict = {}
            for node_type, results in embedding_results.items():
                metadata_dict[node_type] = [result.to_dict() for result in results]

            metadata_file = output_dir / "embeddings_metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata_dict, f, ensure_ascii=False, indent=2)

            saved_files["metadata"] = metadata_file

        # 3. 통계 정보 저장
        stats_file = output_dir / "embedding_statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f, ensure_ascii=False, indent=2)

        saved_files["statistics"] = stats_file

        # 4. 인덱스 파일 생성 (검색용)
        index_data = {
            "node_id_to_index": {node_id: idx for idx, node_id in enumerate(node_ids)},
            "index_to_node_id": {idx: node_id for idx, node_id in enumerate(node_ids)},
            "node_type_mapping": {
                node_id: node_type for node_id, node_type in zip(node_ids, node_types)
            },
            "embedding_dimension": embedding_dim,
            "total_nodes": total_nodes,
        }

        index_file = output_dir / "node_index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

        saved_files["index"] = index_file

        logger.info(f"✅ Embeddings saved successfully:")
        for format_name, file_path in saved_files.items():
            file_size = (
                file_path.stat().st_size / 1024 / 1024 if file_path.exists() else 0
            )
            logger.info(f"   📄 {format_name}: {file_path} ({file_size:.1f} MB)")

        return saved_files

    def run_full_pipeline(
        self,
        output_dir: str,
        node_types: Optional[List[str]] = None,
        use_cache: bool = True,
        save_formats: List[str] = ["numpy", "json"],
        show_progress: bool = True,
    ) -> Tuple[Dict[str, List[EmbeddingResult]], Dict[str, Path]]:
        """전체 파이프라인 실행"""

        logger.info("🚀 Starting MultiNodeEmbedder full pipeline...")

        # 1. 그래프 구조 분석
        analysis = self.analyze_graph_structure()

        # 2. 임베딩 생성
        embedding_results = self.generate_embeddings(
            node_types=node_types, use_cache=use_cache, show_progress=show_progress
        )

        # 3. 결과 저장
        saved_files = self.save_embeddings(
            embedding_results=embedding_results,
            output_dir=output_dir,
            formats=save_formats,
        )

        # 4. 요약 출력
        total_nodes = sum(len(results) for results in embedding_results.values())
        embedding_dim = self.embedding_model.get_embedding_dimension()

        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"📊 Results:")
        logger.info(f"   📄 Total nodes embedded: {total_nodes:,}")
        logger.info(f"   📏 Embedding dimension: {embedding_dim}")
        logger.info(f"   💾 Output directory: {output_dir}")

        for node_type, results in embedding_results.items():
            logger.info(f"   📝 {node_type}: {len(results):,} embeddings")

        return embedding_results, saved_files


def main():
    """테스트 실행"""
    # 기본 import 경로 설정
    import sys
    from pathlib import Path

    # src 디렉토리를 Python path에 추가
    src_dir = Path(__file__).parent.parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from src import GRAPHS_DIR, RAW_EXTRACTIONS_DIR

    print("🧪 Testing MultiNodeEmbedder...")

    # 통합 그래프 파일 확인
    unified_graph_file = GRAPHS_DIR / "unified" / "unified_knowledge_graph.json"

    if not unified_graph_file.exists():
        print(f"❌ Unified graph not found: {unified_graph_file}")
        print("Please run unified_graph_builder.py first")
        return

    try:
        # MultiNodeEmbedder 초기화
        embedder = MultiNodeEmbedder(
            unified_graph_path=str(unified_graph_file),
            embedding_model="auto",  # 자동 모델 선택
            batch_size=16,
            max_text_length=256,
            language="mixed",
            cache_dir=str(GRAPHS_DIR / "embeddings_cache"),
        )

        # 그래프 구조 분석
        analysis = embedder.analyze_graph_structure()

        # 샘플 처리 (빠른 테스트)
        print(f"\n📝 Processing sample nodes...")

        # 처음 100개 노드만으로 테스트
        embedder.graph_data["nodes"] = embedder.graph_data["nodes"][:100]

        # 임베딩 생성 및 저장
        embedding_results, saved_files = embedder.run_full_pipeline(
            output_dir=str(GRAPHS_DIR / "embeddings"),
            node_types=["paper", "author", "keyword"],  # 일부만 테스트
            use_cache=True,
            show_progress=True,
        )

        print(f"\n✅ MultiNodeEmbedder test completed!")
        print(f"📁 Check output: {GRAPHS_DIR / 'embeddings'}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
