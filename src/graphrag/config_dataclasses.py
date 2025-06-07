"""
GraphRAG 설정 관리 모듈 - 확장된 dataclass 구조
YAML 파일 구조에 완전히 매칭되는 dataclass들
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# 설정 파일 로딩
try:
    from dotenv import load_dotenv

    _dotenv_available = True
except ImportError:
    _dotenv_available = False

# 로깅 설정
logger = logging.getLogger(__name__)


class ConfigSource(Enum):
    """설정 소스 우선순위"""

    DEFAULT = 1
    CONFIG_FILE = 2
    ENVIRONMENT = 3


# ============================================================================
# LLM 관련 설정들
# ============================================================================


@dataclass
class HuggingFaceLocalConfig:
    """HuggingFace 로컬 모델 설정"""

    model_path: str = "/DATA/MODELS/models--meta-llama--Llama-3.1-8B-Instruct"
    max_new_tokens: int = 2048
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    batch_size: int = 32


@dataclass
class OpenAIConfig:
    """OpenAI API 설정"""

    api_key: str = "${OPENAI_API_KEY}"
    model_name: str = "gpt-4o"
    timeout: int = 60


@dataclass
class AnthropicConfig:
    """Anthropic API 설정"""

    api_key: str = "${ANTHROPIC_API_KEY}"
    model_name: str = "claude-3-5-sonnet"
    timeout: int = 60


@dataclass
class HuggingFaceAPIConfig:
    """HuggingFace API 설정"""

    model_name: str = "microsoft/DialoGPT-large"
    api_key: str = "${HUGGINGFACE_API_KEY}"


@dataclass
class LLMConfig:
    """LLM 통합 설정"""

    provider: str = "huggingface_local"
    temperature: float = 0.1

    # 각 프로바이더별 중첩 설정
    huggingface_local: HuggingFaceLocalConfig = field(
        default_factory=HuggingFaceLocalConfig
    )
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    huggingface_api: HuggingFaceAPIConfig = field(default_factory=HuggingFaceAPIConfig)


# ============================================================================
# 임베딩 관련 설정들
# ============================================================================


@dataclass
class SentenceTransformersConfig:
    """SentenceTransformers 설정"""

    model_name: str = "paraphrase-multilingual-mpnet-base-v2"
    device: str = "auto"
    batch_size: int = 32
    cache_dir: str = "./cache/embeddings"


@dataclass
class EmbeddingsConfig:
    """임베딩 설정 (YAML의 embeddings 키에 대응)"""

    model_type: str = "sentence-transformers"
    save_directory: str = "./data/processed/vector_store/embeddings"

    sentence_transformers: SentenceTransformersConfig = field(
        default_factory=SentenceTransformersConfig
    )


# ============================================================================
# 벡터 저장소 관련 설정들
# ============================================================================


@dataclass
class FAISSConfig:
    """FAISS 벡터 저장소 설정"""

    persist_directory: str = "./data/processed/vector_store/faiss"
    index_type: str = "flat"
    distance_metric: str = "cosine"
    use_gpu: bool = False
    gpu_id: int = 0
    gpu_memory_fraction: float = 0.5


@dataclass
class ChromaDBConfig:
    """ChromaDB 벡터 저장소 설정"""

    persist_directory: str = "./data/processed/vector_store/chromadb"
    collection_name: str = "graphrag_embeddings"
    distance_metric: str = "cosine"


@dataclass
class SimpleStoreConfig:
    """Simple 벡터 저장소 설정"""

    persist_directory: str = "./data/processed/vector_store/simple"


@dataclass
class VectorStoreConfig:
    """벡터 저장소 통합 설정"""

    store_type: str = "faiss"
    batch_size: int = 128
    persist_directory: str = "./data/processed/vector_store"

    # 각 저장소별 중첩 설정
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    chromadb: ChromaDBConfig = field(default_factory=ChromaDBConfig)
    simple: SimpleStoreConfig = field(default_factory=SimpleStoreConfig)


# ============================================================================
# 그래프 처리 관련 설정들
# ============================================================================


@dataclass
class NodeEmbeddingsConfig:
    """노드 임베딩 설정"""

    max_text_length: int = 512
    batch_size: int = 32
    cache_embeddings: bool = True
    cache_dir: str = "./cache/embeddings"
    output_directory: str = "./data/processed/vector_store/embeddings"


@dataclass
class SubgraphExtractionConfig:
    """서브그래프 추출 설정"""

    max_nodes: int = 300
    max_edges: int = 800
    max_hops: int = 3
    initial_top_k: int = 25
    similarity_threshold: float = 0.5
    expansion_factor: float = 2.5


@dataclass
class ContextSerializationConfig:
    """컨텍스트 직렬화 설정"""

    max_tokens: int = 8000
    format_style: str = "structured"
    language: str = "mixed"
    include_statistics: bool = True
    include_relationships: bool = True


@dataclass
class GraphProcessingConfig:
    """그래프 처리 통합 설정"""

    node_embeddings: NodeEmbeddingsConfig = field(default_factory=NodeEmbeddingsConfig)
    subgraph_extraction: SubgraphExtractionConfig = field(
        default_factory=SubgraphExtractionConfig
    )
    context_serialization: ContextSerializationConfig = field(
        default_factory=ContextSerializationConfig
    )


# ============================================================================
# 하드웨어 및 성능 설정들
# ============================================================================


@dataclass
class HardwareConfig:
    """하드웨어 최적화 설정"""

    use_gpu: bool = True
    gpu_memory_fraction: float = 0.7
    mixed_precision: bool = True
    cpu_threads: int = 8
    enable_gradient_checkpointing: bool = True
    enable_cpu_offload: bool = False


@dataclass
class PerformanceConfig:
    """성능 최적화 설정"""

    enable_parallel: bool = True
    max_workers: int = 4
    enable_caching: bool = True
    cache_size_limit: str = "8GB"
    batch_processing: bool = True
    memory_limit: str = "16GB"
    enable_flash_attention: bool = True
    enable_model_parallelism: bool = True


# ============================================================================
# 쿼리 분석 설정들
# ============================================================================


@dataclass
class ComplexityThresholds:
    """복잡도 임계값 설정"""

    simple_max: float = 0.3
    medium_max: float = 0.6
    complex_max: float = 0.8


@dataclass
class LanguageDetectionConfig:
    """언어 감지 설정"""

    default_language: str = "ko"
    supported_languages: List[str] = field(default_factory=lambda: ["ko", "en"])


@dataclass
class QueryTimeouts:
    """쿼리 타임아웃 설정"""

    simple: int = 20
    medium: int = 45
    complex: int = 120
    exploratory: int = 240


@dataclass
class QueryAnalysisConfig:
    """쿼리 분석 설정"""

    complexity_thresholds: ComplexityThresholds = field(
        default_factory=ComplexityThresholds
    )
    language_detection: LanguageDetectionConfig = field(
        default_factory=LanguageDetectionConfig
    )
    timeouts: QueryTimeouts = field(default_factory=QueryTimeouts)


# ============================================================================
# 경로 관리 설정들
# ============================================================================


@dataclass
class VectorStorePathsConfig:
    """벡터 저장소 경로들"""

    embeddings: str = "./data/processed/vector_store/embeddings"
    faiss: str = "./data/processed/vector_store/faiss"
    chromadb: str = "./data/processed/vector_store/chromadb"
    simple: str = "./data/processed/vector_store/simple"


@dataclass
class PathsConfig:
    """경로 설정 통합"""

    data_dir: str = "./data"
    processed_dir: str = "./data/processed"
    unified_graph: str = "./data/processed/graphs/unified/unified_knowledge_graph.json"
    individual_graphs_dir: str = "./data/processed/graphs"
    vector_store_root: str = "./data/processed/vector_store"
    cache_dir: str = "./cache"
    embeddings_cache: str = "./cache/embeddings"
    query_cache: str = "./cache/queries"
    logs_dir: str = "./logs"
    models_dir: str = "/DATA/MODELS"

    # 벡터 저장소 하위 구조
    vector_store: VectorStorePathsConfig = field(default_factory=VectorStorePathsConfig)


# ============================================================================
# 로깅 설정들
# ============================================================================


@dataclass
class FileLoggingConfig:
    """파일 로깅 설정"""

    enabled: bool = True
    log_file: str = "./logs/graphrag.log"
    max_size: str = "50MB"
    backup_count: int = 5


@dataclass
class ConsoleLoggingConfig:
    """콘솔 로깅 설정"""

    enabled: bool = True
    colored: bool = True


@dataclass
class LoggingConfig:
    """로깅 통합 설정"""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_logging: FileLoggingConfig = field(default_factory=FileLoggingConfig)
    console_logging: ConsoleLoggingConfig = field(default_factory=ConsoleLoggingConfig)


# ============================================================================
# 개발 및 서버 설정들
# ============================================================================


@dataclass
class DevelopmentConfig:
    """개발 설정"""

    debug_mode: bool = False
    test_mode: bool = False
    sample_data_only: bool = False
    max_test_nodes: int = 200
    enable_profiling: bool = True


@dataclass
class ServerConfig:
    """서버 환경 설정"""

    preload_models: bool = True
    model_cache_size: int = 2
    auto_cleanup: bool = True
    cleanup_interval: int = 3600
    restrict_model_access: bool = True


# ============================================================================
# 기존 단순 설정들 (호환성 유지)
# ============================================================================


@dataclass
class GraphConfig:
    """그래프 설정 (기존 호환성 유지)"""

    unified_graph_path: str = (
        "./data/processed/graphs/unified/unified_knowledge_graph.json"
    )
    vector_store_path: str = "./data/processed/vector_store"
    graphs_directory: str = "./data/processed/graphs"
    cache_enabled: bool = True
    cache_ttl_hours: int = 24


@dataclass
class QAConfig:
    """QA 체인 설정"""

    chain_type: str = "retrieval_qa"
    max_docs: int = 10
    min_relevance_score: float = 0.3
    return_source_documents: bool = True
    enable_memory: bool = False
    memory_type: str = "buffer"
    max_memory_tokens: int = 4000


@dataclass
class SystemConfig:
    """시스템 설정 (기본)"""

    log_level: str = "INFO"
    verbose: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    temp_directory: str = "./tmp"
    enable_monitoring: bool = False


# ============================================================================
# 메인 설정 클래스
# ============================================================================


@dataclass
class GraphRAGConfig:
    """GraphRAG 전체 설정 - YAML 구조에 완전 매칭"""

    # 메인 설정 섹션들
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    graph_processing: GraphProcessingConfig = field(
        default_factory=GraphProcessingConfig
    )
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    query_analysis: QueryAnalysisConfig = field(default_factory=QueryAnalysisConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    # 기존 호환성 유지용
    graph: GraphConfig = field(default_factory=GraphConfig)
    qa: QAConfig = field(default_factory=QAConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    # 메타데이터
    version: str = "1.0.0"
    config_source: ConfigSource = ConfigSource.DEFAULT
    last_updated: Optional[str] = None


def main():
    """dataclass 구조 테스트"""

    print("🧪 Testing expanded dataclass structure...")

    try:
        # 기본 설정 생성
        config = GraphRAGConfig()

        # 중첩 구조 접근 테스트
        print(f"✅ LLM Provider: {config.llm.provider}")
        print(f"✅ HuggingFace Model Path: {config.llm.huggingface_local.model_path}")
        print(f"✅ Vector Store Type: {config.vector_store.store_type}")
        print(f"✅ FAISS Directory: {config.vector_store.faiss.persist_directory}")
        print(f"✅ Embeddings Directory: {config.paths.vector_store.embeddings}")
        print(f"✅ Log Level: {config.logging.level}")

        # YAML과 매칭되는 구조인지 확인
        yaml_sections = [
            "llm",
            "embeddings",
            "vector_store",
            "graph_processing",
            "hardware",
            "performance",
            "query_analysis",
            "paths",
            "logging",
            "development",
            "server",
        ]

        for section in yaml_sections:
            if hasattr(config, section):
                print(f"✅ Section '{section}' available")
            else:
                print(f"❌ Section '{section}' missing")

        print(f"\n✅ Expanded dataclass structure test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
