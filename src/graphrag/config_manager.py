"""
GraphRAG 설정 관리 모듈 - 완전 통합 버전
1단계 (확장된 dataclass) + 2단계 (개선된 파싱) 통합
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
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
# 1단계: YAML 구조에 맞춘 확장된 dataclass들
# ============================================================================


# LLM 관련 설정들
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
    api_key: str = os.getenv("HUGGINGFACE_API_KEY", "${HUGGINGFACE_API_KEY}")


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


# 임베딩 관련 설정들
@dataclass
class SentenceTransformersConfig:
    """SentenceTransformers 설정"""

    model_name: str = "paraphrase-multilingual-mpnet-base-v2"
    device: str = "auto"
    batch_size: int = 32
    cache_dir: str = "./cache/embeddings"


@dataclass
class OpenAIEmbeddingConfig:
    """OpenAI 임베딩 설정"""

    model_name: str = "text-embedding-ada-002"
    api_key: str = "${OPENAI_API_KEY}"
    batch_size: int = 16
    max_length: int = 8192


@dataclass
class HuggingFaceAPIEmbeddingConfig:
    """HuggingFace API 임베딩 설정"""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    api_key: str = "${HUGGINGFACE_API_KEY}"
    batch_size: int = 32
    max_length: int = 512


@dataclass
class EmbeddingsConfig:
    """임베딩 설정 (확장된 버전)"""

    model_type: str = "sentence-transformers"
    save_directory: str = "./data/processed/vector_store/embeddings"

    # 각 타입별 설정
    sentence_transformers: SentenceTransformersConfig = field(
        default_factory=SentenceTransformersConfig
    )
    openai: OpenAIEmbeddingConfig = field(default_factory=OpenAIEmbeddingConfig)
    huggingface_api: HuggingFaceAPIEmbeddingConfig = field(
        default_factory=HuggingFaceAPIEmbeddingConfig
    )


# 벡터 저장소 관련 설정들
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


# 그래프 처리 관련 설정들
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


# 하드웨어 및 성능 설정들
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


# 쿼리 분석 설정들
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


# 경로 관리 설정들
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


# 로깅 설정들
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


# 개발 및 서버 설정들
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


# 기존 단순 설정들 (호환성 유지)
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


# 메인 설정 클래스
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

    @property
    def embedding(self):
        """임베딩 설정 호환성 속성 (embedding -> embeddings)"""

        # 호환성을 위한 가짜 객체 생성
        class EmbeddingCompat:
            def __init__(self, embeddings_config):
                self.model_name = embeddings_config.sentence_transformers.model_name
                self.device = embeddings_config.sentence_transformers.device
                self.batch_size = embeddings_config.sentence_transformers.batch_size
                self.save_directory = embeddings_config.save_directory
                self.cache_dir = embeddings_config.sentence_transformers.cache_dir

        return EmbeddingCompat(self.embeddings)


# ============================================================================
# 2단계: 개선된 GraphRAGConfigManager 클래스
# ============================================================================


class GraphRAGConfigManager:
    """단순화된 GraphRAG 설정 관리자 - 개선된 YAML 파싱"""

    def __init__(
        self,
        config_file: Optional[str] = None,
        env_file: Optional[str] = None,
        auto_load: bool = True,
    ):
        self.config_file = Path(config_file) if config_file else None
        self.env_file = Path(env_file) if env_file else None

        # 기본 설정으로 시작
        self.config = GraphRAGConfig()

        if auto_load:
            self.load_all()

        logger.info("✅ GraphRAGConfigManager initialized (complete version)")

    def load_all(self) -> None:
        """모든 설정 소스 로딩 (개선된 순서)"""
        logger.info("🔧 Loading configuration with improved parsing...")

        # 1. 핵심 환경변수 로딩 (4개만)
        self._load_core_environment_variables()

        # 2. YAML 설정 파일 로딩 (메인 설정)
        if self.config_file and self.config_file.exists():
            self._load_yaml_config_file()
        elif Path("graphrag_config.yaml").exists():
            self._load_yaml_config_file(Path("graphrag_config.yaml"))

        # 3. 설정 검증 및 디렉토리 생성
        self._validate_and_setup()

        logger.info("✅ Configuration loaded with improved parsing")

    def _load_core_environment_variables(self) -> None:
        """핵심 환경변수 4개만 로딩 (단순화됨)"""
        logger.info("🌍 Loading core environment variables (4 only)...")

        # .env 파일 로딩 (있으면)
        if _dotenv_available:
            env_path = self.env_file or Path(".env")
            if env_path.exists():
                load_dotenv(env_path, override=False)
                logger.info(f"📂 Loaded .env file: {env_path}")
        else:
            logger.warning("⚠️ python-dotenv not available")

        # 1. GPU 설정
        cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
            logger.info(f"🔧 CUDA devices: {cuda_devices}")

        # 2. HuggingFace API 키
        hf_key = os.getenv("HUGGINGFACE_API_KEY")
        if hf_key:
            self.config.llm.huggingface_api.api_key = hf_key
            logger.info("🔑 HuggingFace API key loaded")

        # 3. 로그 레벨
        log_level = os.getenv("GRAPHRAG_LOG_LEVEL")
        if log_level:
            self.config.system.log_level = log_level.upper()
            self.config.logging.level = log_level.upper()
            logging.getLogger().setLevel(
                getattr(logging, log_level.upper(), logging.INFO)
            )
            logger.info(f"📝 Log level: {log_level}")

        # 4. Verbose 모드
        verbose = os.getenv("GRAPHRAG_VERBOSE")
        if verbose:
            verbose_bool = verbose.lower() in ("true", "1", "yes")
            self.config.system.verbose = verbose_bool
            logger.info(f"🔍 Verbose mode: {verbose_bool}")

    def _load_yaml_config_file(self, config_path: Optional[Path] = None) -> None:
        """YAML 설정 파일 로딩"""
        config_path = config_path or self.config_file
        logger.info(f"📂 Loading YAML config: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            if not yaml_data:
                logger.warning("⚠️ Empty YAML file")
                return

            # YAML → dataclass 매핑
            self._apply_yaml_to_dataclass(yaml_data)
            logger.info("✅ YAML config applied successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load YAML config: {e}")
            raise

    def _apply_yaml_to_dataclass(self, yaml_data: Dict[str, Any]) -> None:
        """YAML 데이터를 dataclass에 적용"""

        # 각 섹션별로 처리
        section_handlers = {
            "llm": self._apply_llm_config,
            "embeddings": self._apply_embeddings_config,
            "vector_store": self._apply_vector_store_config,
            "graph_processing": self._apply_graph_processing_config,
            "hardware": self._apply_hardware_config,
            "performance": self._apply_performance_config,
            "query_analysis": self._apply_query_analysis_config,
            "paths": self._apply_paths_config,
            "logging": self._apply_logging_config,
            "development": self._apply_development_config,
            "server": self._apply_server_config,
            # 기존 호환성 섹션들
            "graph": self._apply_graph_config,
            "qa": self._apply_qa_config,
            "system": self._apply_system_config,
        }

        for section_name, handler in section_handlers.items():
            if section_name in yaml_data:
                try:
                    handler(yaml_data[section_name])
                    logger.debug(f"✅ Applied {section_name} section")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to apply {section_name}: {e}")

    def _apply_llm_config(self, llm_data: Dict[str, Any]) -> None:
        """LLM 설정 적용"""
        # 최상위 설정
        for key, value in llm_data.items():
            if hasattr(self.config.llm, key) and not isinstance(value, dict):
                setattr(self.config.llm, key, value)

        # 중첩 설정들
        provider_configs = {
            "huggingface_local": self.config.llm.huggingface_local,
            "openai": self.config.llm.openai,
            "anthropic": self.config.llm.anthropic,
            "huggingface_api": self.config.llm.huggingface_api,
        }

        for provider, config_obj in provider_configs.items():
            if provider in llm_data and isinstance(llm_data[provider], dict):
                self._apply_nested_config(llm_data[provider], config_obj)

    def _apply_embeddings_config(self, embeddings_data: Dict[str, Any]) -> None:
        """임베딩 설정 적용"""
        for key, value in embeddings_data.items():
            if hasattr(self.config.embeddings, key) and not isinstance(value, dict):
                setattr(self.config.embeddings, key, value)

        if "sentence_transformers" in embeddings_data:
            self._apply_nested_config(
                embeddings_data["sentence_transformers"],
                self.config.embeddings.sentence_transformers,
            )

    def _apply_vector_store_config(self, vector_data: Dict[str, Any]) -> None:
        """벡터 저장소 설정 적용"""
        for key, value in vector_data.items():
            if hasattr(self.config.vector_store, key) and not isinstance(value, dict):
                setattr(self.config.vector_store, key, value)

        store_configs = {
            "faiss": self.config.vector_store.faiss,
            "chromadb": self.config.vector_store.chromadb,
            "simple": self.config.vector_store.simple,
        }

        for store_type, config_obj in store_configs.items():
            if store_type in vector_data:
                self._apply_nested_config(vector_data[store_type], config_obj)

    def _apply_graph_processing_config(self, graph_proc_data: Dict[str, Any]) -> None:
        """그래프 처리 설정 적용"""
        processing_configs = {
            "node_embeddings": self.config.graph_processing.node_embeddings,
            "subgraph_extraction": self.config.graph_processing.subgraph_extraction,
            "context_serialization": self.config.graph_processing.context_serialization,
        }

        for proc_type, config_obj in processing_configs.items():
            if proc_type in graph_proc_data:
                self._apply_nested_config(graph_proc_data[proc_type], config_obj)

    def _apply_paths_config(self, paths_data: Dict[str, Any]) -> None:
        """경로 설정 적용"""
        for key, value in paths_data.items():
            if hasattr(self.config.paths, key) and not isinstance(value, dict):
                setattr(self.config.paths, key, value)

        if "vector_store" in paths_data and isinstance(
            paths_data["vector_store"], dict
        ):
            self._apply_nested_config(
                paths_data["vector_store"], self.config.paths.vector_store
            )

    def _apply_hardware_config(self, hardware_data: Dict[str, Any]) -> None:
        """하드웨어 설정 적용"""
        self._apply_nested_config(hardware_data, self.config.hardware)

    def _apply_performance_config(self, perf_data: Dict[str, Any]) -> None:
        """성능 설정 적용"""
        self._apply_nested_config(perf_data, self.config.performance)

    def _apply_query_analysis_config(self, query_data: Dict[str, Any]) -> None:
        """쿼리 분석 설정 적용"""
        nested_configs = {
            "complexity_thresholds": self.config.query_analysis.complexity_thresholds,
            "language_detection": self.config.query_analysis.language_detection,
            "timeouts": self.config.query_analysis.timeouts,
        }

        for nested_key, config_obj in nested_configs.items():
            if nested_key in query_data:
                self._apply_nested_config(query_data[nested_key], config_obj)

    def _apply_logging_config(self, logging_data: Dict[str, Any]) -> None:
        """로깅 설정 적용 (환경변수 우선)"""
        for key, value in logging_data.items():
            if key == "level" and os.getenv("GRAPHRAG_LOG_LEVEL"):
                continue  # 환경변수 우선
            if hasattr(self.config.logging, key) and not isinstance(value, dict):
                setattr(self.config.logging, key, value)

        if "file_logging" in logging_data:
            self._apply_nested_config(
                logging_data["file_logging"], self.config.logging.file_logging
            )
        if "console_logging" in logging_data:
            self._apply_nested_config(
                logging_data["console_logging"], self.config.logging.console_logging
            )

    def _apply_development_config(self, dev_data: Dict[str, Any]) -> None:
        """개발 설정 적용"""
        self._apply_nested_config(dev_data, self.config.development)

    def _apply_server_config(self, server_data: Dict[str, Any]) -> None:
        """서버 설정 적용"""
        self._apply_nested_config(server_data, self.config.server)

    # 기존 호환성 메서드들
    def _apply_graph_config(self, graph_data: Dict[str, Any]) -> None:
        """기존 그래프 설정 적용"""
        self._apply_nested_config(graph_data, self.config.graph)

    def _apply_qa_config(self, qa_data: Dict[str, Any]) -> None:
        """QA 설정 적용"""
        self._apply_nested_config(qa_data, self.config.qa)

    def _apply_system_config(self, system_data: Dict[str, Any]) -> None:
        """시스템 설정 적용 (환경변수 우선)"""
        for key, value in system_data.items():
            if key == "log_level" and os.getenv("GRAPHRAG_LOG_LEVEL"):
                continue
            if key == "verbose" and os.getenv("GRAPHRAG_VERBOSE"):
                continue
            if hasattr(self.config.system, key):
                setattr(self.config.system, key, value)

    def _apply_nested_config(
        self, yaml_section: Dict[str, Any], config_obj: Any
    ) -> None:
        """중첩된 설정을 dataclass 객체에 적용"""
        for key, value in yaml_section.items():
            if hasattr(config_obj, key):
                if isinstance(value, list):
                    setattr(config_obj, key, value)
                elif not isinstance(value, dict):
                    setattr(config_obj, key, value)

    def _validate_and_setup(self) -> None:
        """설정 검증 및 디렉토리 생성"""
        logger.info("🔍 Validating configuration...")

        errors = []
        warnings = []

        # LLM 설정 검증
        if self.config.llm.provider == "huggingface_local":
            model_path = self.config.llm.huggingface_local.model_path
            if not model_path:
                errors.append("Local model path is required")
            elif not Path(model_path).exists():
                warnings.append(f"Local model path not found: {model_path}")

        # 온도 범위 검증
        if not (0 <= self.config.llm.temperature <= 1):
            errors.append("LLM temperature must be between 0 and 1")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"- {err}" for err in errors
            )
            raise ValueError(error_msg)

        for warning in warnings:
            logger.warning(f"⚠️ {warning}")

        # 디렉토리 생성
        self._create_directories()
        logger.info("✅ Configuration validated and directories created")

    def _create_directories(self) -> None:
        """필요한 디렉토리들 생성"""
        directories = [
            self.config.paths.data_dir,
            self.config.paths.processed_dir,
            self.config.paths.vector_store_root,
            self.config.paths.vector_store.embeddings,
            self.config.paths.vector_store.faiss,
            self.config.paths.vector_store.chromadb,
            self.config.paths.vector_store.simple,
            self.config.paths.cache_dir,
            self.config.paths.embeddings_cache,
            self.config.paths.query_cache,
            self.config.paths.logs_dir,
            self.config.system.temp_directory,
        ]

        created_count = 0
        for directory in directories:
            if directory:
                Path(directory).mkdir(exist_ok=True)
                created_count += 1

        logger.info(f"📁 Created/verified {created_count} directories")

    # ========================================================================
    # Pipeline 호환 메서드들
    # ========================================================================

    def get_llm_config(self) -> Dict[str, Any]:
        """LLM 설정 반환 (pipeline 호환)"""
        llm_config = {
            "provider": self.config.llm.provider,
            "temperature": self.config.llm.temperature,
        }

        # 프로바이더별 설정 추가
        if self.config.llm.provider == "huggingface_api":
            llm_config.update(
                {
                    "model_name": getattr(
                        self.config.llm,
                        "model_name",
                        "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    ),
                    "api_key": os.getenv("HUGGINGFACE_API_KEY"),
                    "timeout": getattr(self.config.llm, "timeout", 30),
                    "top_p": getattr(self.config.llm, "top_p", 0.9),
                }
            )
        elif self.config.llm.provider == "huggingface_local":
            hf_config = asdict(self.config.llm.huggingface_local)
            llm_config.update(hf_config)
        elif self.config.llm.provider == "openai":
            openai_config = asdict(self.config.llm.openai)
            llm_config.update(openai_config)

        return llm_config

    def get_vector_store_config(self) -> Dict[str, Any]:
        """벡터 저장소 설정 반환"""
        base_config = {
            "store_type": self.config.vector_store.store_type,
            "batch_size": self.config.vector_store.batch_size,
            "persist_directory": self.config.vector_store.persist_directory,
        }

        # 저장소 타입별 설정 추가
        if self.config.vector_store.store_type == "faiss":
            faiss_config = asdict(self.config.vector_store.faiss)
            base_config.update(faiss_config)
        elif self.config.vector_store.store_type == "chromadb":
            chroma_config = asdict(self.config.vector_store.chromadb)
            base_config.update(chroma_config)

        return base_config

    def get_embeddings_config(self) -> Dict[str, Any]:
        """임베딩 설정 반환 (모든 model_type 지원)"""

        # 기본 설정
        base_config = {
            "model_type": self.config.embeddings.model_type,
            "save_directory": self.config.embeddings.save_directory,
        }

        # model_type에 따라 안전하게 설정 추가
        if self.config.embeddings.model_type == "sentence-transformers":
            # SentenceTransformers 설정
            st_config = self.config.embeddings.sentence_transformers
            base_config.update(
                {
                    "model_name": st_config.model_name,
                    "device": st_config.device,
                    "batch_size": st_config.batch_size,
                    "cache_dir": st_config.cache_dir,
                    "max_length": 512,  # 기본값
                    "sentence_transformers": asdict(st_config),
                }
            )

        elif self.config.embeddings.model_type == "openai":
            # OpenAI 임베딩 설정 (향후 확장용)
            base_config.update(
                {
                    "model_name": "text-embedding-ada-002",  # OpenAI 기본값
                    "device": "auto",
                    "batch_size": 16,
                    "cache_dir": "./cache/embeddings",
                    "max_length": 8192,
                }
            )

        elif self.config.embeddings.model_type == "huggingface_api":
            # HuggingFace API 임베딩 설정 (향후 확장용)
            base_config.update(
                {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "device": "auto",
                    "batch_size": 32,
                    "cache_dir": "./cache/embeddings",
                    "max_length": 512,
                }
            )

        else:
            # 기본값 (안전장치)
            logger.warning(
                f"⚠️ Unknown model_type: {self.config.embeddings.model_type}, using defaults"
            )
            base_config.update(
                {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "device": "auto",
                    "batch_size": 32,
                    "cache_dir": "./cache/embeddings",
                    "max_length": 512,
                }
            )

        return base_config

    def get_paths_config(self) -> Dict[str, Any]:
        """경로 설정 반환"""
        return asdict(self.config.paths)

    def get_config_summary(self) -> str:
        """설정 요약 정보"""
        summary = f"""
GraphRAG Configuration Summary (Complete Version):
================================================
LLM Provider: {self.config.llm.provider}
Model Path: {getattr(self.config.llm.huggingface_local, 'model_path', 'N/A')}
Vector Store: {self.config.vector_store.store_type}
Embeddings Model: {getattr(self.config.embeddings.sentence_transformers, 'model_name', 'N/A')}
Log Level: {self.config.logging.level} (System: {self.config.system.log_level})
Verbose: {self.config.system.verbose}
Max Workers: {self.config.performance.max_workers}
Data Directory: {self.config.paths.data_dir}
Vector Store Root: {self.config.paths.vector_store_root}
"""
        return summary.strip()

    def get_embedding_config(self) -> Dict[str, Any]:
        """임베딩 설정 반환 (pipeline 호환성)"""
        return {
            "model_name": self.config.embeddings.sentence_transformers.model_name,
            "device": self.config.embeddings.sentence_transformers.device,
            "batch_size": self.config.embeddings.sentence_transformers.batch_size,
            "save_directory": self.config.embeddings.save_directory,
            "cache_dir": self.config.embeddings.sentence_transformers.cache_dir,
        }

    def get_system_config(self) -> Dict[str, Any]:
        """시스템 설정 반환 (pipeline 호환성)"""
        return asdict(self.config.system)

    def get_graph_config(self) -> Dict[str, Any]:
        """그래프 설정 반환 (pipeline 호환성)"""
        return asdict(self.config.graph)

    def get_qa_config(self) -> Dict[str, Any]:
        """QA 설정 반환 (pipeline 호환성)"""
        return asdict(self.config.qa)


# ============================================================================
# 팩토리 및 테스트 함수들
# ============================================================================


def create_config_manager(
    config_file: Optional[str] = None, env_file: Optional[str] = None
) -> GraphRAGConfigManager:
    """설정 관리자 팩토리 함수"""
    return GraphRAGConfigManager(config_file=config_file, env_file=env_file)


def main():
    """완전한 설정 관리자 테스트"""

    print("🧪 Testing Complete Config Manager (1단계+2단계)...")

    try:
        # 설정 관리자 초기화
        config_manager = GraphRAGConfigManager(auto_load=False)
        print("✅ Basic initialization")

        # 설정 로딩
        config_manager.load_all()
        print("✅ Configuration loading")

        # 중첩 구조 접근 테스트
        print(f"✅ LLM Provider: {config_manager.config.llm.provider}")
        print(
            f"✅ Model Path: {config_manager.config.llm.huggingface_local.model_path}"
        )
        print(f"✅ Vector Store: {config_manager.config.vector_store.store_type}")
        print(
            f"✅ FAISS Dir: {config_manager.config.vector_store.faiss.persist_directory}"
        )

        # Pipeline 호환 메서드 테스트
        llm_config = config_manager.get_llm_config()
        print(f"✅ Pipeline LLM Config: {llm_config.get('provider')}")

        vector_config = config_manager.get_vector_store_config()
        print(f"✅ Pipeline Vector Config: {vector_config.get('store_type')}")

        # 설정 요약
        print(f"\n📋 Configuration Summary:")
        print(config_manager.get_config_summary())

        print(f"\n✅ Complete config manager test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
