"""
GraphRAG 설정 관리 모듈
Configuration Manager for GraphRAG System

시스템 전체 설정 및 인증 정보 통합 관리
- 환경 변수 및 .env 파일 지원
- YAML/JSON 설정 파일 로딩
- API 키 및 인증 정보 보안 관리
- 설정 검증 및 기본값 제공
- 런타임 설정 업데이트
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings

# 설정 파일 로딩
try:
    from dotenv import load_dotenv

    _dotenv_available = True
except ImportError:
    _dotenv_available = False
    warnings.warn(
        "python-dotenv not available. Install with: pip install python-dotenv"
    )

# 로깅 설정
logger = logging.getLogger(__name__)


class ConfigSource(Enum):
    """설정 소스 우선순위 (높은 숫자가 우선)"""

    DEFAULT = 1
    CONFIG_FILE = 2
    ENV_FILE = 3
    ENVIRONMENT = 4
    RUNTIME = 5


@dataclass
class LLMConfig:
    """LLM 설정 - YAML 구조에 맞춤"""

    provider: str = "huggingface_local"  # YAML 기본값에 맞춤
    model_name: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    streaming: bool = False
    timeout: int = 60

    # 로컬 모델 설정 (YAML에서 huggingface_local 섹션)
    model_path: Optional[str] = (
        "/DATA/MODELS/models--meta-llama--Llama-3.1-8B-Instruct"  # YAML 기본값
    )
    device_map: str = "auto"  # GPU 할당
    torch_dtype: str = "bfloat16"  # 메모리 효율성
    max_new_tokens: int = 2048  # 생성 토큰 수
    trust_remote_code: bool = True  # HF 모델용
    load_in_8bit: bool = False  # 양자화 옵션
    load_in_4bit: bool = False  # 더 강한 양자화

    # 생성 설정 (YAML에 맞춤)
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    batch_size: int = 1


@dataclass
class PathConfig:
    """경로 설정 클래스 - YAML paths 섹션에 맞춤"""

    # 기본 디렉토리 (YAML과 일치)
    data_dir: str = "./data"
    processed_dir: str = "./data/processed"

    # 그래프 관련 (YAML paths에 맞춤)
    unified_graph: str = "./data/processed/graphs/unified/unified_knowledge_graph.json"
    individual_graphs_dir: str = "./data/processed/graphs"

    # 벡터 저장소 구조 (YAML vector_store 섹션과 일치)
    vector_store_root: str = "./data/processed/vector_store"
    vector_store_embeddings: str = "./data/processed/vector_store/embeddings"
    vector_store_faiss: str = "./data/processed/vector_store/faiss"
    vector_store_chromadb: str = "./data/processed/vector_store/chromadb"
    vector_store_simple: str = "./data/processed/vector_store/simple"

    # 캐시 디렉토리 (YAML과 일치)
    cache_dir: str = "./cache"
    embeddings_cache: str = "./cache/embeddings"
    query_cache: str = "./cache/queries"
    logs_dir: str = "./logs"

    # 서버 모델 경로 (YAML과 일치)
    models_dir: str = "/DATA/MODELS"


@dataclass
class EmbeddingConfig:
    """임베딩 설정 - YAML embeddings 섹션에 맞춤"""

    model_type: str = "sentence-transformers"  # YAML 기본값
    model_name: str = "paraphrase-multilingual-mpnet-base-v2"  # YAML 기본값
    device: str = "auto"
    batch_size: int = 32
    max_length: int = 512
    cache_dir: str = "./cache/embeddings"

    # 임베딩 파일 저장 경로 (YAML과 일치)
    save_directory: str = "./data/processed/vector_store/embeddings"


@dataclass
class VectorStoreConfig:
    """벡터 저장소 설정 - YAML vector_store 섹션에 맞춤"""

    store_type: str = "faiss"  # YAML 기본값
    persist_directory: str = "./data/processed/vector_store"  # YAML 루트 경로
    collection_name: str = "graphrag_embeddings"
    distance_metric: str = "cosine"
    index_type: str = "flat"
    batch_size: int = 128  # YAML 값에 맞춤
    cache_size: int = 10000

    # 중첩 구조 저장소별 설정 (YAML에서 로드)
    faiss: Optional[Dict[str, Any]] = None
    chromadb: Optional[Dict[str, Any]] = None
    simple: Optional[Dict[str, Any]] = None

    # 기존 평면화된 속성들 (하위 호환성용)
    faiss_directory: str = "./data/processed/vector_store/faiss"
    chromadb_directory: str = "./data/processed/vector_store/chromadb"
    simple_directory: str = "./data/processed/vector_store/simple"

    # FAISS 관련 설정들 (YAML 기본값에 맞춤)
    faiss_index_type: str = "flat"
    faiss_distance_metric: str = "cosine"
    use_gpu: bool = False  # YAML에서 개발 단계는 CPU 사용
    gpu_id: int = 0
    gpu_memory_fraction: float = 0.5

    # ChromaDB 관련 설정들
    chromadb_collection_name: str = "graphrag_embeddings"
    chromadb_distance_metric: str = "cosine"

    def __post_init__(self):
        """서브 디렉토리 자동 설정 - 중첩 구조와 평면 구조 모두 지원"""
        # 중첩 구조에서 평면 구조로 변환 (필요시)
        if self.faiss and isinstance(self.faiss, dict):
            self.faiss_directory = self.faiss.get(
                "persist_directory", f"{self.persist_directory}/faiss"
            )
            self.faiss_index_type = self.faiss.get("index_type", "flat")
            self.faiss_distance_metric = self.faiss.get("distance_metric", "cosine")
            self.use_gpu = self.faiss.get("use_gpu", False)
            self.gpu_id = self.faiss.get("gpu_id", 0)
            self.gpu_memory_fraction = self.faiss.get("gpu_memory_fraction", 0.5)

        if self.chromadb and isinstance(self.chromadb, dict):
            self.chromadb_directory = self.chromadb.get(
                "persist_directory", f"{self.persist_directory}/chromadb"
            )
            self.chromadb_collection_name = self.chromadb.get(
                "collection_name", "graphrag_embeddings"
            )
            self.chromadb_distance_metric = self.chromadb.get(
                "distance_metric", "cosine"
            )

        if self.simple and isinstance(self.simple, dict):
            self.simple_directory = self.simple.get(
                "persist_directory", f"{self.persist_directory}/simple"
            )


@dataclass
class GraphConfig:
    """그래프 설정 - YAML graph 섹션에 맞춤"""

    unified_graph_path: str = (
        "./data/processed/graphs/unified/unified_knowledge_graph.json"
    )
    vector_store_path: str = "./data/processed/vector_store"  # 루트 경로
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
    """시스템 설정 - YAML 성능 최적화 섹션 반영"""

    log_level: str = "INFO"
    verbose: bool = False
    parallel_processing: bool = True
    max_workers: int = 4  # YAML performance 섹션과 일치
    temp_directory: str = "./tmp"
    enable_monitoring: bool = False

    # YAML performance 섹션 설정들 추가
    enable_parallel: bool = True
    enable_caching: bool = True
    cache_size_limit: str = "8GB"
    batch_processing: bool = True
    memory_limit: str = "16GB"

    # 고급 최적화 (YAML에서)
    enable_flash_attention: bool = True
    enable_model_parallelism: bool = True


@dataclass
class GraphRAGConfig:
    """GraphRAG 전체 설정 - YAML 구조에 완전히 맞춤"""

    # 하위 설정들 - 기본값들이 YAML과 일치
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    qa: QAConfig = field(default_factory=QAConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    # 경로 및 벡터 저장소 설정
    paths: PathConfig = field(default_factory=PathConfig)
    vector_store: VectorStoreConfig = field(
        default_factory=lambda: VectorStoreConfig(store_type="faiss")
    )

    # 메타데이터
    version: str = "1.0.0"
    config_source: ConfigSource = ConfigSource.DEFAULT
    last_updated: Optional[str] = None


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
    """시스템 설정"""

    log_level: str = "INFO"
    verbose: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    temp_directory: str = "./tmp"
    enable_monitoring: bool = False


class GraphRAGConfigManager:
    """GraphRAG 설정 관리자"""

    def __init__(
        self,
        config_file: Optional[str] = None,
        env_file: Optional[str] = None,
        auto_load: bool = True,
    ):
        """
        Args:
            config_file: 설정 파일 경로 (YAML/JSON)
            env_file: .env 파일 경로
            auto_load: 자동 로딩 여부
        """
        self.config_file = Path(config_file) if config_file else None
        self.env_file = Path(env_file) if env_file else None

        # 기본 설정으로 시작
        self.config = GraphRAGConfig()

        # 설정 소스별 저장 (디버깅용)
        self.config_sources: Dict[str, Any] = {}

        if auto_load:
            self.load_all()

        logger.info("✅ GraphRAGConfigManager initialized")

    def load_all(self) -> None:
        """모든 설정 소스를 우선순위에 따라 로딩"""

        logger.info("🔧 Loading configuration from all sources...")

        # 1. 기본 설정 (이미 적용됨)
        self.config_sources["default"] = asdict(self.config)

        # 2. 설정 파일
        if self.config_file and self.config_file.exists():
            self._load_config_file()

        # 3. .env 파일
        if self.env_file and self.env_file.exists():
            self._load_env_file()
        elif Path(".env").exists():
            self._load_env_file(Path(".env"))

        # 4. 환경 변수
        self._load_environment_variables()

        # 5. 설정 검증
        self._validate_config()

        logger.info("✅ Configuration loaded successfully")

    def _load_config_file(self) -> None:
        """설정 파일 로딩 (YAML/JSON)"""

        logger.info(f"📂 Loading config file: {self.config_file}")

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                if self.config_file.suffix.lower() in [".yml", ".yaml"]:
                    file_config = yaml.safe_load(f)
                elif self.config_file.suffix.lower() == ".json":
                    file_config = json.load(f)
                else:
                    raise ValueError(
                        f"Unsupported config file format: {self.config_file.suffix}"
                    )

            self.config_sources["config_file"] = file_config
            self._merge_config(file_config, ConfigSource.CONFIG_FILE)

            logger.info("✅ Config file loaded")

        except Exception as e:
            logger.error(f"❌ Failed to load config file: {e}")
            raise

    def _load_env_file(self, env_path: Optional[Path] = None) -> None:
        """환경 파일 로딩"""

        if not _dotenv_available:
            logger.warning("⚠️ python-dotenv not available, skipping .env file")
            return

        env_path = env_path or self.env_file
        logger.info(f"🔐 Loading environment file: {env_path}")

        try:
            # .env 파일 로딩
            load_dotenv(env_path, override=False)

            # GraphRAG 관련 환경 변수들 추출
            env_config = self._extract_env_variables()
            self.config_sources["env_file"] = env_config
            self._merge_config(env_config, ConfigSource.ENV_FILE)

            logger.info("✅ Environment file loaded")

        except Exception as e:
            logger.error(f"❌ Failed to load environment file: {e}")

    def _load_environment_variables(self) -> None:
        """환경 변수 로딩"""

        logger.info("🌍 Loading environment variables...")

        env_config = self._extract_env_variables()
        self.config_sources["environment"] = env_config
        self._merge_config(env_config, ConfigSource.ENVIRONMENT)

    def _extract_env_variables(self) -> Dict[str, Any]:
        """환경 변수에서 GraphRAG 설정 추출"""

        env_config = {}

        # LLM 설정
        llm_config = {}
        if os.getenv("OPENAI_API_KEY"):
            llm_config["api_key"] = os.getenv("OPENAI_API_KEY")
            llm_config["provider"] = "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            llm_config["api_key"] = os.getenv("ANTHROPIC_API_KEY")
            llm_config["provider"] = "anthropic"
        if os.getenv("HUGGINGFACE_API_TOKEN"):
            llm_config["api_key"] = os.getenv("HUGGINGFACE_API_TOKEN")
            llm_config["provider"] = "huggingface"

        if os.getenv("GRAPHRAG_LLM_MODEL"):
            llm_config["model_name"] = os.getenv("GRAPHRAG_LLM_MODEL")
        if os.getenv("GRAPHRAG_LLM_TEMPERATURE"):
            llm_config["temperature"] = float(os.getenv("GRAPHRAG_LLM_TEMPERATURE"))
        if os.getenv("GRAPHRAG_LLM_MAX_TOKENS"):
            llm_config["max_tokens"] = int(os.getenv("GRAPHRAG_LLM_MAX_TOKENS"))

        if os.getenv("GRAPHRAG_LOCAL_MODEL_PATH"):
            llm_config["model_path"] = os.getenv("GRAPHRAG_LOCAL_MODEL_PATH")
            llm_config["provider"] = "huggingface_local"

        if os.getenv("GRAPHRAG_MODEL_DEVICE_MAP"):
            llm_config["device_map"] = os.getenv("GRAPHRAG_MODEL_DEVICE_MAP")

        if os.getenv("GRAPHRAG_MODEL_DTYPE"):
            llm_config["torch_dtype"] = os.getenv("GRAPHRAG_MODEL_DTYPE")

        if llm_config:
            env_config["llm"] = llm_config

        # 임베딩 설정
        embedding_config = {}
        if os.getenv("GRAPHRAG_EMBEDDING_MODEL"):
            embedding_config["model_name"] = os.getenv("GRAPHRAG_EMBEDDING_MODEL")
        if os.getenv("GRAPHRAG_EMBEDDING_DEVICE"):
            embedding_config["device"] = os.getenv("GRAPHRAG_EMBEDDING_DEVICE")
        if os.getenv("GRAPHRAG_EMBEDDING_BATCH_SIZE"):
            embedding_config["batch_size"] = int(
                os.getenv("GRAPHRAG_EMBEDDING_BATCH_SIZE")
            )

        if embedding_config:
            env_config["embedding"] = embedding_config

        # 그래프 설정
        graph_config = {}
        if os.getenv("GRAPHRAG_UNIFIED_GRAPH_PATH"):
            graph_config["unified_graph_path"] = os.getenv(
                "GRAPHRAG_UNIFIED_GRAPH_PATH"
            )
        if os.getenv("GRAPHRAG_VECTOR_STORE_PATH"):
            graph_config["vector_store_path"] = os.getenv("GRAPHRAG_VECTOR_STORE_PATH")
        if os.getenv("GRAPHRAG_GRAPHS_DIRECTORY"):
            graph_config["graphs_directory"] = os.getenv("GRAPHRAG_GRAPHS_DIRECTORY")

        if graph_config:
            env_config["graph"] = graph_config

        # 시스템 설정
        system_config = {}
        if os.getenv("GRAPHRAG_LOG_LEVEL"):
            system_config["log_level"] = os.getenv("GRAPHRAG_LOG_LEVEL")
        if os.getenv("GRAPHRAG_VERBOSE"):
            system_config["verbose"] = os.getenv("GRAPHRAG_VERBOSE").lower() == "true"
        if os.getenv("GRAPHRAG_MAX_WORKERS"):
            system_config["max_workers"] = int(os.getenv("GRAPHRAG_MAX_WORKERS"))

        if system_config:
            env_config["system"] = system_config

        return env_config

    def _merge_config(self, new_config: Dict[str, Any], source: ConfigSource) -> None:
        """새로운 설정을 기존 설정에 병합"""

        # 깊은 병합 수행
        def deep_merge(base: Dict, update: Dict) -> Dict:
            for key, value in update.items():
                if (
                    key in base
                    and isinstance(base[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        # dataclass를 dict로 변환
        config_dict = asdict(self.config)

        # 병합
        deep_merge(config_dict, new_config)

        # dict를 다시 dataclass로 변환
        self.config = self._dict_to_config(config_dict)
        self.config.config_source = source

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> GraphRAGConfig:
        """딕셔너리를 GraphRAGConfig로 변환 (중첩 구조 지원)"""
        print(config_dict.keys())
        import pdb

        pdb.set_trace()
        # LLM 설정 처리 - 중첩 구조 평면화
        llm_data = config_dict.get("llm", {}).copy()
        embedding_data = config_dict.get("embedding", {}).copy()
        if not embedding_data:
            embedding_data = config_dict.get("embeddings", {}).copy()

        if "provider" in llm_data:
            provider = llm_data["provider"]

            # provider별 중첩 설정 평면화
            if provider in llm_data:  # huggingface_local, openai, anthropic 등
                logger.info(f"🔧 Processing nested config for provider: {provider}")

                nested_config = llm_data[provider]

                # 중첩된 설정을 상위로 병합 (기존 키 우선)
                for key, value in nested_config.items():
                    if key not in llm_data:  # 기존 키가 없으면 추가
                        llm_data[key] = value
                        logger.debug(f"   Added {key} = {value}")

                # 중첩 섹션 제거
                del llm_data[provider]
                logger.info(f"✅ Flattened {provider} config")

        # 임베딩 설정 처리 - 저장 경로 추가
        if "model_type" in embedding_data:
            model_type = embedding_data["model_type"]
            if model_type in embedding_data:
                logger.info(f"🔧 Processing nested embedding config for: {model_type}")
                nested_config = embedding_data[model_type]
                for key, value in nested_config.items():
                    if key not in embedding_data:
                        embedding_data[key] = value
                        logger.debug(f"   Added embedding {key} = {value}")
                del embedding_data[model_type]
                logger.info(f"✅ Flattened {model_type} embedding config")

        # 경로 설정 처리
        paths_data = config_dict.get("paths", {})

        # paths에서 vector_store 섹션 처리
        if "vector_store" in paths_data:
            vs_paths = paths_data["vector_store"]
            paths_data.update(
                {
                    "vector_store_embeddings": vs_paths.get("embeddings", ""),
                    "vector_store_faiss": vs_paths.get("faiss", ""),
                    "vector_store_chromadb": vs_paths.get("chromadb", ""),
                    "vector_store_simple": vs_paths.get("simple", ""),
                }
            )
            # 중첩 구조 제거
            del paths_data["vector_store"]

        # 벡터 저장소 설정 처리
        vector_store_data = config_dict.get("vector_store", {}).copy()

        # YAML에서 로드된 중첩 구조 유지하면서 필요한 필드만 평면화
        store_type = vector_store_data.get("store_type", "faiss")

        # 각 저장소별 중첩 설정을 attributes로 보존
        if "faiss" in vector_store_data:
            vector_store_data["faiss"] = vector_store_data["faiss"]  # 중첩 구조 보존
        if "chromadb" in vector_store_data:
            vector_store_data["chromadb"] = vector_store_data[
                "chromadb"
            ]  # 중첩 구조 보존
        if "simple" in vector_store_data:
            vector_store_data["simple"] = vector_store_data["simple"]  # 중첩 구조 보존

        # 기본값 설정
        if "store_type" not in vector_store_data:
            vector_store_data["store_type"] = store_type

        # 임베딩에 저장 경로 추가
        if "save_directory" not in embedding_data:
            embedding_data["save_directory"] = (
                "./data/processed/vector_store/embeddings"
            )

        # 각 설정 클래스 생성
        try:
            llm_config = LLMConfig(**llm_data)
            logger.info("✅ LLMConfig created successfully")
        except Exception as e:
            logger.error(f"❌ LLMConfig creation failed: {e}")
            llm_config = LLMConfig()

        try:
            embedding_config = EmbeddingConfig(**embedding_data)
            logger.info("✅ EmbeddingConfig created successfully")
        except Exception as e:
            logger.error(f"❌ EmbeddingConfig creation failed: {e}")
            embedding_config = EmbeddingConfig()

        try:
            paths_config = PathConfig(**paths_data)
            logger.info("✅ PathConfig created successfully")
        except Exception as e:
            logger.error(f"❌ PathConfig creation failed: {e}")
            paths_config = PathConfig()

        try:
            vector_store_config = VectorStoreConfig(**vector_store_data)
            logger.info("✅ VectorStoreConfig created successfully")
        except Exception as e:
            logger.error(f"❌ VectorStoreConfig creation failed: {e}")
            vector_store_config = VectorStoreConfig()

        print()
        # 나머지 설정들
        graph_config = GraphConfig(**config_dict.get("graph", {}))
        qa_config = QAConfig(**config_dict.get("qa", {}))
        system_config = SystemConfig(**config_dict.get("system", {}))

        # 메타데이터
        meta_fields = {
            "version": config_dict.get("version", "1.0.0"),
            "config_source": config_dict.get("config_source", ConfigSource.DEFAULT),
            "last_updated": config_dict.get("last_updated"),
        }

        return GraphRAGConfig(
            llm=llm_config,
            embedding=embedding_config,
            graph=graph_config,
            qa=qa_config,
            system=system_config,
            paths=paths_config,
            vector_store=vector_store_config,
            **meta_fields,
        )

    def _validate_config(self) -> None:
        """설정 검증"""

        errors = []
        warnings = []

        # LLM 설정 검증
        if not self.config.llm.api_key:
            warnings.append(
                f"No API key found for LLM provider: {self.config.llm.provider}"
            )

        if self.config.llm.temperature < 0 or self.config.llm.temperature > 1:
            errors.append("LLM temperature must be between 0 and 1")

        # 🆕 로컬 모델 경로 검증
        if self.config.llm.provider == "huggingface_local":
            if not self.config.llm.model_path:
                errors.append(
                    "Local model path is required for huggingface_local provider"
                )
            elif not Path(self.config.llm.model_path).exists():
                warnings.append(
                    f"Local model path not found: {self.config.llm.model_path}"
                )

        # 그래프 경로 검증
        if self.config.graph.unified_graph_path:
            graph_path = Path(self.config.graph.unified_graph_path)
            if not graph_path.exists():
                warnings.append(f"Unified graph file not found: {graph_path}")

        if self.config.graph.vector_store_path:
            vector_path = Path(self.config.graph.vector_store_path)
            if not vector_path.exists():
                warnings.append(f"Vector store directory not found: {vector_path}")

        # QA 설정 검증
        if (
            self.config.qa.min_relevance_score < 0
            or self.config.qa.min_relevance_score > 1
        ):
            errors.append("QA relevance score must be between 0 and 1")

        # 에러가 있으면 예외 발생
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"- {err}" for err in errors
            )
            raise ValueError(error_msg)

        # 경고 출력
        if warnings:
            for warning in warnings:
                logger.warning(f"⚠️ Config warning: {warning}")

    def update_config(self, **kwargs) -> None:
        """런타임에 설정 업데이트"""

        logger.info("📝 Updating configuration at runtime...")

        # 중첩된 설정 업데이트 지원
        config_dict = asdict(self.config)

        def update_nested(base: Dict, updates: Dict, path: str = "") -> None:
            for key, value in updates.items():
                full_path = f"{path}.{key}" if path else key

                if "." in key:
                    # 중첩된 키 (예: "llm.temperature")
                    parts = key.split(".", 1)
                    section, sub_key = parts[0], parts[1]

                    if section not in base:
                        base[section] = {}

                    update_nested(base[section], {sub_key: value}, section)
                else:
                    base[key] = value
                    logger.info(f"   Updated {full_path} = {value}")

        update_nested(config_dict, kwargs)

        # 새로운 설정으로 교체
        self.config = self._dict_to_config(config_dict)
        self.config.config_source = ConfigSource.RUNTIME

        # 업데이트된 설정 재검증
        self._validate_config()

    def get_llm_config(self) -> Dict[str, Any]:
        """LLM 설정을 LangChain 호환 형태로 반환"""
        if self.config.llm.provider == "huggingface_local":
            return {
                "model_path": self.config.llm.model_path,
                "device_map": self.config.llm.device_map,
                "torch_dtype": self.config.llm.torch_dtype,
                "max_new_tokens": self.config.llm.max_new_tokens,
                "temperature": self.config.llm.temperature,
                "trust_remote_code": self.config.llm.trust_remote_code,
                "load_in_8bit": self.config.llm.load_in_8bit,
                "load_in_4bit": self.config.llm.load_in_4bit,
            }
        llm_config = {
            "model": self.config.llm.model_name,
            "temperature": self.config.llm.temperature,
            "timeout": self.config.llm.timeout,
        }

        if self.config.llm.api_key:
            # 제공자별 API 키 설정
            if self.config.llm.provider == "openai":
                llm_config["openai_api_key"] = self.config.llm.api_key
            elif self.config.llm.provider == "anthropic":
                llm_config["anthropic_api_key"] = self.config.llm.api_key
            # 환경 변수로도 설정 (LangChain이 자동으로 읽음)
            if self.config.llm.provider == "openai":
                os.environ["OPENAI_API_KEY"] = self.config.llm.api_key
            elif self.config.llm.provider == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = self.config.llm.api_key

        if self.config.llm.api_base:
            llm_config["base_url"] = self.config.llm.api_base

        if self.config.llm.max_tokens:
            llm_config["max_tokens"] = self.config.llm.max_tokens

        return llm_config

    def get_retriever_config(self) -> Dict[str, Any]:
        """리트리버 설정 반환"""
        return {
            "unified_graph_path": self.config.graph.unified_graph_path,
            "vector_store_path": self.config.graph.vector_store_path,
            "max_docs": self.config.qa.max_docs,
            "min_relevance_score": self.config.qa.min_relevance_score,
            "enable_caching": self.config.graph.cache_enabled,
        }

    def get_vector_store_config(
        self, store_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """벡터 저장소 설정을 타입별로 반환 - YAML 중첩 구조 지원"""

        if store_type is None:
            store_type = self.config.vector_store.store_type

        base_config = {
            "store_type": store_type,
            "batch_size": self.config.vector_store.batch_size,
            "persist_directory": self.config.vector_store.persist_directory,
        }

        if store_type == "faiss":
            # YAML의 vector_store.faiss 섹션에서 설정 가져오기
            faiss_config = getattr(self.config.vector_store, "faiss", {})

            # 딕셔너리인 경우와 객체인 경우 모두 처리
            if isinstance(faiss_config, dict):
                persist_dir = faiss_config.get(
                    "persist_directory",
                    self.config.vector_store.persist_directory + "/faiss",
                )
                index_type = faiss_config.get("index_type", "flat")
                distance_metric = faiss_config.get("distance_metric", "cosine")
                use_gpu = faiss_config.get("use_gpu", False)
                gpu_id = faiss_config.get("gpu_id", 0)
                gpu_memory_fraction = faiss_config.get("gpu_memory_fraction", 0.5)
            else:
                # 기본값 사용 (평면화되지 않은 경우)
                persist_dir = getattr(
                    self.config.vector_store,
                    "faiss_directory",
                    self.config.vector_store.persist_directory + "/faiss",
                )
                index_type = getattr(
                    self.config.vector_store, "faiss_index_type", "flat"
                )
                distance_metric = getattr(
                    self.config.vector_store, "faiss_distance_metric", "cosine"
                )
                use_gpu = getattr(self.config.vector_store, "use_gpu", False)
                gpu_id = getattr(self.config.vector_store, "gpu_id", 0)
                gpu_memory_fraction = getattr(
                    self.config.vector_store, "gpu_memory_fraction", 0.5
                )

            # 중복 경로 방지
            if persist_dir.endswith("/faiss/faiss"):
                persist_dir = persist_dir.replace("/faiss/faiss", "/faiss")

            base_config.update(
                {
                    "persist_directory": persist_dir,
                    "index_type": index_type,
                    "distance_metric": distance_metric,
                    "use_gpu": use_gpu,
                    "gpu_id": gpu_id,
                    "gpu_memory_fraction": gpu_memory_fraction,
                }
            )

        elif store_type == "chromadb":
            # YAML의 vector_store.chromadb 섹션에서 설정 가져오기
            chromadb_config = getattr(self.config.vector_store, "chromadb", {})

            if isinstance(chromadb_config, dict):
                persist_dir = chromadb_config.get(
                    "persist_directory",
                    self.config.vector_store.persist_directory + "/chromadb",
                )
                collection_name = chromadb_config.get(
                    "collection_name", "graphrag_embeddings"
                )
                distance_metric = chromadb_config.get("distance_metric", "cosine")
            else:
                # 기본값 사용
                persist_dir = getattr(
                    self.config.vector_store,
                    "chromadb_directory",
                    self.config.vector_store.persist_directory + "/chromadb",
                )
                collection_name = getattr(
                    self.config.vector_store,
                    "chromadb_collection_name",
                    "graphrag_embeddings",
                )
                distance_metric = getattr(
                    self.config.vector_store, "chromadb_distance_metric", "cosine"
                )

            # 중복 경로 방지
            if persist_dir.endswith("/chromadb/chromadb"):
                persist_dir = persist_dir.replace("/chromadb/chromadb", "/chromadb")

            base_config.update(
                {
                    "persist_directory": persist_dir,
                    "collection_name": collection_name,
                    "distance_metric": distance_metric,
                }
            )

        elif store_type == "simple":
            # YAML의 vector_store.simple 섹션에서 설정 가져오기
            simple_config = getattr(self.config.vector_store, "simple", {})

            if isinstance(simple_config, dict):
                persist_dir = simple_config.get(
                    "persist_directory",
                    self.config.vector_store.persist_directory + "/simple",
                )
            else:
                # 기본값 사용
                persist_dir = getattr(
                    self.config.vector_store,
                    "simple_directory",
                    self.config.vector_store.persist_directory + "/simple",
                )

            # 중복 경로 방지
            if persist_dir.endswith("/simple/simple"):
                persist_dir = persist_dir.replace("/simple/simple", "/simple")

            base_config.update(
                {
                    "persist_directory": persist_dir,
                }
            )

        return base_config

    def get_embeddings_config(self) -> Dict[str, Any]:
        """임베딩 설정 반환 (저장 경로 포함)"""
        return {
            "model_name": self.config.embedding.model_name,
            "device": self.config.embedding.device,
            "batch_size": self.config.embedding.batch_size,
            "max_length": self.config.embedding.max_length,
            "cache_dir": self.config.embedding.cache_dir,
            "save_directory": self.config.embedding.save_directory,
            "model_type": self.config.embedding.model_type,
        }

    def get_all_directories(self) -> List[str]:
        """생성해야 할 모든 디렉토리 목록 반환"""
        return [
            self.config.paths.data_dir,
            self.config.paths.processed_dir,
            self.config.paths.vector_store_root,
            self.config.paths.vector_store_embeddings,
            self.config.paths.vector_store_faiss,
            self.config.paths.vector_store_chromadb,
            self.config.paths.vector_store_simple,
            self.config.paths.cache_dir,
            self.config.paths.embeddings_cache,
            self.config.paths.query_cache,
            self.config.paths.logs_dir,
        ]

    def create_directories(self) -> None:
        """필요한 디렉토리들 생성"""
        from pathlib import Path

        directories = self.get_all_directories()

        for directory in directories:
            if directory:  # 빈 문자열 체크
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f"📁 Created directory: {directory}")

        logger.info(f"✅ Created {len(directories)} directories")

    def save_config(self, file_path: Optional[str] = None) -> None:
        """현재 설정을 파일로 저장"""

        output_path = Path(file_path) if file_path else self.config_file
        if not output_path:
            output_path = Path("graphrag_config.yaml")

        config_dict = asdict(self.config)

        # 민감한 정보 제거 (저장시)
        if "llm" in config_dict and "api_key" in config_dict["llm"]:
            config_dict["llm"]["api_key"] = "***REDACTED***"

        logger.info(f"💾 Saving config to: {output_path}")

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                if output_path.suffix.lower() in [".yml", ".yaml"]:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)

            logger.info("✅ Config saved successfully")

        except Exception as e:
            logger.error(f"❌ Failed to save config: {e}")
            raise

    def get_config_summary(self) -> str:
        """설정 요약 정보 반환"""

        summary = [
            f"GraphRAG Configuration Summary",
            f"================================",
            f"LLM: {self.config.llm.provider} / {self.config.llm.model_name}",
            f"API Key: {'✅ Set' if self.config.llm.api_key else '❌ Missing'}",
            f"Embedding Model: {self.config.embedding.model_name}",
            f"Graph Path: {self.config.graph.unified_graph_path or 'Not set'}",
            f"Vector Store: {self.config.graph.vector_store_path or 'Not set'}",
            f"Log Level: {self.config.system.log_level}",
            f"Config Source: {self.config.config_source.name}",
        ]

        return "\n".join(summary)


# 편의 함수들
def load_config(
    config_file: Optional[str] = None, env_file: Optional[str] = None
) -> GraphRAGConfigManager:
    """설정 매니저 생성 및 로딩"""
    return GraphRAGConfigManager(config_file=config_file, env_file=env_file)


def create_sample_config(file_path: str = "graphrag_config.yaml") -> None:
    """샘플 설정 파일 생성"""

    sample_config = {
        "llm": {
            # 🆕 로컬 모델 설정 예시 추가
            "provider": "huggingface_local",  # 또는 "openai", "anthropic"
            "model_path": "/DATA/MODELS/models--meta-llama--Llama-3.1-8B-Instruct",
            "device_map": "auto",
            "torch_dtype": "bfloat16",
            "temperature": 0.1,
            "max_new_tokens": 2048,
            # API 기반 대안 설정
            # "provider": "openai",
            # "model_name": "gpt-4",
            # "api_key": "${OPENAI_API_KEY}",
        },
        "embedding": {"model_name": "auto", "device": "auto", "batch_size": 32},
        "graph": {
            "unified_graph_path": "./graphs/unified/unified_knowledge_graph.json",
            "vector_store_path": "./graphs/embeddings",
            "cache_enabled": True,
        },
        "qa": {
            "chain_type": "retrieval_qa",
            "max_docs": 10,
            "min_relevance_score": 0.3,
            "enable_memory": False,
        },
        "system": {"log_level": "INFO", "verbose": False, "max_workers": 4},
    }

    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)

    print(f"✅ Sample config created: {file_path}")


def main():
    """ConfigManager 테스트"""

    print("🧪 Testing GraphRAGConfigManager...")

    try:
        # 1. 기본 설정으로 초기화
        config_manager = GraphRAGConfigManager(auto_load=False)
        print("✅ Basic initialization successful")

        # 2. 설정 업데이트
        config_manager.update_config(
            **{
                "llm.temperature": 0.2,
                "llm.model_name": "gpt-4",
                "qa.max_docs": 15,
                "system.verbose": True,
            }
        )
        print("✅ Configuration update successful")

        # 3. LLM 설정 조회
        llm_config = config_manager.get_llm_config()
        print(f"🤖 LLM Config: {llm_config}")

        # 4. 설정 요약
        print(f"\n📋 Configuration Summary:")
        print(config_manager.get_config_summary())

        # 5. 샘플 설정 파일 생성
        create_sample_config("test_config.yaml")

        print(f"\n✅ GraphRAGConfigManager test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
