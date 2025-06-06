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
    """LLM 설정"""

    provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    streaming: bool = False
    timeout: int = 60


@dataclass
class EmbeddingConfig:
    """임베딩 설정"""

    model_name: str = "auto"
    device: str = "auto"
    batch_size: int = 32
    max_length: int = 512
    cache_dir: Optional[str] = None


@dataclass
class GraphConfig:
    """그래프 설정"""

    unified_graph_path: str = ""
    vector_store_path: str = ""
    graphs_directory: str = "./graphs"
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
    """시스템 설정"""

    log_level: str = "INFO"
    verbose: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    temp_directory: str = "./tmp"
    enable_monitoring: bool = False


@dataclass
class GraphRAGConfig:
    """GraphRAG 전체 설정"""

    # 하위 설정들
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    qa: QAConfig = field(default_factory=QAConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    # 메타데이터
    version: str = "1.0.0"
    config_source: ConfigSource = ConfigSource.DEFAULT
    last_updated: Optional[str] = None


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
        """딕셔너리를 GraphRAGConfig로 변환"""

        # 하위 설정들 생성
        llm_config = LLMConfig(**config_dict.get("llm", {}))
        embedding_config = EmbeddingConfig(**config_dict.get("embedding", {}))
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
            "provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.1,
            "max_tokens": 2000,
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
