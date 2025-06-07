"""
GraphRAG 메인 파이프라인
Main GraphRAG Pipeline Integration

전체 GraphRAG 시스템의 통합 인터페이스
- 설정 관리 및 시스템 초기화
- 임베딩 생성 및 벡터 저장소 구축
- 쿼리 분석 및 서브그래프 추출
- 컨텍스트 직렬화 및 LLM 연동
- 통합 QA 인터페이스 제공
"""

import os
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

# GraphRAG 컴포넌트들
try:
    from .config_manager import GraphRAGConfigManager
    from .query_analyzer import QueryAnalyzer, QueryAnalysisResult
    from .embeddings import (
        MultiNodeEmbedder,
        VectorStoreManager,
        create_embedding_model,
        is_ready as embeddings_ready,
    )
    from .embeddings.subgraph_extractor import SubgraphExtractor, SubgraphResult
    from .embeddings.context_serializer import ContextSerializer, SerializedContext
except ImportError as e:
    warnings.warn(f"Some GraphRAG components not available: {e}")

# LLM 관련
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        GenerationConfig,
        BitsAndBytesConfig,
    )

    _transformers_available = True
except ImportError:
    _transformers_available = False
    warnings.warn("Transformers not available. Local LLM will not work.")

# 로깅 설정
logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """파이프라인 상태"""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"


@dataclass
class PipelineState:
    """파이프라인 상태 정보"""

    status: PipelineStatus
    components_loaded: Dict[str, bool]
    last_error: Optional[str]
    initialization_time: Optional[float]
    total_queries_processed: int
    last_query_time: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {**asdict(self), "status": self.status.value}


@dataclass
class QAResult:
    """QA 결과 클래스"""

    query: str
    answer: str
    subgraph_result: Optional[SubgraphResult]
    serialized_context: Optional[SerializedContext]
    query_analysis: Optional[QueryAnalysisResult]
    processing_time: float
    confidence_score: float
    source_nodes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "query": self.query,
            "answer": self.answer,
            "processing_time": self.processing_time,
            "confidence_score": self.confidence_score,
            "source_nodes": self.source_nodes,
            "subgraph_nodes": (
                self.subgraph_result.total_nodes if self.subgraph_result else 0
            ),
            "subgraph_edges": (
                self.subgraph_result.total_edges if self.subgraph_result else 0
            ),
            "query_complexity": (
                self.query_analysis.complexity.value
                if self.query_analysis
                else "unknown"
            ),
            "query_type": (
                self.query_analysis.query_type.value
                if self.query_analysis
                else "unknown"
            ),
        }


class LocalLLMManager:
    """로컬 LLM 관리 클래스"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: LLM 설정 딕셔너리
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.is_loaded = False

    def load_model(self) -> None:
        """모델 로드"""
        if not _transformers_available:
            raise ImportError("Transformers not available for local LLM")

        model_path = self.config.get("model_path")
        # if not model_path or not Path(model_path).exists():
        #     raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"🤖 Loading local LLM: {model_path}")

        # try:
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=self.config.get("trust_remote_code", True)
        )

        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 양자화 설정 (메모리 절약)
        quantization_config = None
        if self.config.get("load_in_4bit", False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.get("load_in_8bit", False):
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # 모델 로드
        model_kwargs = {
            "trust_remote_code": self.config.get("trust_remote_code", True),
            "device_map": self.config.get("device_map", "auto"),
            "torch_dtype": getattr(torch, self.config.get("torch_dtype", "bfloat16")),
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        # 생성 설정
        self.generation_config = GenerationConfig(
            temperature=self.config.get("temperature", 0.1),
            max_new_tokens=self.config.get("max_new_tokens", 2048),
            do_sample=self.config.get("do_sample", True),
            top_p=self.config.get("top_p", 0.9),
            top_k=self.config.get("top_k", 50),
            repetition_penalty=self.config.get("repetition_penalty", 1.1),
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        self.is_loaded = True
        logger.info("✅ Local LLM loaded successfully")

    # except Exception as e:
    #     logger.error(f"❌ Failed to load local LLM: {e}")
    #     raise

    def generate(self, prompt: str, max_length: Optional[int] = None) -> str:
        """텍스트 생성"""
        if not self.is_loaded:
            self.load_model()

        # try:
        # 프롬프트 토크나이징
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding=True,
        )

        # 디바이스로 이동
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # attention_mask 확인
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        # YAML 설정 기반 생성 설정 (기본값은 안전하게)
        generation_config = GenerationConfig(
            temperature=max(
                0.01, min(2.0, self.config.get("temperature", 0.1))
            ),  # 안전 범위
            max_new_tokens=min(
                max_length or self.config.get("max_new_tokens", 512),
                self.config.get("max_new_tokens", 512),
            ),
            do_sample=self.config.get("do_sample", True),  # YAML 설정 우선
            top_p=max(0.1, min(1.0, self.config.get("top_p", 0.9))),  # 안전 범위
            top_k=max(1, min(100, self.config.get("top_k", 50))),  # 안전 범위
            repetition_penalty=max(
                1.0, min(2.0, self.config.get("repetition_penalty", 1.1))
            ),  # 안전 범위
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        logger.info(
            f"🔍 Generation config: temp={generation_config.temperature}, "
            f"do_sample={generation_config.do_sample}, "
            f"max_tokens={generation_config.max_new_tokens}"
        )

        # 첫 번째 시도: YAML 설정대로
        with torch.no_grad():
            # try:
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config,
                use_cache=True,
            )

            # except RuntimeError as cuda_error:
            #     if "CUDA" in str(cuda_error) or "device-side assert" in str(cuda_error):
            #         logger.warning(f"⚠️ CUDA error with YAML settings: {cuda_error}")
            #         logger.info("🔄 Retrying with safer settings...")

            #         # 두 번째 시도: 안전한 설정으로 재시도
            #         return self._retry_with_safe_settings(inputs, max_length)
            #     else:
            #         raise

        # 생성된 텍스트 디코딩
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        generated_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        result = generated_text.strip()
        if not result:
            result = "답변을 생성할 수 없었습니다."

        logger.info(f"✅ Generated {len(result)} characters with YAML settings")
        return result

    # except Exception as e:
    #     logger.error(f"❌ Text generation failed: {e}")
    #     return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"

    def unload_model(self) -> None:
        """모델 언로드 (메모리 해제)"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_loaded = False
        logger.info("🗑️ Local LLM unloaded")


class GraphRAGPipeline:
    """GraphRAG 메인 파이프라인"""

    def __init__(
        self,
        config_file: str = "graphrag_config.yaml",
        env_file: Optional[str] = None,
        auto_setup: bool = False,
    ):
        """
        Args:
            config_file: 설정 파일 경로
            env_file: 환경변수 파일 경로
            auto_setup: 자동 초기화 여부
        """
        self.config_file = Path(config_file)
        self.env_file = Path(env_file) if env_file else None

        # 상태 관리
        self.state = PipelineState(
            status=PipelineStatus.UNINITIALIZED,
            components_loaded={},
            last_error=None,
            initialization_time=None,
            total_queries_processed=0,
            last_query_time=None,
        )

        # 컴포넌트들
        self.config_manager = None
        self.query_analyzer = None
        self.embedder = None
        self.vector_store = None
        self.subgraph_extractor = None
        self.context_serializer = None
        self.llm_manager = None

        # 캐시
        self.query_cache = {}
        self.embeddings_loaded = False

        logger.info("🚀 GraphRAG Pipeline initialized")

        if auto_setup:
            self.setup()

    def setup(self) -> None:
        """시스템 전체 초기화"""
        logger.info("🔧 Setting up GraphRAG Pipeline...")
        self.state.status = PipelineStatus.INITIALIZING

        start_time = time.time()

        try:
            # 1. 설정 관리자 초기화
            self._setup_config_manager()

            # 2. 쿼리 분석기 초기화
            self._setup_query_analyzer()

            # 3. 임베딩 시스템 확인
            self._check_embeddings_system()

            # 4. 컨텍스트 직렬화기 초기화
            self._setup_context_serializer()

            # 5. LLM 관리자 초기화 (지연 로딩)
            self._setup_llm_manager()

            # 6. 시스템 상태 업데이트
            self.state.initialization_time = time.time() - start_time
            self.state.status = PipelineStatus.READY
            self.state.last_error = None

            logger.info(
                f"✅ GraphRAG Pipeline setup completed ({self.state.initialization_time:.2f}s)"
            )

        except Exception as e:
            self.state.status = PipelineStatus.ERROR
            self.state.last_error = str(e)
            logger.error(f"❌ Pipeline setup failed: {e}")
            raise

    def _setup_config_manager(self) -> None:
        """설정 관리자 초기화"""
        logger.info("📋 Setting up config manager...")

        self.config_manager = GraphRAGConfigManager(
            config_file=str(self.config_file),
            env_file=str(self.env_file) if self.env_file else None,
        )

        self.state.components_loaded["config_manager"] = True
        logger.info("✅ Config manager ready")

    def _setup_query_analyzer(self) -> None:
        """쿼리 분석기 초기화"""
        logger.info("🔍 Setting up query analyzer...")

        self.query_analyzer = QueryAnalyzer()
        self.state.components_loaded["query_analyzer"] = True
        logger.info("✅ Query analyzer ready")

    def _check_embeddings_system(self) -> None:
        """임베딩 시스템 확인"""
        logger.info("🔍 Checking embeddings system...")

        if not embeddings_ready():
            logger.warning("⚠️ Embeddings system not fully ready")
            self.state.components_loaded["embeddings"] = False
        else:
            self.state.components_loaded["embeddings"] = True
            logger.info("✅ Embeddings system ready")

    def _setup_context_serializer(self) -> None:
        """컨텍스트 직렬화기 초기화"""
        logger.info("📝 Setting up context serializer...")

        self.context_serializer = ContextSerializer()
        self.state.components_loaded["context_serializer"] = True
        logger.info("✅ Context serializer ready")

    def _setup_llm_manager(self) -> None:
        """LLM 관리자 초기화 (지연 로딩)"""
        logger.info("🤖 Setting up LLM manager...")

        llm_config = self.config_manager.get_llm_config()

        if llm_config.get("model_path"):  # 로컬 모델
            self.llm_manager = LocalLLMManager(llm_config)
            self.state.components_loaded["llm_manager"] = True
            logger.info("✅ Local LLM manager ready (model will be loaded on demand)")
        else:
            logger.warning("⚠️ No local LLM configuration found")
            self.state.components_loaded["llm_manager"] = False

    # def _ensure_embeddings_loaded(self) -> None:
    #     """임베딩 시스템 로드 확인 - 벡터 저장소 연동 개선"""
    #     if self.embeddings_loaded:
    #         return

    #     logger.info("📥 Loading embeddings system...")

    #     # 설정 가져오기
    #     config = self.config_manager.config

    #     # 통합 그래프 파일 확인
    #     unified_graph_path = config.graph.unified_graph_path
    #     if not Path(unified_graph_path).exists():
    #         raise FileNotFoundError(f"Unified graph not found: {unified_graph_path}")

    #     # 벡터 저장소 경로 확인
    #     vector_store_root = config.graph.vector_store_path
    #     if not Path(vector_store_root).exists():
    #         logger.warning(f"Vector store root not found: {vector_store_root}")
    #         logger.info("💡 Run build_embeddings() first to create vector store")
    #         return

    #     # 벡터 저장소 설정 가져오기
    #     vector_store_config = self.config_manager.get_vector_store_config()
    #     store_directory = vector_store_config["persist_directory"]

    #     logger.info(f"📂 Vector store directory: {store_directory}")
    #     logger.info(f"📂 Store type: {vector_store_config['store_type']}")

    #     # 벡터 저장소 로드 또는 생성
    #     try:
    #         from .embeddings.vector_store_manager import create_vector_store_from_config

    #         self.vector_store = create_vector_store_from_config(
    #             config_manager=self.config_manager
    #         )

    #         # 벡터 저장소가 비어있으면 임베딩에서 로드 시도
    #         if self.vector_store.store.total_vectors == 0:
    #             embeddings_dir = config.paths.vector_store.embeddings

    #             if (
    #                 Path(embeddings_dir).exists()
    #                 and (Path(embeddings_dir) / "embeddings.npy").exists()
    #             ):
    #                 logger.info("🔄 Loading from saved embeddings...")
    #                 self.vector_store.load_from_saved_embeddings(vector_store_root)
    #             else:
    #                 logger.warning(f"No vector data found in: {store_directory}")
    #                 logger.warning(f"No embeddings found in: {embeddings_dir}")
    #                 logger.info("💡 Run build_embeddings() first")
    #                 return

    #         logger.info(
    #             f"✅ Vector store loaded: {self.vector_store.store.total_vectors:,} vectors"
    #         )

    #         # SubgraphExtractor 초기화 - VectorStoreManager 인스턴스 직접 전달
    #         self.subgraph_extractor = SubgraphExtractor(
    #             unified_graph_path=unified_graph_path,
    #             vector_store_path=store_directory,  # 경로만 전달
    #             embedding_model=config.embeddings.model_name,
    #             device=config.embeddings.device,
    #         )

    #         # SubgraphExtractor의 벡터 저장소를 수동으로 설정
    #         self.subgraph_extractor.vector_store = self.vector_store

    #         self.embeddings_loaded = True
    #         logger.info("✅ Embeddings system loaded successfully")

    #     except Exception as e:
    #         logger.error(f"❌ Failed to load embeddings system: {e}")
    #         logger.error(f"   Store directory: {store_directory}")
    #         logger.error(f"   Store exists: {Path(store_directory).exists()}")

    #         # 디버깅 정보
    #         if Path(store_directory).exists():
    #             files = list(Path(store_directory).glob("*"))
    #             logger.error(f"   Files in store: {[f.name for f in files]}")

    #         raise
    def _ensure_embeddings_loaded(self) -> None:
        """임베딩 시스템 로드 확인 - 모든 model_type 지원"""
        if self.embeddings_loaded:
            return

        logger.info("📥 Loading embeddings system...")

        # 설정 가져오기
        config = self.config_manager.config

        # 통합 그래프 파일 확인
        unified_graph_path = config.graph.unified_graph_path
        if not Path(unified_graph_path).exists():
            raise FileNotFoundError(f"Unified graph not found: {unified_graph_path}")

        # 벡터 저장소 경로 확인
        vector_store_root = config.paths.vector_store_root
        if not Path(vector_store_root).exists():
            Path(vector_store_root).mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created vector store directory: {vector_store_root}")

        # 벡터 저장소 설정 가져오기
        vector_config = self.config_manager.get_vector_store_config()

        # VectorStoreManager 초기화
        from .embeddings.vector_store_manager import VectorStoreManager

        self.vector_store = VectorStoreManager(
            store_type=vector_config["store_type"],
            persist_directory=vector_config["persist_directory"],
            **{
                k: v
                for k, v in vector_config.items()
                if k not in ["store_type", "persist_directory"]
            },
        )

        # 임베딩 설정 가져오기 (모든 타입 지원)
        embedding_config = self.config_manager.get_embeddings_config()  # ✅ 유연함

        from .embeddings import create_embedding_model

        embedder_model = create_embedding_model(
            model_name=embedding_config["model_name"],  # ✅ 타입 무관
            device=embedding_config["device"],  # ✅ 타입 무관
            cache_dir=embedding_config["cache_dir"],  # ✅ 타입 무관
        )

        # MultiNodeEmbedder 초기화
        from .embeddings.multi_node_embedder import MultiNodeEmbedder

        self.embedder = MultiNodeEmbedder(
            unified_graph_path=config.graph.unified_graph_path,
            embedding_model=embedder_model,
            vector_store=self.vector_store,
            batch_size=embedding_config["batch_size"],  # ✅ 타입 무관
        )

        # SubgraphExtractor 초기화 (새로운 설정 구조)
        embedding_config = self.config_manager.get_embeddings_config()
        from .embeddings.subgraph_extractor import SubgraphExtractor

        self.subgraph_extractor = SubgraphExtractor(
            unified_graph_path=config.graph.unified_graph_path,
            vector_store_path=config.paths.vector_store_root,  # ✅ 경로 전달
            embedding_model=embedding_config["model_name"],
            device=embedding_config["device"],
        )

        self.embeddings_loaded = True
        logger.info("✅ Embeddings system loaded with flexible config structure")

    def build_embeddings(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """임베딩 생성 및 벡터 저장소 구축 - 새로운 경로 구조 사용"""
        logger.info("🏗️ Building embeddings and vector store...")

        if self.state.status != PipelineStatus.READY:
            raise RuntimeError("Pipeline not ready. Call setup() first.")

        config = self.config_manager.config

        # 필요한 디렉토리 생성
        self.config_manager._create_directories()

        # 임베딩 생성기 초기화 (설정 관리자 사용)
        from .embeddings.multi_node_embedder import create_embedder_with_config

        embedding_config = self.config_manager.get_embeddings_config()
        self.embedder = create_embedder_with_config(
            unified_graph_path=config.graph.unified_graph_path,
            config_manager=self.config_manager,
            device=embedding_config["device"],
        )

        # 벡터 저장소 설정 가져오기
        vector_store_config = self.config_manager.get_vector_store_config()

        # 임베딩 생성 (새로운 경로 구조로)
        embedding_results, saved_files = self.embedder.run_full_pipeline(
            output_dir=config.graph.vector_store_path,  # 루트 디렉토리
            use_cache=not force_rebuild,
            show_progress=True,
            vector_store_config=vector_store_config,
        )

        # 벡터 저장소 구축 (설정 관리자 사용)
        from .embeddings.vector_store_manager import create_vector_store_from_config

        self.vector_store = create_vector_store_from_config(
            config_manager=self.config_manager,
            store_type=vector_store_config["store_type"],
        )

        # 임베딩 결과로부터 벡터 저장소 로드
        self.vector_store.load_from_embeddings(
            embedding_results,
            embeddings_dir=config.paths.vector_store.embeddings,
        )

        # 통계 반환
        total_nodes = sum(len(results) for results in embedding_results.values())

        result = {
            "total_embeddings": total_nodes,
            "embeddings_by_type": {k: len(v) for k, v in embedding_results.items()},
            "saved_files": {k: str(v) for k, v in saved_files.items()},
            "vector_store_info": self.vector_store.get_store_info(),
            "path_structure": {
                "vector_store_root": config.graph.vector_store_path,
                "embeddings_dir": config.paths.vector_store.embeddings,
                "store_specific_dir": vector_store_config["persist_directory"],
            },
        }

        logger.info(f"✅ Embeddings built: {total_nodes:,} nodes")
        logger.info(f"📂 Structure created:")
        logger.info(f"   Root: {config.graph.vector_store_path}")
        logger.info(f"   Embeddings: {config.paths.vector_store.embeddings}")
        logger.info(f"   Vector Store: {vector_store_config['persist_directory']}")

        return result

    def ask(self, query: str, return_context: bool = False) -> Union[str, QAResult]:
        """메인 QA 인터페이스

        Args:
            query: 사용자 질문
            return_context: 상세 컨텍스트 반환 여부

        Returns:
            답변 문자열 또는 QAResult 객체
        """
        if self.state.status != PipelineStatus.READY:
            raise RuntimeError("Pipeline not ready. Call setup() first.")

        start_time = time.time()
        self.state.status = PipelineStatus.PROCESSING

        # try:
        logger.info(f"❓ Processing query: '{query[:50]}...'")

        # 1. 캐시 확인
        if query in self.query_cache:
            logger.info("✅ Cache hit")
            cached_result = self.query_cache[query]
            if return_context:
                return cached_result
            else:
                return cached_result.answer

        # 2. 임베딩 시스템 로드
        self._ensure_embeddings_loaded()

        # 3. 쿼리 분석
        query_analysis = self.query_analyzer.analyze(query)

        # 4. 서브그래프 추출
        subgraph_result = self.subgraph_extractor.extract_subgraph(
            query=query, query_analysis=query_analysis
        )

        # 5. 컨텍스트 직렬화
        serialized_context = self.context_serializer.serialize(
            subgraph_result=subgraph_result, query_analysis=query_analysis
        )

        # 6. LLM으로 답변 생성
        context_text = getattr(serialized_context, "main_text", "") or ""
        if not isinstance(context_text, str):
            context_text = str(context_text) if context_text else "No context available"

        answer = self._generate_answer(query, context_text)
        # answer = self._generate_answer(query, serialized_context.main_text)

        # 7. 결과 구성
        processing_time = time.time() - start_time

        qa_result = QAResult(
            query=query,
            answer=answer,
            subgraph_result=subgraph_result,
            serialized_context=serialized_context,
            query_analysis=query_analysis,
            processing_time=processing_time,
            confidence_score=subgraph_result.confidence_score,
            source_nodes=list(subgraph_result.nodes.keys()),
        )

        # 8. 캐시 저장
        self.query_cache[query] = qa_result

        # 9. 상태 업데이트
        self.state.total_queries_processed += 1
        self.state.last_query_time = processing_time
        self.state.status = PipelineStatus.READY

        logger.info(f"✅ Query processed ({processing_time:.2f}s)")

        if return_context:
            return qa_result
        else:
            return answer

    # except Exception as e:
    #     self.state.status = PipelineStatus.ERROR
    #     self.state.last_error = str(e)
    #     logger.error(f"❌ Query processing failed: {e}")

    #     # 간단한 답변 반환
    #     fallback_answer = f"죄송합니다. 질문 처리 중 오류가 발생했습니다: {str(e)}"

    #     if return_context:
    #         return QAResult(
    #             query=query,
    #             answer=fallback_answer,
    #             subgraph_result=None,
    #             serialized_context=None,
    #             query_analysis=None,
    #             processing_time=time.time() - start_time,
    #             confidence_score=0.0,
    #             source_nodes=[],
    #         )
    #     else:
    #         return fallback_answer

    def _generate_answer(self, query: str, context: str) -> str:
        """LLM으로 답변 생성 (안전한 버전)"""

        # 입력 검증
        if not isinstance(query, str):
            logger.error(f"❌ Query is not a string: {type(query)} - {query}")
            query = str(query) if query is not None else "Unknown query"

        if not isinstance(context, str):
            logger.error(f"❌ Context is not a string: {type(context)} - {context}")
            context = str(context) if context is not None else "No context available"

        # 빈 문자열 처리
        if not query.strip():
            query = "Unknown query"

        if not context.strip():
            context = "No context available"

        # try:
        # 프롬프트 구성
        prompt = self._build_qa_prompt(query, context)

        # 프롬프트 검증
        if not isinstance(prompt, str):
            logger.error(f"❌ Prompt is not a string: {type(prompt)}")
            prompt = f"질문: {query}\n답변을 생성해주세요."

        logger.debug(f"🔍 Generated prompt length: {len(prompt)}")

        # LLM으로 답변 생성
        if self.llm_manager and self.llm_manager.config.get("model_path"):
            # 로컬 LLM 사용
            # try:
            logger.info("🤖 Generating answer with local LLM...")
            answer = self.llm_manager.generate(prompt, max_length=1000)

            # 답변 검증
            if not isinstance(answer, str):
                logger.error(f"❌ LLM returned non-string: {type(answer)}")
                return f"LLM 응답 형식 오류가 발생했습니다."

            if not answer.strip():
                logger.warning("⚠️ LLM returned empty response")
                return "죄송합니다. 적절한 답변을 생성할 수 없었습니다."

            return answer.strip()

        # except Exception as e:
        #     logger.error(f"❌ Local LLM generation failed: {e}")
        #     logger.debug(f"❌ Prompt that caused error: {prompt[:200]}...")
        #     return f"로컬 LLM 오류: {str(e)}"
        else:
            # API LLM 폴백 또는 기본 답변
            return (
                "죄송합니다. LLM이 설정되지 않았습니다. 로컬 모델 경로를 확인해주세요."
            )

    # except Exception as e:
    #     logger.error(f"❌ Answer generation failed: {e}")
    #     return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

    def _build_qa_prompt(self, query: str, context: str) -> str:
        """QA 프롬프트 구성 (안전한 버전)"""

        # 입력 검증 및 정리
        query = str(query).strip() if query else "Unknown query"
        context = str(context).strip() if context else "No context available"

        # 컨텍스트 길이 제한 (토크나이저 한계 고려)
        max_context_length = 3000  # 안전한 길이
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            logger.warning(f"⚠️ Context truncated to {max_context_length} chars")

        prompt_template = """다음은 학술 연구 문헌 분석 시스템입니다. 제공된 컨텍스트를 바탕으로 사용자의 질문에 정확하고 유용한 답변을 제공하세요.

    **컨텍스트:**
    {context}

    **질문:** {query}

    **답변 가이드라인:**
    1. 컨텍스트에 기반하여 정확한 정보를 제공하세요
    2. 한국어와 영어를 적절히 혼용하여 답변하세요
    3. 구체적인 논문, 저자, 연구 결과를 언급하세요
    4. 불확실한 내용은 명시하세요
    5. 답변은 500자 이내로 간결하게 작성하세요

    **답변:**"""

        try:
            formatted_prompt = prompt_template.format(context=context, query=query)

            # 프롬프트 길이 확인
            if len(formatted_prompt) > 8000:  # 토크나이저 한계 고려
                logger.warning(
                    f"⚠️ Prompt too long: {len(formatted_prompt)} chars, truncating..."
                )
                # 컨텍스트를 더 줄임
                shorter_context = context[:1500] + "..."
                formatted_prompt = prompt_template.format(
                    context=shorter_context, query=query
                )

            return formatted_prompt

        except Exception as e:
            logger.error(f"❌ Prompt formatting failed: {e}")
            # 최소한의 안전한 프롬프트
            return f"질문: {query}\n\n위 질문에 대해 답변해주세요."

    def get_subgraph(self, query: str) -> Optional[SubgraphResult]:
        """서브그래프만 추출 (LLM 없이)"""
        try:
            self._ensure_embeddings_loaded()

            query_analysis = self.query_analyzer.analyze(query)
            subgraph_result = self.subgraph_extractor.extract_subgraph(
                query=query, query_analysis=query_analysis
            )

            return subgraph_result

        except Exception as e:
            logger.error(f"❌ Subgraph extraction failed: {e}")
            return None

    def batch_process(self, queries: List[str], max_workers: int = 2) -> List[QAResult]:
        """배치 쿼리 처리"""
        logger.info(f"📋 Processing {len(queries)} queries in batch...")

        results = []
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}")
            result = self.ask(query, return_context=True)
            results.append(result)

        logger.info(f"✅ Batch processing completed: {len(results)} results")
        return results

    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 확인 - 경로 정보 포함"""
        status = {
            "pipeline_state": self.state.to_dict(),
            "components": self.state.components_loaded,
            "embeddings_loaded": self.embeddings_loaded,
            "cache_size": len(self.query_cache),
            "memory_usage": self._get_memory_usage(),
        }

        # 설정 관리자 정보
        if self.config_manager:
            config = self.config_manager.config
            embedding_config = self.config_manager.get_embeddings_config()
            status["configuration"] = {
                "llm_provider": config.llm.provider,
                "embedding_model": embedding_config["model_name"],
                "vector_store_type": config.vector_store.store_type,
                "paths": {
                    "unified_graph": config.graph.unified_graph_path,
                    "vector_store_root": config.paths.vector_store_root,
                    "embeddings_dir": config.paths.vector_store.embeddings,
                    "store_directory": self.config_manager.get_vector_store_config()[
                        "persist_directory"
                    ],
                },
            }

        # 벡터 저장소 정보
        if self.vector_store:
            status["vector_store"] = self.vector_store.get_store_info()

        # LLM 상태
        if self.llm_manager:
            status["llm_loaded"] = self.llm_manager.is_loaded
            status["llm_model_path"] = self.llm_manager.config.get("model_path")

        return status

    def setup_from_config(
        self,
        config_file: str = "graphrag_config.yaml",
        auto_build_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """설정 파일로부터 완전 자동 설정"""

        logger.info(f"🚀 Setting up GraphRAG from config: {config_file}")

        # 설정 파일로 초기화
        self.config_file = Path(config_file)

        # 설정 및 시스템 초기화
        self.setup()

        setup_result = {
            "setup_completed": True,
            "config_loaded": True,
            "directories_created": True,
        }

        # 자동 임베딩 구축
        if auto_build_embeddings:
            try:
                build_result = self.build_embeddings()
                setup_result.update(
                    {
                        "embeddings_built": True,
                        "embedding_stats": build_result,
                    }
                )
            except Exception as e:
                logger.warning(f"⚠️ Auto embedding build failed: {e}")
                setup_result["embeddings_built"] = False
                setup_result["embedding_error"] = str(e)

        # 시스템 상태 확인
        status = self.get_system_status()
        setup_result["system_status"] = status

        logger.info("✅ GraphRAG setup completed!")
        return setup_result

    def rebuild_vector_store(
        self,
        new_store_type: Optional[str] = None,
        force_rebuild_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """벡터 저장소 재구축"""

        logger.info("🔄 Rebuilding vector store...")

        config = self.config_manager.config
        current_store_type = config.vector_store.store_type
        target_store_type = new_store_type or current_store_type

        result = {
            "previous_store_type": current_store_type,
            "new_store_type": target_store_type,
            "embeddings_rebuilt": False,
            "store_migrated": False,
        }

        # 임베딩 재구축 (필요시)
        if force_rebuild_embeddings:
            logger.info("🏗️ Rebuilding embeddings...")
            build_result = self.build_embeddings(force_rebuild=True)
            result["embeddings_rebuilt"] = True
            result["embedding_stats"] = build_result

        # 벡터 저장소 타입 변경 (필요시)
        if new_store_type and new_store_type != current_store_type:
            logger.info(f"🔄 Migrating from {current_store_type} to {new_store_type}")

            # 기존 벡터 저장소 로드
            self._ensure_embeddings_loaded()

            if self.vector_store:
                # 마이그레이션 수행
                new_vector_store = self.vector_store.migrate_store_type(new_store_type)
                self.vector_store = new_vector_store

                # 설정 업데이트
                self.config_manager.update_config(
                    **{"vector_store.store_type": new_store_type}
                )

                result["store_migrated"] = True
                result["new_store_info"] = self.vector_store.get_store_info()

        logger.info("✅ Vector store rebuild completed!")
        return result

    @classmethod
    def from_config_file(
        cls,
        config_file: str = "graphrag_config.yaml",
        auto_setup: bool = True,
        auto_build_embeddings: bool = False,
    ) -> "GraphRAGPipeline":
        """설정 파일로부터 파이프라인 생성 (클래스 메서드)"""

        pipeline = cls(config_file=config_file, auto_setup=False)

        if auto_setup:
            pipeline.setup_from_config(
                config_file=config_file,
                auto_build_embeddings=auto_build_embeddings,
            )

        return pipeline

    def _get_memory_usage(self) -> Dict[str, str]:
        """메모리 사용량 조회"""
        try:
            import psutil

            # 시스템 메모리
            memory = psutil.virtual_memory()

            usage = {
                "system_total": f"{memory.total / 1024**3:.1f}GB",
                "system_available": f"{memory.available / 1024**3:.1f}GB",
                "system_percent": f"{memory.percent:.1f}%",
            }

            # GPU 메모리 (가능한 경우)
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_allocated = torch.cuda.memory_allocated(0)
                gpu_cached = torch.cuda.memory_reserved(0)

                usage.update(
                    {
                        "gpu_total": f"{gpu_memory / 1024**3:.1f}GB",
                        "gpu_allocated": f"{gpu_allocated / 1024**3:.1f}GB",
                        "gpu_cached": f"{gpu_cached / 1024**3:.1f}GB",
                        "gpu_percent": f"{(gpu_allocated / gpu_memory) * 100:.1f}%",
                    }
                )

            return usage

        except ImportError:
            return {"status": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}

    def clear_cache(self) -> None:
        """캐시 정리"""
        self.query_cache.clear()
        logger.info("🗑️ Query cache cleared")

    def shutdown(self) -> None:
        """시스템 종료 및 정리"""
        logger.info("🔌 Shutting down GraphRAG Pipeline...")

        # LLM 언로드
        if self.llm_manager:
            self.llm_manager.unload_model()

        # 캐시 정리
        self.clear_cache()

        # 상태 리셋
        self.state.status = PipelineStatus.UNINITIALIZED
        self.embeddings_loaded = False

        logger.info("✅ Pipeline shutdown completed")


def create_graphrag_pipeline(
    config_file: str = "graphrag_config.yaml",
    auto_setup: bool = True,
    auto_build_embeddings: bool = False,
) -> GraphRAGPipeline:
    """GraphRAG 파이프라인 생성 편의 함수"""

    return GraphRAGPipeline.from_config_file(
        config_file=config_file,
        auto_setup=auto_setup,
        auto_build_embeddings=auto_build_embeddings,
    )


def main():
    """GraphRAG Pipeline 테스트"""
    print("🧪 Testing GraphRAG Pipeline...")

    try:
        # 1. 파이프라인 초기화
        pipeline = GraphRAGPipeline(config_file="graphrag_config.yaml", auto_setup=True)

        # 2. 시스템 상태 확인
        status = pipeline.get_system_status()
        print(f"📊 System Status:")
        print(f"   Pipeline: {status['pipeline_state']['status']}")
        print(f"   Components: {status['components']}")

        # 3. 임베딩 구축 (필요한 경우)
        if not status["embeddings_loaded"]:
            print(f"\n🏗️ Building embeddings...")
            build_result = pipeline.build_embeddings()
            print(f"✅ Built {build_result['total_embeddings']} embeddings")

        # 4. 테스트 질문
        test_queries = [
            "배터리 SoC 예측에 사용된 머신러닝 기법들은?",
            "AI 및 머신러닝 기법이 적용된 주요 task는 무엇인가요?"
            "배터리 전극 공정에서 AI가 적용될 수 있는 부분은 무엇이 있을까요?",
        ]

        print(f"\n❓ Testing queries...")
        for i, query in enumerate(test_queries[:1]):  # 첫 번째만 테스트
            print(f"\n{i+1}. {query}")

            result = pipeline.ask(query, return_context=True)

            print(f"✅ Answer: {result.answer[:200]}...")
            print(
                f"📊 Stats: {result.processing_time:.2f}s, {result.confidence_score:.3f} confidence"
            )
            print(f"📄 Sources: {len(result.source_nodes)} nodes")

        # 5. 최종 상태
        final_status = pipeline.get_system_status()
        print(f"\n📈 Final Stats:")
        print(
            f"   Queries processed: {final_status['pipeline_state']['total_queries_processed']}"
        )
        print(f"   Cache size: {final_status['cache_size']}")

        print(f"\n✅ GraphRAG Pipeline test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
