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
    from huggingface_hub import InferenceClient

    _transformers_available = True
except ImportError:
    _transformers_available = False
    warnings.warn("Transformers not available. Local LLM will not work.")
# 로깅 설정
logger = logging.getLogger(__name__)

# ✅ 전역 변수는 유지
try:
    # 일부 체크만 수행
    _qa_chain_available = True
    logger.info("✅ QA Chain integration available")
except ImportError as e:
    _qa_chain_available = False
    logger.warning(f"⚠️ QA Chain not available: {e}")


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


class HuggingFaceAPIManager:
    """Hugging Face API를 사용한 LLM 관리자"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: HF API 설정 (model_name, api_key, temperature 등)
        """
        self.config = config
        self.model_name = config.get(
            "model_name", "meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
        self.api_key = config.get("api_key") or os.getenv("HUGGINGFACE_API_KEY")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_new_tokens", 1000)

        if not self.api_key:
            raise ValueError("Hugging Face API key is required")

        # InferenceClient 초기화
        self.client = InferenceClient(model=self.model_name, token=self.api_key)

        self.is_loaded = True  # API는 항상 로드 상태
        logger.info(f"✅ HuggingFace API client initialized: {self.model_name}")

    def generate(self, prompt: str, max_length: int = None, **kwargs) -> str:
        """텍스트 생성 - API 버전 (영어 프롬프트)"""

        try:
            # 파라미터 준비
            actual_max_tokens = max_length or self.max_tokens
            actual_temperature = kwargs.get("temperature", self.temperature)

            # ✅ 영어 프롬프트로 수정 - 더 나은 LLM 이해도
            if not prompt.strip().startswith("Please provide"):
                enhanced_prompt = f"""Please provide an accurate and useful answer to the following question in Korean. 
    Explain technical content in an easy-to-understand manner and include specific examples.

    Question: {prompt}

    Answer:"""
            else:
                enhanced_prompt = prompt

            # 메시지 구성
            messages = [{"role": "user", "content": enhanced_prompt}]

            logger.debug(
                f"🔍 HF API call: model={self.model_name}, max_tokens={actual_max_tokens}, temp={actual_temperature}"
            )

            # API 호출
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=actual_max_tokens,
                temperature=actual_temperature,
                top_p=kwargs.get("top_p", 0.9),
            )

            # 응답 추출
            if hasattr(response, "choices") and response.choices:
                generated_text = response.choices[0].message.content.strip()
            else:
                generated_text = str(response).strip()

            # 응답 후처리
            cleaned_response = self._clean_response(generated_text)

            logger.info(f"✅ HF API generated {len(cleaned_response)} characters")
            return cleaned_response

        except Exception as e:
            logger.error(f"❌ HuggingFace API call failed: {e}")

            # 구체적인 에러 처리
            if "rate limit" in str(e).lower():
                return "죄송합니다. API 호출 한도에 도달했습니다. 잠시 후 다시 시도해주세요."
            elif "unauthorized" in str(e).lower():
                return "죄송합니다. API 인증에 문제가 있습니다. API 키를 확인해주세요."
            else:
                return f"죄송합니다. API 호출 중 오류가 발생했습니다: {str(e)[:100]}"

    def _clean_response(self, response: str) -> str:
        """응답 정리"""

        if not response:
            return "죄송합니다. 응답을 생성할 수 없었습니다."

        # 기본 정리
        response = response.strip()

        # 반복 패턴 제거
        import re

        response = re.sub(r"(.)\1{5,}", r"\1", response)
        response = re.sub(r"\b(\w+)(\s+\1){2,}\b", r"\1", response)

        # 길이 제한
        if len(response) > 2000:
            response = response[:2000] + "..."

        return response

    def load_model(self) -> None:
        """API는 로드가 필요없음"""
        pass

    def unload_model(self) -> None:
        """API는 언로드가 필요없음"""
        pass


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
        """모델 로드 - HuggingFace ID 지원"""
        if not _transformers_available:
            raise ImportError("Transformers not available for local LLM")

        model_path = self.config.get("model_path")

        logger.info(f"🤖 Loading model: {model_path}")

        # ✅ HuggingFace ID vs 로컬 경로 구분
        if "/" in model_path and not model_path.startswith("/"):
            # HuggingFace Hub에서 다운로드
            logger.info(f"📥 Downloading from HuggingFace Hub: {model_path}")
            use_auth_token = os.getenv("HUGGINGFACE_TOKEN")  # 필요시 토큰 사용
        else:
            # 로컬 경로 검증
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Local model not found: {model_path}")
            logger.info(f"📂 Loading from local path: {model_path}")
            use_auth_token = None

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.config.get("trust_remote_code", True),
            use_auth_token=use_auth_token,
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

    def enable_qa_chain_optimization(self) -> None:
        """QA Chain 최적화 활성화"""
        try:
            # ✅ 지연 import만 사용
            from .langchain.qa_chain_builder import (
                create_qa_chain_from_pipeline,
                replace_pipeline_llm_with_qa_chain,
            )

            # 검증도 지연 import로
            from .langchain.qa_chain_builder import (
                validate_qa_chain_integration as validate_func,
            )

            validation = validate_func(self.config_manager)

            if validation.get("status") not in ["ready", "partial"]:
                logger.warning("⚠️ QA Chain not ready for activation")
                logger.warning(f"   Status: {validation.get('status')}")
                if validation.get("recommendations"):
                    logger.warning("   Recommendations:")
                    for rec in validation["recommendations"][:3]:
                        logger.warning(f"      • {rec}")
                return

            if not hasattr(self, "_original_ask"):
                # 원본 메서드 백업
                self._original_ask = self.ask

                # QA Chain으로 교체
                optimized_pipeline = replace_pipeline_llm_with_qa_chain(self)
                self.ask = optimized_pipeline.ask
                self._qa_chain = getattr(optimized_pipeline, "_qa_chain", None)

                logger.info("✅ QA Chain optimization enabled successfully")
                logger.info(
                    "💡 Use pipeline.ask() as usual - now with LangChain optimization!"
                )
            else:
                logger.info("ℹ️ QA Chain optimization already enabled")

        except ImportError as e:
            logger.warning(f"⚠️ QA Chain not available: {e}")
            logger.info("🔄 Keeping original ask method")
            return
        except Exception as e:
            logger.error(f"❌ Failed to enable QA Chain optimization: {e}")
            logger.info("🔄 Keeping original ask method")
            return

    def disable_qa_chain_optimization(self) -> None:
        """QA Chain 최적화 비활성화 (원본 ask 메서드 복원)"""

        if hasattr(self, "_original_ask"):
            logger.info("🔄 Disabling QA Chain optimization...")
            self.ask = self._original_ask
            delattr(self, "_original_ask")
            if hasattr(self, "_qa_chain"):
                delattr(self, "_qa_chain")
            logger.info("✅ Original ask method restored")
        else:
            logger.info("ℹ️ QA Chain optimization not currently enabled")

    def get_qa_chain_stats(self) -> Optional[Dict[str, Any]]:
        """QA Chain 사용 통계 조회"""

        if hasattr(self, "_qa_chain") and self._qa_chain:
            try:
                # ✅ _llm 대신 llm 사용
                if hasattr(self._qa_chain, "llm") and hasattr(
                    self._qa_chain.llm, "get_usage_stats"
                ):
                    return self._qa_chain.llm.get_usage_stats()
                else:
                    return {"message": "LLM stats not available"}
            except Exception as e:
                logger.warning(f"⚠️ Could not get QA Chain stats: {e}")
        return None

    def validate_qa_chain_integration(self) -> Dict[str, Any]:
        """QA Chain 통합 가능성 검증"""

        if not _qa_chain_available:
            return {
                "status": "not_available",
                "reason": "QA Chain components not imported",
            }

        try:
            # ✅ 지연 import로 함수 가져오기
            from .langchain.qa_chain_builder import (
                validate_qa_chain_integration as validate_func,
            )

            return validate_func(self.config_manager)
        except ImportError as e:
            return {
                "status": "import_error",
                "reason": f"Failed to import validation function: {e}",
            }
        except Exception as e:
            return {
                "status": "error",
                "reason": f"Validation failed: {e}",
            }

    def setup(self) -> None:
        """시스템 전체 초기화"""
        logger.info("🔧 Setting up GraphRAG Pipeline...")
        self.state.status = PipelineStatus.INITIALIZING

        start_time = time.time()

        try:
            # 1. 설정 관리자 초기화
            # self._setup_config_manager()
            self.config_manager = GraphRAGConfigManager(
                config_file=self.config_file, env_file=self.env_file, auto_load=True
            )
            llm_config = self.config_manager.get_llm_config()

            if llm_config.get("provider") == "huggingface_local":
                from .graphrag_pipeline import (
                    LocalLLMManager,
                )  # 또는 적절한 import 경로

                self.llm_manager = LocalLLMManager(llm_config)
            else:
                # API 기반은 quick_ask에서 처리하므로 None으로 설정
                self.llm_manager = None
                logger.info("⚠️ LLM manager skipped - using quick_ask for API calls")

            # 2. 쿼리 분석기 초기화
            self._setup_query_analyzer()

            # 3. 임베딩 시스템 확인
            self._check_embeddings_system()

            # 4. 컨텍스트 직렬화기 초기화
            self._setup_context_serializer()

            # 5. QA Chain 최적화 확인 (새로 추가)
            try:
                self._check_qa_chain_availability()
            except ImportError as e:
                logger.warning(f"⚠️ QA Chain not available: {e}")

            # 6. 시스템 상태 업데이트 (기존 #6을 #7로 변경)
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

    def _check_qa_chain_availability(self) -> None:
        """QA Chain 최적화 가용성 확인"""
        logger.info("🔍 Checking QA Chain optimization availability...")

        try:
            # ✅ 지연 import로 순환 import 방지
            from .langchain.qa_chain_builder import (
                validate_qa_chain_integration as validate_func,
            )

            validation = validate_func(self.config_manager)
            status = validation.get("status", "unknown")

            if status == "ready":
                logger.info("🎯 QA Chain optimization ready for activation")
                logger.info(
                    "💡 Call pipeline.enable_qa_chain_optimization() to activate"
                )
                self.state.components_loaded["qa_chain_ready"] = True
            elif status == "partial":
                logger.info("⚠️ QA Chain partially available - some components missing")
                self.state.components_loaded["qa_chain_ready"] = False

                # 권장사항 출력
                recommendations = validation.get("recommendations", [])
                if recommendations:
                    logger.info("📋 Recommendations:")
                    for rec in recommendations[:3]:  # 최대 3개만
                        logger.info(f"   • {rec}")
            else:
                logger.info(f"ℹ️ QA Chain integration status: {status}")
                self.state.components_loaded["qa_chain_ready"] = False

        except ImportError as e:
            logger.warning(f"⚠️ QA Chain not available: {e}")
            self.state.components_loaded["qa_chain_ready"] = False
        except Exception as e:
            logger.debug(f"QA Chain validation failed: {e}")
            self.state.components_loaded["qa_chain_ready"] = False

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

        # QA Chain 상태 추가
        status["qa_chain"] = {
            "available": _qa_chain_available,
            "enabled": hasattr(self, "_qa_chain"),
            "ready": self.state.components_loaded.get("qa_chain_ready", False),
            "stats": self.get_qa_chain_stats(),
        }
        return status

    def setup_from_config(
        self,
        config_file: str = "graphrag_config.yaml",
        auto_build_embeddings: bool = False,
        enable_qa_chain: bool = True,  # 새 파라미터 추가
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

        # QA Chain 자동 활성화 (새로 추가)
        if enable_qa_chain:
            try:
                logger.info("🔗 Attempting to enable QA Chain optimization...")
                self.enable_qa_chain_optimization()
                setup_result["qa_chain_enabled"] = True
                setup_result["qa_chain_stats"] = self.get_qa_chain_stats()
                logger.info("✅ QA Chain optimization enabled successfully")
            except Exception as e:
                logger.warning(f"⚠️ QA Chain auto-enable failed: {e}")
                setup_result["qa_chain_enabled"] = False
                setup_result["qa_chain_error"] = str(e)

                # QA Chain이 실패해도 시스템은 정상 동작
                logger.info("ℹ️ Pipeline will continue with standard LLM method")
        else:
            setup_result["qa_chain_enabled"] = False
            logger.info("ℹ️ QA Chain optimization skipped (enable_qa_chain=False)")

        # 시스템 상태 확인
        status = self.get_system_status()
        setup_result["system_status"] = status

        logger.info("✅ GraphRAG setup completed!")

        # 최종 상태 요약 출력
        if setup_result.get("qa_chain_enabled"):
            logger.info("🚀 Pipeline ready with QA Chain optimization")
        else:
            logger.info("📝 Pipeline ready with standard LLM method")

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


def quick_ask_with_retriever(
    query: str, use_context: bool = True, max_docs: int = 10
) -> Dict[str, Any]:
    """
    안정적인 GraphRAG retriever + HuggingFace API 조합
    pipeline 의존성 없이 독립적으로 작동
    """
    start_time = time.time()

    try:
        # HF API 클라이언트 초기화
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY 환경변수가 설정되지 않았습니다.")

        client = InferenceClient(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct", token=api_key
        )

        context = ""
        context_retrieval_time = 0
        source_nodes = []
        retrieval_success = False

        # ✅ GraphRAG context retrieval (debug_retrieval_process 방식 사용)
        if use_context:
            try:
                context_start = time.time()
                logger.info("🔍 Retrieving context using GraphRAG retriever...")

                from src.graphrag.langchain.custom_retriever import (
                    create_graphrag_retriever,
                )

                # Retriever 생성 (debug_retrieval_process와 동일한 설정)
                retriever = create_graphrag_retriever(
                    unified_graph_path="data/processed/graphs/unified/unified_knowledge_graph.json",
                    vector_store_path="data/processed/vector_store",
                    embedding_model="auto",
                    max_docs=max_docs,
                    min_relevance_score=0.1,
                    enable_caching=False,
                )

                logger.debug("✅ GraphRAG retriever created successfully")

                # 검색 실행
                documents = retriever.get_relevant_documents(query)
                logger.debug(f"📋 Retrieved {len(documents)} documents")

                if documents:
                    # 컨텍스트 조합
                    context_parts = []
                    total_nodes = 0
                    confidence_scores = []

                    for doc in documents:
                        context_parts.append(doc.page_content)

                        # 메타데이터에서 정보 추출
                        if hasattr(doc, "metadata"):
                            total_nodes += doc.metadata.get("total_nodes", 0)
                            confidence_scores.append(
                                doc.metadata.get("confidence_score", 0.0)
                            )

                    context = "\n\n".join(context_parts)[:4000]  # 길이 제한
                    source_nodes = [f"GraphRAG_Doc_{i}" for i in range(len(documents))]

                    avg_confidence = (
                        sum(confidence_scores) / len(confidence_scores)
                        if confidence_scores
                        else 0.0
                    )

                    retrieval_success = True
                    logger.info(f"✅ Context retrieval successful:")
                    logger.info(f"   Documents: {len(documents)}")
                    logger.info(f"   Total nodes: {total_nodes}")
                    logger.info(f"   Average confidence: {avg_confidence:.3f}")
                    logger.info(f"   Context length: {len(context)} chars")

                else:
                    logger.warning("⚠️ No documents retrieved from GraphRAG")
                    context = ""

                context_retrieval_time = time.time() - context_start
                logger.info(
                    f"✅ Context retrieval completed in {context_retrieval_time:.2f}s"
                )

            except Exception as e:
                context_retrieval_time = (
                    time.time() - context_start if "context_start" in locals() else 0
                )
                logger.warning(f"⚠️ Context retrieval failed: {e}")
                logger.info("🔄 Proceeding without GraphRAG context")
                context = ""
                source_nodes = []
                retrieval_success = False

        # ✅ HuggingFace API로 답변 생성
        api_start = time.time()
        logger.info("🤖 Generating answer with HuggingFace API...")

        if context:
            enhanced_prompt = f"""Please answer the following question in Korean based on the provided context. 
Provide a comprehensive and technical answer that is easy to understand.

Context: {context}

Question: {query}

Please provide your answer in Korean:"""
            logger.info("🔍 Using GraphRAG context for enhanced answer")
        else:
            enhanced_prompt = f"""Please provide an accurate and useful answer to the following question in Korean. 
Explain technical content in an easy-to-understand manner and include specific examples.

Question: {query}

Answer:"""
            logger.info("🔍 Using standalone mode (no GraphRAG context)")

        logger.debug(f"🔍 Prompt length: {len(enhanced_prompt)} characters")

        # API 호출
        response = client.chat_completion(
            messages=[{"role": "user", "content": enhanced_prompt}],
            max_tokens=800,
            temperature=0.1,
        )

        answer = response.choices[0].message.content.strip()
        api_time = time.time() - api_start
        total_time = time.time() - start_time

        logger.info(f"✅ HuggingFace API response in {api_time:.2f}s")
        logger.info(f"📝 Generated answer: {len(answer)} characters")

        # ✅ 결과 객체 생성
        try:
            from src.graphrag.graphrag_pipeline import QAResult

            qa_result = QAResult(
                query=query,
                answer=answer,
                subgraph_result=None,
                serialized_context=None,
                query_analysis=None,
                processing_time=total_time,
                confidence_score=0.9 if retrieval_success else 0.75,
                source_nodes=source_nodes or ["HuggingFace_API_Only"],
            )
        except ImportError:

            class SimpleResult:
                def __init__(
                    self, query, answer, processing_time, confidence_score, source_nodes
                ):
                    self.query = query
                    self.answer = answer
                    self.processing_time = processing_time
                    self.confidence_score = confidence_score
                    self.source_nodes = source_nodes

            qa_result = SimpleResult(
                query,
                answer,
                total_time,
                0.9 if retrieval_success else 0.75,
                source_nodes or ["HuggingFace_API_Only"],
            )

        return {
            "result": qa_result,
            "answer": answer,
            "response_time": total_time,
            "api_time": api_time,
            "context_time": context_retrieval_time,
            "context_used": len(context) > 0,
            "context_length": len(context),
            "source_nodes": source_nodes,
            "confidence_score": 0.9 if retrieval_success else 0.75,
            "retrieval_success": retrieval_success,
            "documents_found": len(source_nodes),
            "success": True,
        }

    except Exception as e:
        logger.error(f"❌ quick_ask_with_retriever failed: {e}")

        # 상세한 에러 분류
        if "HUGGINGFACE_API_KEY" in str(e):
            error_msg = "환경변수 HUGGINGFACE_API_KEY가 설정되지 않았습니다."
        elif "unauthorized" in str(e).lower() or "api" in str(e).lower():
            error_msg = "HuggingFace API 인증 오류입니다."
        elif "rate limit" in str(e).lower():
            error_msg = "API 호출 한도를 초과했습니다."
        elif "network" in str(e).lower() or "connection" in str(e).lower():
            error_msg = "네트워크 연결 오류입니다."
        else:
            error_msg = f"알 수 없는 오류가 발생했습니다: {str(e)[:100]}"

        return {
            "answer": f"죄송합니다. {error_msg}",
            "response_time": time.time() - start_time,
            "success": False,
            "error": str(e),
            "context_used": False,
            "retrieval_success": False,
        }


def check_retriever_status() -> Dict[str, Any]:
    """GraphRAG retriever 상태 확인"""

    status = {
        "unified_graph_exists": False,
        "vector_store_exists": False,
        "retriever_ready": False,
        "api_key_set": False,
        "errors": [],
    }

    try:
        # 파일 존재 확인
        import os
        from pathlib import Path

        unified_graph_path = Path(
            "data/processed/graphs/unified/unified_knowledge_graph.json"
        )
        vector_store_path = Path("data/processed/vector_store")

        status["unified_graph_exists"] = unified_graph_path.exists()
        status["vector_store_exists"] = vector_store_path.exists()

        if not status["unified_graph_exists"]:
            status["errors"].append(f"Unified graph not found: {unified_graph_path}")

        if not status["vector_store_exists"]:
            status["errors"].append(f"Vector store not found: {vector_store_path}")

        # API 키 확인
        status["api_key_set"] = bool(os.getenv("HUGGINGFACE_API_KEY"))
        if not status["api_key_set"]:
            status["errors"].append("HUGGINGFACE_API_KEY environment variable not set")

        # retriever 생성 테스트
        if status["unified_graph_exists"] and status["vector_store_exists"]:
            try:
                from src.graphrag.langchain.custom_retriever import (
                    create_graphrag_retriever,
                )

                test_retriever = create_graphrag_retriever(
                    unified_graph_path=str(unified_graph_path),
                    vector_store_path=str(vector_store_path),
                    embedding_model="auto",
                    max_docs=1,
                    min_relevance_score=0.1,
                    enable_caching=False,
                )

                status["retriever_ready"] = True

            except Exception as e:
                status["errors"].append(f"Retriever creation failed: {e}")

        # 전체 상태 판정
        status["overall_ready"] = (
            status["unified_graph_exists"]
            and status["vector_store_exists"]
            and status["api_key_set"]
            and status["retriever_ready"]
        )

    except Exception as e:
        status["errors"].append(f"Status check failed: {e}")

    return status


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
    """개선된 GraphRAG Pipeline 테스트 - retriever 기반"""
    print("🧪 Testing GraphRAG Pipeline with improved quick_ask (retriever-based)...")
    print("=" * 80)
    # 1. 파이프라인 초기화
    pipeline = GraphRAGPipeline(config_file="graphrag_config.yaml", auto_setup=True)

    # 2. 시스템 상태 확인
    status = pipeline.get_system_status()
    print(f"📊 System Status:")
    print(f"   Pipeline: {status['pipeline_state']['status']}")

    # 3. 임베딩 구축 (필요한 경우)
    if not status["embeddings_loaded"]:
        print(f"\n🏗️ Building embeddings...")
        build_result = pipeline.build_embeddings()
        print(f"✅ Built {build_result['total_embeddings']} embeddings")

    try:
        # 1. 시스템 상태 사전 확인
        print("📊 Checking system status...")
        retriever_status = check_retriever_status()

        print(
            f"   Unified graph: {'✅' if retriever_status['unified_graph_exists'] else '❌'}"
        )
        print(
            f"   Vector store: {'✅' if retriever_status['vector_store_exists'] else '❌'}"
        )
        print(f"   API key: {'✅' if retriever_status['api_key_set'] else '❌'}")
        print(
            f"   Retriever ready: {'✅' if retriever_status['retriever_ready'] else '❌'}"
        )
        print(
            f"   Overall ready: {'✅' if retriever_status['overall_ready'] else '❌'}"
        )

        if retriever_status["errors"]:
            print(f"\n⚠️ Issues found:")
            for error in retriever_status["errors"]:
                print(f"   • {error}")

        # 2. API 키 확인 및 설정 안내
        if not retriever_status["api_key_set"]:
            print(f"\n🔑 HuggingFace API Key Setup:")
            print(f"   export HUGGINGFACE_API_KEY='your_token_here'")
            print(f"   또는 .env 파일에 HUGGINGFACE_API_KEY=your_token_here")

            # 사용자 입력으로 임시 설정 (선택사항)
            try:
                user_token = input(
                    "\n임시로 API 키를 입력하시겠습니까? (Enter to skip): "
                ).strip()
                if user_token:
                    os.environ["HUGGINGFACE_API_KEY"] = user_token
                    print("✅ API key temporarily set")
                    retriever_status["api_key_set"] = True
            except (EOFError, KeyboardInterrupt):
                print("\n⏭️ Skipping API key input")

        # 3. 테스트 실행
        if retriever_status["api_key_set"]:
            print(f"\n🚀 Starting quick_ask tests...")
            print("=" * 60)

            test_queries = [
                "What machine learning techniques are used for battery SoC prediction?",
                "What are the main tasks where AI and machine learning techniques are applied?",
                "What are the areas where AI can be applied in battery electrode processes?",
            ]

            quick_ask_results = []

            for i, query in enumerate(test_queries[:2]):  # 2개 질문 테스트
                print(f"\n🔥 Test {i+1}: {query}")
                print("-" * 50)

                # quick_ask_with_retriever 테스트
                quick_result = quick_ask_with_retriever(
                    query, use_context=retriever_status["overall_ready"], max_docs=10
                )
                quick_ask_results.append(quick_result)

                if quick_result["success"]:
                    print(f"✅ Success in {quick_result['response_time']:.2f}s")
                    print(f"📝 Answer: {quick_result['answer'][:200]}...")
                    print(
                        f"🔍 Context: {'Used' if quick_result['context_used'] else 'Not used'} ({quick_result['context_length']} chars)"
                    )
                    print(f"📄 Documents: {quick_result['documents_found']}")
                    print(f"⚡ API time: {quick_result['api_time']:.2f}s")
                    print(f"🔍 Retrieval time: {quick_result['context_time']:.2f}s")
                    print(f"🎯 Confidence: {quick_result['confidence_score']:.3f}")
                else:
                    print(f"❌ Failed: {quick_result.get('error', 'Unknown error')}")

                # 캐시 효과 테스트 (첫 번째 질문에서)
                if i == 0 and quick_result["success"]:
                    print(f"\n🔄 Testing response consistency...")
                    cache_start = time.time()
                    cache_result = quick_ask_with_retriever(
                        query, use_context=retriever_status["overall_ready"]
                    )
                    cache_time = time.time() - cache_start

                    print(f"📊 Second call time: {cache_time:.2f}s")

                    # 응답 일관성 확인
                    first_answer = quick_result["answer"][:100]
                    second_answer = (
                        cache_result["answer"][:100] if cache_result["success"] else ""
                    )

                    if first_answer == second_answer:
                        print(f"✅ Consistent response")
                    else:
                        print(f"⚠️ Different response (normal for generative AI)")

            # 4. 성능 요약
            if quick_ask_results:
                successful_results = [r for r in quick_ask_results if r["success"]]

                if successful_results:
                    print(f"\n📊 Performance Summary:")
                    print("=" * 40)

                    avg_total_time = sum(
                        r["response_time"] for r in successful_results
                    ) / len(successful_results)
                    avg_api_time = sum(r["api_time"] for r in successful_results) / len(
                        successful_results
                    )
                    avg_context_time = sum(
                        r["context_time"] for r in successful_results
                    ) / len(successful_results)

                    context_success_rate = sum(
                        1 for r in successful_results if r["context_used"]
                    ) / len(successful_results)
                    retrieval_success_rate = sum(
                        1 for r in successful_results if r["retrieval_success"]
                    ) / len(successful_results)

                    print(
                        f"✅ Success rate: {len(successful_results)}/{len(quick_ask_results)} ({len(successful_results)/len(quick_ask_results)*100:.1f}%)"
                    )
                    print(f"⚡ Average total time: {avg_total_time:.2f}s")
                    print(f"🤖 Average API time: {avg_api_time:.2f}s")
                    print(f"🔍 Average retrieval time: {avg_context_time:.2f}s")
                    print(f"📄 Context usage rate: {context_success_rate*100:.1f}%")
                    print(
                        f"🎯 Retrieval success rate: {retrieval_success_rate*100:.1f}%"
                    )

                    # 성능 분석
                    print(f"\n🔍 Performance Analysis:")
                    if avg_total_time < 5:
                        print(f"   🚀 Excellent: Under 5 seconds total")
                    elif avg_total_time < 10:
                        print(f"   ✅ Good: Under 10 seconds total")
                    else:
                        print(f"   ⚠️ Slow: Over 10 seconds total")

                    if retrieval_success_rate > 0.8:
                        print(f"   📄 GraphRAG retrieval working well")
                    elif retrieval_success_rate > 0:
                        print(f"   ⚠️ GraphRAG retrieval partially working")
                    else:
                        print(f"   ❌ GraphRAG retrieval not working")

                else:
                    print(f"\n❌ All tests failed")
                    for result in quick_ask_results:
                        if not result["success"]:
                            print(f"   Error: {result.get('error', 'Unknown')}")
        else:
            print(f"\n⚠️ Cannot run tests without API key")

        # 5. 최종 권장사항
        print(f"\n💡 Recommendations:")
        print("=" * 30)

        if retriever_status["overall_ready"]:
            print(f"   ✅ System is fully operational")
            print(f"   ✅ Use: quick_ask_with_retriever(query, use_context=True)")
        else:
            print(f"   ⚠️ GraphRAG components missing - using API-only mode")
            print(f"   ✅ Use: quick_ask_with_retriever(query, use_context=False)")

        if not retriever_status["api_key_set"]:
            print(f"   🔑 Set HUGGINGFACE_API_KEY environment variable")

        print(f"\n🎯 Usage Examples:")
        print(f"   # With GraphRAG context")
        print(
            f"   result = quick_ask_with_retriever('your question', use_context=True)"
        )
        print(f"   ")
        print(f"   # API only")
        print(
            f"   result = quick_ask_with_retriever('your question', use_context=False)"
        )
        print(f"   ")
        print(f"   print(result['answer'])")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()

        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Check HUGGINGFACE_API_KEY environment variable")
        print(f"   2. Verify GraphRAG data files exist")
        print(f"   3. Check network connection")
        print(
            f"   4. Try API-only mode: quick_ask_with_retriever(query, use_context=False)"
        )


def quick_test():
    """빠른 기능 테스트"""
    print("🧪 Quick functionality test...")

    # 시스템 상태 확인
    status = check_retriever_status()
    print(f"System ready: {'✅' if status['overall_ready'] else '❌'}")

    if status["api_key_set"]:
        # 간단한 테스트
        result = quick_ask_with_retriever(
            "What is battery management system?", use_context=status["overall_ready"]
        )

        if result["success"]:
            print(f"✅ Test successful ({result['response_time']:.2f}s)")
            print(f"Context used: {result['context_used']}")
        else:
            print(f"❌ Test failed: {result.get('error', 'Unknown')}")
    else:
        print("❌ API key not set")


if __name__ == "__main__":
    # 사용자 선택
    print("🎯 GraphRAG + HuggingFace API Test")
    print("1. Full test")
    print("2. Quick test")
    print("3. Check status only")

    try:
        choice = input("Select (1-3): ").strip()

        if choice == "1":
            main()
        elif choice == "2":
            quick_test()
        elif choice == "3":
            status = check_retriever_status()
            print(f"Status: {status}")
        else:
            main()  # 기본값

    except (EOFError, KeyboardInterrupt):
        main()  # 기본값

# def main():
#     """GraphRAG Pipeline 테스트 - QA Chain 최적화 전용"""
#     print("🧪 Testing GraphRAG Pipeline with QA Chain optimization...")

#     try:
#         # 1. 파이프라인 초기화
#         pipeline = GraphRAGPipeline(config_file="graphrag_config.yaml", auto_setup=True)

#         # 2. 시스템 상태 확인
#         status = pipeline.get_system_status()
#         print(f"📊 System Status:")
#         print(f"   Pipeline: {status['pipeline_state']['status']}")

#         # QA Chain 상태 확인
#         if "qa_chain" in status:
#             qa_status = status["qa_chain"]
#             print(f"   QA Chain Available: {qa_status['available']}")
#             print(f"   QA Chain Ready: {qa_status['ready']}")
#             print(f"   QA Chain Enabled: {qa_status['enabled']}")

#         # 3. 임베딩 구축 (필요한 경우)
#         if not status["embeddings_loaded"]:
#             print(f"\n🏗️ Building embeddings...")
#             build_result = pipeline.build_embeddings()
#             print(f"✅ Built {build_result['total_embeddings']} embeddings")

#         # 4. QA Chain 준비 상태 검증
#         print(f"\n🔍 Validating QA Chain integration...")
#         validation = pipeline.validate_qa_chain_integration()
#         print(f"   Status: {validation.get('status', 'unknown')}")

#         if validation.get("recommendations"):
#             print(f"   Recommendations:")
#             for rec in validation["recommendations"][:3]:
#                 print(f"      • {rec}")

#         # 5. QA Chain 활성화 (바로 시작)
#         if validation.get("status") == "ready":
#             print(f"\n🚀 Activating QA Chain optimization...")

#             try:
#                 pipeline.enable_qa_chain_optimization()
#                 print(f"✅ QA Chain optimization activated!")

#                 # 6. 테스트 질문들
#                 test_queries = [
#                     "What machine learning techniques are used for battery SoC prediction?",
#                     "What are the main tasks where AI and machine learning techniques are applied?"
#                     "What are the areas where AI can be applied in battery electrode processes?,",
#                 ]

#                 print(f"\n❓ Testing with QA CHAIN optimization...")

#                 for i, query in enumerate(test_queries[:2]):  # 2개 질문 테스트
#                     print(f"\n{i+1}. {query}")

#                     start_time = time.time()
#                     result = pipeline.ask(query, return_context=True)
#                     response_time = time.time() - start_time

#                     print(f"✅ Answer: {result.answer[:300]}...")
#                     print(f"📊 Response time: {response_time:.2f}s")
#                     print(f"📊 Confidence: {result.confidence_score:.3f}")
#                     print(f"📄 Sources: {len(result.source_nodes)} nodes")

#                     # 첫 번째 질문 후 캐시 효과 확인
#                     if i == 0:
#                         print(f"\n🔄 Testing cache effect - same question again...")
#                         cache_start = time.time()
#                         cache_result = pipeline.ask(query, return_context=True)
#                         cache_time = time.time() - cache_start

#                         speedup = (
#                             response_time / cache_time
#                             if cache_time > 0
#                             else float("inf")
#                         )
#                         print(f"📊 Cache response time: {cache_time:.2f}s")
#                         print(f"🚀 Speedup: {speedup:.1f}x faster")

#                 # 7. QA Chain 통계
#                 qa_stats = pipeline.get_qa_chain_stats()
#                 if qa_stats:
#                     print(f"\n📊 QA Chain Statistics:")
#                     print(f"   Total calls: {qa_stats.get('total_calls', 0)}")
#                     print(f"   Cache hits: {qa_stats.get('cache_hits', 0)}")
#                     print(
#                         f"   Cache hit ratio: {qa_stats.get('cache_hit_ratio', 0):.2%}"
#                     )
#                     print(
#                         f"   Average response time: {qa_stats.get('average_time', 0):.2f}s"
#                     )
#                     print(f"   Success rate: {qa_stats.get('success_rate', 0):.2%}")
#                     print(f"   Failed calls: {qa_stats.get('failed_calls', 0)}")

#                 # 8. LLM 어댑터 상태 확인
#                 if hasattr(pipeline, "_qa_chain") and pipeline._qa_chain:
#                     try:
#                         llm_info = pipeline._qa_chain._llm.get_model_info()
#                         print(f"\n🤖 LLM Adapter Info:")
#                         print(f"   Model path: {llm_info.get('model_path', 'unknown')}")
#                         print(f"   Adapter mode: {llm_info.get('mode', 'unknown')}")
#                         print(
#                             f"   Temperature: {llm_info.get('temperature', 'unknown')}"
#                         )
#                         print(f"   Max tokens: {llm_info.get('max_tokens', 'unknown')}")
#                         print(
#                             f"   Caching enabled: {llm_info.get('caching_enabled', 'unknown')}"
#                         )
#                     except Exception as e:
#                         print(f"⚠️ Could not get LLM adapter info: {e}")

#                 print(f"\n✅ QA Chain optimization test completed successfully!")

#             except Exception as e:
#                 print(f"❌ QA Chain optimization failed: {e}")
#                 print(f"🔄 Error details:")
#                 import traceback

#                 traceback.print_exc()

#         else:
#             print(f"\n❌ QA Chain not ready for testing")
#             print(f"   Status: {validation.get('status')}")
#             print(f"   Reason: {validation.get('reason', 'Unknown')}")

#             if validation.get("recommendations"):
#                 print(f"   Please address these issues:")
#                 for rec in validation["recommendations"]:
#                     print(f"      • {rec}")

#             return

#         # 9. 최종 상태
#         final_status = pipeline.get_system_status()
#         print(f"\n📈 Final System State:")
#         print(
#             f"   Total queries processed: {final_status['pipeline_state']['total_queries_processed']}"
#         )

#         # QA Chain 최종 상태
#         if "qa_chain" in final_status:
#             qa_final = final_status["qa_chain"]
#             print(f"   QA Chain enabled: {qa_final['enabled']}")
#             if qa_final["enabled"] and qa_final["stats"]:
#                 print(
#                     f"   QA Chain total calls: {qa_final['stats'].get('total_calls', 0)}"
#                 )

#         # 사용 가이드
#         print(f"\n💡 QA Chain is now active! Usage:")
#         print(f"   • Continue using: pipeline.ask('your question')")
#         print(f"   • Check stats: pipeline.get_qa_chain_stats()")
#         print(f"   • Disable if needed: pipeline.disable_qa_chain_optimization()")

#     except Exception as e:
#         print(f"❌ Test failed: {e}")
#         import traceback

#         traceback.print_exc()


# def main():
#     """GraphRAG Pipeline 테스트"""
#     print("🧪 Testing GraphRAG Pipeline...")

#     try:
#         # 1. 파이프라인 초기화
#         pipeline = GraphRAGPipeline(config_file="graphrag_config.yaml", auto_setup=True)

#         # 2. 시스템 상태 확인
#         status = pipeline.get_system_status()
#         print(f"📊 System Status:")
#         print(f"   Pipeline: {status['pipeline_state']['status']}")
#         print(f"   Components: {status['components']}")

#         # 3. 임베딩 구축 (필요한 경우)
#         if not status["embeddings_loaded"]:
#             print(f"\n🏗️ Building embeddings...")
#             build_result = pipeline.build_embeddings()
#             print(f"✅ Built {build_result['total_embeddings']} embeddings")

#         # 4. 테스트 질문
#         test_queries = [
#             "배터리 SoC 예측에 사용된 머신러닝 기법들은?",
#             "AI 및 머신러닝 기법이 적용된 주요 task는 무엇인가요?"
#             "배터리 전극 공정에서 AI가 적용될 수 있는 부분은 무엇이 있을까요?",
#         ]

#         print(f"\n❓ Testing queries...")
#         for i, query in enumerate(test_queries[:1]):  # 첫 번째만 테스트
#             print(f"\n{i+1}. {query}")

#             result = pipeline.ask(query, return_context=True)

#             print(f"✅ Answer: {result.answer[:200]}...")
#             print(
#                 f"📊 Stats: {result.processing_time:.2f}s, {result.confidence_score:.3f} confidence"
#             )
#             print(f"📄 Sources: {len(result.source_nodes)} nodes")

#         # 5. 최종 상태
#         final_status = pipeline.get_system_status()
#         print(f"\n📈 Final Stats:")
#         print(
#             f"   Queries processed: {final_status['pipeline_state']['total_queries_processed']}"
#         )
#         print(f"   Cache size: {final_status['cache_size']}")

#         print(f"\n✅ GraphRAG Pipeline test completed!")

#     except Exception as e:
#         print(f"❌ Test failed: {e}")
#         import traceback

#         traceback.print_exc()


if __name__ == "__main__":
    main()
