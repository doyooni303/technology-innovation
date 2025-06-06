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
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"🤖 Loading local LLM: {model_path}")

        try:
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
                "torch_dtype": getattr(
                    torch, self.config.get("torch_dtype", "bfloat16")
                ),
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, **model_kwargs
            )

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

        except Exception as e:
            logger.error(f"❌ Failed to load local LLM: {e}")
            raise

    def generate(self, prompt: str, max_length: Optional[int] = None) -> str:
        """텍스트 생성"""
        if not self.is_loaded:
            self.load_model()

        try:
            # 프롬프트 토크나이징
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,  # 입력 길이 제한
            )

            # 디바이스로 이동
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # 생성 설정 업데이트
            if max_length:
                generation_config = GenerationConfig.from_dict(
                    {**self.generation_config.to_dict(), "max_new_tokens": max_length}
                )
            else:
                generation_config = self.generation_config

            # 텍스트 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, generation_config=generation_config, use_cache=True
                )

            # 생성된 텍스트 디코딩 (입력 부분 제거)
            generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            generated_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            return generated_text.strip()

        except Exception as e:
            logger.error(f"❌ Text generation failed: {e}")
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"

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

    def _ensure_embeddings_loaded(self) -> None:
        """임베딩 시스템 로드 확인"""
        if self.embeddings_loaded:
            return

        logger.info("📥 Loading embeddings system...")

        # 설정 가져오기
        config = self.config_manager.config

        # 임베딩 생성기 초기화
        unified_graph_path = config.graph.unified_graph_path
        if not Path(unified_graph_path).exists():
            raise FileNotFoundError(f"Unified graph not found: {unified_graph_path}")

        # 벡터 저장소 경로 확인
        vector_store_path = config.graph.vector_store_path
        if not Path(vector_store_path).exists():
            logger.warning(f"Vector store not found: {vector_store_path}")
            logger.info("💡 Run build_embeddings() first to create vector store")
            return

        # 서브그래프 추출기 초기화
        self.subgraph_extractor = SubgraphExtractor(
            unified_graph_path=unified_graph_path,
            vector_store_path=vector_store_path,
            embedding_model=config.embedding.model_name,
            device=config.embedding.device,
        )

        self.embeddings_loaded = True
        logger.info("✅ Embeddings system loaded")

    def build_embeddings(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """임베딩 생성 및 벡터 저장소 구축"""
        logger.info("🏗️ Building embeddings and vector store...")

        if self.state.status != PipelineStatus.READY:
            raise RuntimeError("Pipeline not ready. Call setup() first.")

        config = self.config_manager.config

        # 임베딩 생성기 초기화
        self.embedder = MultiNodeEmbedder(
            unified_graph_path=config.graph.unified_graph_path,
            embedding_model=config.embedding.model_name,
            batch_size=config.embedding.batch_size,
            max_text_length=config.embedding.max_length,
            language="mixed",
            cache_dir=(
                config.paths.embeddings_cache if hasattr(config, "paths") else None
            ),
            device=config.embedding.device,
        )

        # 임베딩 생성
        embedding_results, saved_files = self.embedder.run_full_pipeline(
            output_dir=config.graph.vector_store_path,
            use_cache=not force_rebuild,
            show_progress=True,
        )

        # 벡터 저장소 구축
        self.vector_store = VectorStoreManager(
            store_type=getattr(config, "vector_store", {}).get("store_type", "auto"),
            persist_directory=config.graph.vector_store_path,
        )

        self.vector_store.load_from_embeddings(embedding_results)

        # 통계 반환
        total_nodes = sum(len(results) for results in embedding_results.values())

        result = {
            "total_embeddings": total_nodes,
            "embeddings_by_type": {k: len(v) for k, v in embedding_results.items()},
            "saved_files": {k: str(v) for k, v in saved_files.items()},
            "vector_store_path": config.graph.vector_store_path,
        }

        logger.info(f"✅ Embeddings built: {total_nodes:,} nodes")
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

        try:
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
            answer = self._generate_answer(query, serialized_context.main_text)

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

        except Exception as e:
            self.state.status = PipelineStatus.ERROR
            self.state.last_error = str(e)
            logger.error(f"❌ Query processing failed: {e}")

            # 간단한 답변 반환
            fallback_answer = f"죄송합니다. 질문 처리 중 오류가 발생했습니다: {str(e)}"

            if return_context:
                return QAResult(
                    query=query,
                    answer=fallback_answer,
                    subgraph_result=None,
                    serialized_context=None,
                    query_analysis=None,
                    processing_time=time.time() - start_time,
                    confidence_score=0.0,
                    source_nodes=[],
                )
            else:
                return fallback_answer

    def _generate_answer(self, query: str, context: str) -> str:
        """LLM으로 답변 생성"""

        # 프롬프트 구성
        prompt = self._build_qa_prompt(query, context)

        # LLM으로 답변 생성
        if self.llm_manager and self.llm_manager.config.get("model_path"):
            # 로컬 LLM 사용
            try:
                answer = self.llm_manager.generate(prompt, max_length=1000)
                return answer
            except Exception as e:
                logger.error(f"❌ Local LLM generation failed: {e}")
                return f"로컬 LLM 오류: {str(e)}"
        else:
            # API LLM 폴백 또는 기본 답변
            return (
                "죄송합니다. LLM이 설정되지 않았습니다. 로컬 모델 경로를 확인해주세요."
            )

    def _build_qa_prompt(self, query: str, context: str) -> str:
        """QA 프롬프트 구성"""

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

        return prompt_template.format(context=context, query=query)

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
        """시스템 상태 확인"""
        status = {
            "pipeline_state": self.state.to_dict(),
            "components": self.state.components_loaded,
            "embeddings_loaded": self.embeddings_loaded,
            "cache_size": len(self.query_cache),
            "memory_usage": self._get_memory_usage(),
        }

        # LLM 상태
        if self.llm_manager:
            status["llm_loaded"] = self.llm_manager.is_loaded
            status["llm_model_path"] = self.llm_manager.config.get("model_path")

        return status

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
