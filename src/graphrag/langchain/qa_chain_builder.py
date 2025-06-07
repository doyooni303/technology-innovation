"""
GraphRAG QA 체인 빌더
QA Chain Builder for GraphRAG System

LangChain 기반 최적화된 QA 체인 구축
- Custom GraphRAG Retriever 통합
- 쿼리 타입별 최적화된 프롬프트 체인
- 메모리 관리 및 대화 히스토리
- 로컬 LLM 최적화 및 에러 핸들링
- 배치 처리 및 캐싱 지원
"""

import logging
import warnings
from typing import Dict, List, Any, Optional, Union, Type, Callable
from pathlib import Path
from pydantic import Field
from dataclasses import dataclass
from enum import Enum

# LangChain Core imports
try:
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.prompts import BasePromptTemplate
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.memory import BaseMemory
    from langchain_core.callbacks import CallbackManagerForChainRun
    from langchain_core.documents import Document
    from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables.utils import Input, Output

    # LangChain Community imports
    from langchain.chains import RetrievalQA
    from langchain.chains.base import Chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains.retrieval import createretrieval_chain
    from langchain.chains.conversational_retrieval.base import (
        ConversationalRetrievalChain,
    )
    from langchain.memory import (
        ConversationBufferMemory,
        ConversationSummaryBufferMemory,
    )

    _langchain_available = True
except ImportError as e:
    _langchain_available = False
    warnings.warn(f"LangChain not available: {e}")

# GraphRAG imports
try:
    from .custom_retriever import GraphRAGRetriever, create_graphrag_retriever
    from .prompt_templates import (
        GraphRAGPromptTemplates,
        create_query_prompt,
        create_chat_prompt,
    )
    from .memory_manager import GraphRAGMemoryManager, create_memory_manager
    from ..query_analyzer import QueryAnalyzer, QueryAnalysisResult
    from ..graphrag_pipeline import LocalLLMManager
except ImportError as e:
    warnings.warn(f"Some GraphRAG components not available: {e}")

# 로깅 설정
logger = logging.getLogger(__name__)


class ChainType(Enum):
    """QA 체인 타입"""

    BASIC_QA = "basic_qa"  # 기본 QA 체인
    RETRIEVAL_QA = "retrieval_qa"  # 검색 기반 QA
    CONVERSATIONAL_QA = "conversational_qa"  # 대화형 QA
    GRAPH_ENHANCED_QA = "graph_enhanced_qa"  # GraphRAG 특화 QA
    MULTI_QUERY_QA = "multi_query_qa"  # 다중 쿼리 QA
    STREAMING_QA = "streaming_qa"  # 스트리밍 QA


@dataclass
class QAChainConfig:
    """QA 체인 설정"""

    # 체인 타입 및 기본 설정
    chain_type: ChainType = ChainType.GRAPH_ENHANCED_QA
    return_source_documents: bool = True
    verbose: bool = False

    # 검색 설정
    search_kwargs: Dict[str, Any] = None
    max_docs_for_context: int = 10
    min_relevance_score: float = 0.3

    # 메모리 설정
    enable_memory: bool = True
    memory_type: str = "summary_buffer"
    max_memory_tokens: int = 4000

    # 응답 설정
    max_answer_tokens: int = 1000
    temperature: float = 0.1
    streaming: bool = False

    # 에러 핸들링
    max_retries: int = 3
    timeout_seconds: int = 300
    fallback_to_simple: bool = True

    def __post_init__(self):
        if self.search_kwargs is None:
            self.search_kwargs = {
                "k": self.max_docs_for_context,
                "score_threshold": self.min_relevance_score,
            }


class GraphRAGQAChain(Chain):
    """GraphRAG 특화 QA 체인 (LangChain Chain 상속) - Pydantic 호환"""

    # LangChain Chain 필수 속성들
    input_keys: List[str] = ["question"]
    output_keys: List[str] = ["answer"]

    # Pydantic 필드로 모든 속성 정의
    retriever: BaseRetriever = Field(description="GraphRAG 커스텀 리트리버")
    llm: BaseLanguageModel = Field(description="언어 모델")
    prompt_template: BasePromptTemplate = Field(description="프롬프트 템플릿")
    memory: Optional[BaseMemory] = Field(default=None, description="메모리 관리자")
    query_analyzer: Optional[QueryAnalyzer] = Field(
        default=None, description="쿼리 분석기"
    )
    config: Optional[QAChainConfig] = Field(default=None, description="QA 체인 설정")

    # 내부 체인들 (exclude=True로 직렬화에서 제외)
    base_chain: Optional[Any] = Field(default=None, exclude=True)
    conversation_chain: Optional[Any] = Field(default=None, exclude=True)
    retrieval_chain: Optional[Any] = Field(default=None, exclude=True)

    # 통계 및 캐싱 (exclude=True)
    query_count: int = Field(default=0, exclude=True)
    cache: Optional[Dict[str, Any]] = Field(default=None, exclude=True)

    class Config:
        """Pydantic 설정"""

        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        """Pydantic 방식으로 초기화"""
        # config 기본값 설정
        if "config" not in kwargs or kwargs["config"] is None:
            kwargs["config"] = QAChainConfig()

        # cache 기본값 설정
        if kwargs["config"].enable_memory:
            kwargs["cache"] = {}

        # 부모 클래스 초기화 (Pydantic 방식)
        super().__init__(**kwargs)

        if not _langchain_available:
            raise ImportError("LangChain is required for QA Chain Builder")

        # 내부 체인들 초기화
        self._initialize_chains()

        logger.info("✅ GraphRAGQAChain initialized")
        logger.info(f"   🔗 Chain type: {self.config.chain_type.value}")
        logger.info(f"   🧠 Memory enabled: {self.config.enable_memory}")
        logger.info(f"   📄 Max docs: {self.config.max_docs_for_context}")

    def _initialize_chains(self) -> None:
        """내부 체인들 초기화"""

        # 1. 기본 문서 결합 체인 생성
        try:
            self.base_chain = create_stuff_documents_chain(
                llm=self.llm,
                prompt=self.prompt_template,
                document_variable_name="context",
                verbose=self.config.verbose,
            )

            # 2. 검색 체인 생성
            self.retrieval_chain = createretrieval_chain(
                retriever=self.retriever, combine_docs_chain=self.base_chain
            )

            # 3. 대화형 체인 생성 (메모리가 있는 경우)
            if self.memory and self.config.enable_memory:
                try:
                    self.conversation_chain = ConversationalRetrievalChain.from_llm(
                        llm=self.llm,
                        retriever=self.retriever,
                        memory=self.memory,
                        return_source_documents=self.config.return_source_documents,
                        verbose=self.config.verbose,
                        combine_docs_chain_kwargs={"prompt": self.prompt_template},
                    )
                    logger.info("✅ Conversational chain initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize conversational chain: {e}")
                    self.conversation_chain = None

        except Exception as e:
            logger.error(f"❌ Failed to initialize chains: {e}")
            # 최소한의 체인이라도 만들기
            self.base_chain = None
            self.retrieval_chain = None
            self.conversation_chain = None

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """메인 체인 실행 (LangChain Chain 인터페이스)"""

        question = inputs.get("question", "")
        if not question:
            return {"answer": "질문이 제공되지 않았습니다."}

        self.query_count += 1
        logger.info(f"🔍 Processing query #{self.query_count}: {question[:50]}...")

        try:
            # 1. 쿼리 분석 (선택적)
            query_analysis = None
            if self.query_analyzer:
                query_analysis = self.query_analyzer.analyze(question)
                logger.debug(f"📊 Query analysis: {query_analysis.query_type.value}")

            # 2. 체인 타입별 처리
            if (
                self.config.chain_type == ChainType.CONVERSATIONAL_QA
                and self.conversation_chain
            ):
                result = self._process_conversational_qa(question, run_manager)
            elif self.config.chain_type == ChainType.GRAPH_ENHANCED_QA:
                result = self._process_graph_enhanced_qa(
                    question, query_analysis, run_manager
                )
            else:
                result = self._process_basic_qa(question, run_manager)

            # 3. 결과 후처리
            processed_result = self._post_process_result(
                result, question, query_analysis
            )

            logger.info(f"✅ Query #{self.query_count} completed")
            return processed_result

        except Exception as e:
            logger.error(f"❌ Query #{self.query_count} failed: {e}")

            # 폴백 처리
            if self.config.fallback_to_simple:
                return self._fallback_answer(question, str(e))
            else:
                raise

    # 나머지 메서드들은 동일...
    def _process_conversational_qa(
        self, question: str, run_manager: Optional[CallbackManagerForChainRun]
    ) -> Dict[str, Any]:
        """대화형 QA 처리"""

        logger.debug("💬 Processing conversational QA")

        if not self.conversation_chain:
            # 폴백: 기본 QA로 처리
            return self._process_basic_qa(question, run_manager)

        try:
            # 대화 히스토리를 고려한 처리
            result = self.conversation_chain(
                {
                    "question": question,
                    "chat_history": (
                        self.memory.chat_memory.messages if self.memory else []
                    ),
                }
            )

            return {
                "answer": result.get("answer", ""),
                "source_documents": result.get("source_documents", []),
                "chat_history": result.get("chat_history", []),
            }

        except Exception as e:
            logger.warning(f"⚠️ Conversational QA failed: {e}, falling back to basic QA")
            return self._process_basic_qa(question, run_manager)

    def _process_graph_enhanced_qa(
        self,
        question: str,
        query_analysis: Optional[QueryAnalysisResult],
        run_manager: Optional[CallbackManagerForChainRun],
    ) -> Dict[str, Any]:
        """GraphRAG 특화 QA 처리"""

        logger.debug("🕸️ Processing graph-enhanced QA")

        try:
            # 1. 동적 프롬프트 생성 (쿼리 분석 결과 활용)
            if query_analysis and hasattr(self.retriever, "update_config"):
                # 쿼리 타입에 따른 검색 설정 조정
                search_config = self._adapt_search_config(query_analysis)
                self.retriever.update_config(**search_config)

            # 2. 검색 체인 실행
            if self.retrieval_chain:
                result = self.retrieval_chain.invoke({"input": question})
            else:
                # 검색 체인이 없으면 직접 검색
                docs = self.retriever.get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in docs])

                # LLM으로 직접 답변 생성
                prompt_text = f"컨텍스트: {context}\n\n질문: {question}\n\n답변:"
                answer = self.llm.invoke(prompt_text)

                result = {
                    "answer": answer if isinstance(answer, str) else str(answer),
                    "context": docs,
                }

            return {
                "answer": result.get("answer", ""),
                "source_documents": result.get("context", []),
                "query_analysis": query_analysis,
            }

        except Exception as e:
            logger.warning(f"⚠️ Graph-enhanced QA failed: {e}, falling back to basic QA")
            return self._process_basic_qa(question, run_manager)

    def _process_basic_qa(
        self, question: str, run_manager: Optional[CallbackManagerForChainRun]
    ) -> Dict[str, Any]:
        """기본 QA 처리"""

        logger.debug("📝 Processing basic QA")

        try:
            if self.retrieval_chain:
                # 검색 체인이 있으면 사용
                result = self.retrieval_chain.invoke({"input": question})
            else:
                # 검색 체인이 없으면 직접 처리
                docs = self.retriever.get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in docs[:5]])

                # 간단한 프롬프트로 답변 생성
                prompt_text = f"다음 컨텍스트를 바탕으로 질문에 답변하세요:\n\n컨텍스트: {context}\n\n질문: {question}\n\n답변:"

                try:
                    if hasattr(self.llm, "invoke"):
                        answer = self.llm.invoke(prompt_text)
                    elif hasattr(self.llm, "_call"):
                        answer = self.llm._call(prompt_text)
                    else:
                        answer = str(self.llm(prompt_text))

                    answer = answer if isinstance(answer, str) else str(answer)
                except Exception as llm_error:
                    logger.error(f"❌ LLM call failed: {llm_error}")
                    answer = "죄송합니다. 답변 생성 중 오류가 발생했습니다."

                result = {"answer": answer, "context": docs}

            return {
                "answer": result.get("answer", ""),
                "source_documents": result.get("context", []),
            }

        except Exception as e:
            logger.error(f"❌ Basic QA failed: {e}")
            return {
                "answer": f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}",
                "source_documents": [],
            }

    # 나머지 메서드들 (동일하게 유지)
    def _adapt_search_config(
        self, query_analysis: QueryAnalysisResult
    ) -> Dict[str, Any]:
        """쿼리 분석 결과에 따른 검색 설정 조정"""
        config = {}

        # 복잡도에 따른 문서 수 조정
        if query_analysis.complexity.value == "simple":
            config["max_docs"] = min(5, self.config.max_docs_for_context)
        elif query_analysis.complexity.value == "complex":
            config["max_docs"] = self.config.max_docs_for_context
        elif query_analysis.complexity.value == "exploratory":
            config["max_docs"] = min(15, self.config.max_docs_for_context + 5)

        return config

    def _post_process_result(
        self,
        result: Dict[str, Any],
        question: str,
        query_analysis: Optional[QueryAnalysisResult],
    ) -> Dict[str, Any]:
        """결과 후처리"""

        answer = result.get("answer", "")
        source_docs = result.get("source_documents", [])

        # 답변 길이 제한
        if len(answer) > self.config.max_answer_tokens * 4:
            answer = answer[: self.config.max_answer_tokens * 4] + "..."
            logger.debug("📏 Answer truncated due to length limit")

        # 최종 결과 구성
        final_result = {
            "answer": answer,
            "question": question,
            "source_count": len(source_docs),
        }

        if self.config.return_source_documents:
            final_result["source_documents"] = source_docs[
                : self.config.max_docs_for_context
            ]

        if query_analysis:
            final_result["query_analysis"] = {
                "type": query_analysis.query_type.value,
                "complexity": query_analysis.complexity.value,
                "confidence": getattr(query_analysis, "confidence_score", 0.0),
            }

        return final_result

    def _fallback_answer(self, question: str, error_msg: str) -> Dict[str, Any]:
        """폴백 답변 생성"""

        fallback_answer = (
            f"죄송합니다. 질문 처리 중 문제가 발생했습니다.\n"
            f"문제: {error_msg}\n\n"
            f"다시 질문해주시거나, 더 구체적으로 질문해보세요."
        )

        return {
            "answer": fallback_answer,
            "question": question,
            "source_count": 0,
            "error": error_msg,
            "is_fallback": True,
        }

    @property
    def _chain_type(self) -> str:
        """LangChain Chain 타입 식별자"""
        return "graphrag_qa_chain"


class QAChainBuilder:
    """GraphRAG QA 체인 빌더 - 팩토리 클래스"""

    def __init__(
        self,
        unified_graph_path: str,
        vector_store_path: str,
        config_manager: Optional[object] = None,
    ):
        """
        Args:
            unified_graph_path: 통합 그래프 경로
            vector_store_path: 벡터 저장소 경로
            config_manager: GraphRAG 설정 관리자 (선택적)
        """

        if not _langchain_available:
            raise ImportError("LangChain is required for QA Chain Builder")

        self.unified_graph_path = Path(unified_graph_path)
        self.vector_store_path = Path(vector_store_path)
        self.config_manager = config_manager

        # 컴포넌트들 (지연 로딩)
        self._retriever = None
        self._llm = None
        self._memory_manager = None
        self._query_analyzer = None
        self._prompt_templates = None

        logger.info("✅ QAChainBuilder initialized")

    def create_chain(
        self,
        chain_type: Union[ChainType, str] = ChainType.GRAPH_ENHANCED_QA,
        llm: Optional[BaseLanguageModel] = None,
        embedding_model: str = "auto",
        config: Optional[QAChainConfig] = None,
        **kwargs,
    ) -> GraphRAGQAChain:
        """QA 체인 생성 메인 메서드"""

        # 체인 타입 변환
        if isinstance(chain_type, str):
            try:
                chain_type = ChainType(chain_type)
            except ValueError:
                logger.warning(f"⚠️ Unknown chain type: {chain_type}, using default")
                chain_type = ChainType.GRAPH_ENHANCED_QA

        # 설정 생성
        config = config or QAChainConfig(chain_type=chain_type)

        logger.info(f"🏗️ Building QA chain: {chain_type.value}")

        # 1. 리트리버 초기화
        retriever = self._get_or_create_retriever(embedding_model, config)

        # 2. LLM 초기화
        llm_model = llm or self._get_or_create_llm(config)

        # 3. 프롬프트 템플릿 초기화
        prompt_template = self._get_or_create_prompt_template(config)

        # 4. 메모리 초기화 (필요시)
        memory = None
        if config.enable_memory:
            memory = self._get_or_create_memory(config)

        # 5. 쿼리 분석기 초기화 (선택적)
        query_analyzer = self._get_or_create_query_analyzer()

        # 6. QA 체인 생성
        qa_chain = GraphRAGQAChain(
            retriever=retriever,
            llm=llm_model,
            prompt_template=prompt_template,
            memory=memory,
            query_analyzer=query_analyzer,
            config=config,
            **kwargs,
        )

        logger.info(f"✅ QA chain created successfully: {chain_type.value}")
        return qa_chain

    def _get_or_create_retriever(
        self, embedding_model: str, config: QAChainConfig
    ) -> BaseRetriever:
        """리트리버 생성 또는 조회"""

        if self._retriever is None:
            logger.info("📥 Creating GraphRAG retriever...")

            self._retriever = create_graphrag_retriever(
                unified_graph_path=str(self.unified_graph_path),
                vector_store_path=str(self.vector_store_path),
                embedding_model=embedding_model,
                max_docs=config.max_docs_for_context,
                min_relevance_score=config.min_relevance_score,
                enable_caching=True,
            )

            logger.info("✅ GraphRAG retriever created")

        return self._retriever

    def _get_or_create_llm(self, config: QAChainConfig) -> BaseLanguageModel:
        """LLM 생성 또는 조회 - YAML 설정 완전 호환"""

        if self._llm is None:
            logger.info("🤖 Creating LLM with YAML config compatibility...")

            if not self.config_manager:
                raise ValueError("config_manager is required for LLM creation")

            try:
                # YAML 설정으로부터 LLM 어댑터 생성
                from .langchain_llm_adapter import create_llm_adapter, AdapterMode

                # 설정에 따른 어댑터 모드 결정
                adapter_mode = (
                    AdapterMode.CACHED if config.enable_memory else AdapterMode.DIRECT
                )

                self._llm = create_llm_adapter(
                    config_manager=self.config_manager,
                    temperature=config.temperature,
                    max_tokens=config.max_answer_tokens,
                    mode=adapter_mode,
                    enable_caching=True,
                    max_retries=config.max_retries,
                )

                # LLM 정보 로깅
                model_info = self._llm.get_model_info()
                logger.info("✅ LLM adapter created from YAML config")
                logger.info(f"   Provider: {model_info.get('model_path', 'unknown')}")
                logger.info(f"   Mode: {model_info.get('mode', 'unknown')}")
                logger.info(
                    f"   Temperature: {model_info.get('temperature', 'unknown')}"
                )

            except ImportError as e:
                logger.error(f"❌ Failed to import LLM adapter: {e}")
                raise ImportError(
                    "langchain_llm_adapter is required. "
                    "Make sure langchain_llm_adapter.py is in the same directory."
                )
            except Exception as e:
                logger.error(f"❌ Failed to create LLM adapter: {e}")
                # 폴백: 기존 방식으로 시도
                logger.info("🔄 Falling back to direct LocalLLMManager...")

                llm_config = self.config_manager.get_llm_config()
                if llm_config.get("provider") == "huggingface_local":
                    # 직접 LocalLLMManager 사용 (LangChain 호환성 없음)
                    from ..graphrag_pipeline import LocalLLMManager
                    from .langchain_llm_adapter import create_llm_adapter_from_manager

                    llm_manager = LocalLLMManager(llm_config)
                    self._llm = create_llm_adapter_from_manager(
                        llm_manager=llm_manager,
                        temperature=config.temperature,
                        max_tokens=config.max_answer_tokens,
                    )

                    logger.info("⚠️ Using fallback LLM adapter")
                else:
                    raise NotImplementedError(
                        f"Provider '{llm_config.get('provider')}' not implemented yet. "
                        f"Currently supported: huggingface_local"
                    )

        return self._llm

    def _get_or_create_prompt_template(
        self, config: QAChainConfig
    ) -> BasePromptTemplate:
        """프롬프트 템플릿 생성 또는 조회 - 간단한 해결책"""

        if self._prompt_templates is None:
            logger.info("📝 Creating prompt templates...")

            try:
                # config 없이 기본값으로 생성
                self._prompt_templates = GraphRAGPromptTemplates()
                logger.info("✅ Prompt templates created")

            except Exception as e:
                logger.warning(f"⚠️ GraphRAGPromptTemplates failed: {e}")

                # 직접 간단한 프롬프트 반환
                from langchain_core.prompts import PromptTemplate

                template = """다음 컨텍스트를 바탕으로 질문에 답변하세요:

    컨텍스트: {context}

    질문: {question}

    답변:"""

                return PromptTemplate(
                    template=template, input_variables=["context", "question"]
                )

        # 프롬프트 생성 시도
        try:
            return self._prompt_templates.create_langchain_prompt()
        except:
            # 실패시 기본 프롬프트
            from langchain_core.prompts import PromptTemplate

            template = """컨텍스트: {context}

    질문: {question}

    답변:"""

            return PromptTemplate(
                template=template, input_variables=["context", "question"]
            )

    def _get_or_create_memory(self, config: QAChainConfig) -> BaseMemory:
        """메모리 생성 또는 조회"""

        if self._memory_manager is None:
            logger.info("🧠 Creating memory manager...")

            # GraphRAG 메모리 관리자를 LangChain 메모리로 변환
            memory_config = {
                "memory_type": config.memory_type,
                "max_token_limit": config.max_memory_tokens,
                "return_messages": True,
            }

            if config.memory_type == "summary_buffer":
                # LLM이 필요한 경우 추후 설정
                self._memory_manager = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True, output_key="answer"
                )
            else:
                self._memory_manager = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True, output_key="answer"
                )

            logger.info("✅ Memory manager created")

        return self._memory_manager

    def _get_or_create_query_analyzer(self) -> Optional[QueryAnalyzer]:
        """쿼리 분석기 생성 또는 조회"""

        if self._query_analyzer is None:
            try:
                logger.info("📊 Creating query analyzer...")
                self._query_analyzer = QueryAnalyzer()
                logger.info("✅ Query analyzer created")
            except Exception as e:
                logger.warning(f"⚠️ Failed to create query analyzer: {e}")
                self._query_analyzer = None

        return self._query_analyzer

    def create_conversational_chain(
        self,
        llm: Optional[BaseLanguageModel] = None,
        session_id: str = "default",
        **kwargs,
    ) -> GraphRAGQAChain:
        """대화형 QA 체인 생성 (편의 메서드)"""

        config = QAChainConfig(
            chain_type=ChainType.CONVERSATIONAL_QA,
            enable_memory=True,
            memory_type="summary_buffer",
            **kwargs,
        )

        return self.create_chain(
            chain_type=ChainType.CONVERSATIONAL_QA, llm=llm, config=config
        )

    def create_basic_chain(
        self, llm: Optional[BaseLanguageModel] = None, **kwargs
    ) -> GraphRAGQAChain:
        """기본 QA 체인 생성 (편의 메서드)"""

        config = QAChainConfig(
            chain_type=ChainType.BASIC_QA, enable_memory=False, **kwargs
        )

        return self.create_chain(chain_type=ChainType.BASIC_QA, llm=llm, config=config)

    def get_chain_info(self) -> Dict[str, Any]:
        """체인 빌더 정보 반환 - 확장된 버전"""

        info = {
            "unified_graph_path": str(self.unified_graph_path),
            "vector_store_path": str(self.vector_store_path),
            "config_manager_available": self.config_manager is not None,
            "components_loaded": {
                "retriever": self._retriever is not None,
                "llm": self._llm is not None,
                "memory": self._memory_manager is not None,
                "query_analyzer": self._query_analyzer is not None,
                "prompt_templates": self._prompt_templates is not None,
            },
            "available_chain_types": [ct.value for ct in ChainType],
            "langchain_available": _langchain_available,
        }

        # LLM 상세 정보 추가
        if self._llm:
            try:
                llm_info = self._llm.get_model_info()
                info["llm_info"] = llm_info

                if hasattr(self._llm, "get_usage_stats"):
                    info["llm_usage_stats"] = self._llm.get_usage_stats()
            except Exception as e:
                logger.warning(f"⚠️ Could not get LLM info: {e}")

        # 설정 관리자 정보 추가
        if self.config_manager:
            try:
                info["yaml_config"] = {
                    "llm_provider": self.config_manager.config.llm.provider,
                    "vector_store_type": self.config_manager.config.vector_store.store_type,
                    "embedding_model": self.config_manager.config.embeddings.sentence_transformers.model_name,
                }
            except Exception as e:
                logger.warning(f"⚠️ Could not get config info: {e}")

        return info

    def integrate_with_pipeline(self, pipeline: "GraphRAGPipeline") -> GraphRAGQAChain:
        """GraphRAG Pipeline과 통합하여 QA 체인 생성"""

        logger.info("🔗 Integrating QA Chain with GraphRAG Pipeline...")

        # Pipeline의 설정 관리자 사용
        self.config_manager = pipeline.config_manager

        # Pipeline의 컴포넌트들 재사용
        if hasattr(pipeline, "vector_store") and pipeline.vector_store:
            logger.info("📚 Reusing pipeline's vector store...")
            # 벡터 저장소가 이미 로드되어 있으면 재사용

        if hasattr(pipeline, "query_analyzer") and pipeline.query_analyzer:
            logger.info("🔍 Reusing pipeline's query analyzer...")
            self._query_analyzer = pipeline.query_analyzer

        # 최적화된 설정으로 QA 체인 생성
        config = QAChainConfig(
            chain_type=ChainType.GRAPH_ENHANCED_QA,
            enable_memory=True,
            memory_type="summary_buffer",
            max_docs_for_context=10,
            temperature=0.1,
            max_retries=3,
            fallback_to_simple=True,
        )

        qa_chain = self.create_chain(config=config)

        logger.info("✅ QA Chain integrated with Pipeline successfully")
        return qa_chain

    def create_optimized_chain_for_pipeline(
        self, config_manager: "GraphRAGConfigManager"
    ) -> GraphRAGQAChain:
        """Pipeline 전용 최적화된 체인 생성"""

        self.config_manager = config_manager

        # YAML 설정에서 최적 파라미터 추출
        try:
            llm_config = config_manager.get_llm_config()
            vector_config = config_manager.get_vector_store_config()

            # 설정 기반 최적화
            config = QAChainConfig(
                chain_type=ChainType.GRAPH_ENHANCED_QA,
                enable_memory=True,
                max_docs_for_context=min(15, vector_config.get("batch_size", 10)),
                temperature=llm_config.get("temperature", 0.1),
                max_answer_tokens=llm_config.get("max_new_tokens", 1000),
                max_retries=3,
                streaming=False,  # YAML 설정에서 do_sample: false이므로
                fallback_to_simple=True,
            )

            logger.info("🎯 Creating optimized chain from YAML config...")
            logger.info(f"   LLM Provider: {llm_config.get('provider')}")
            logger.info(f"   Vector Store: {vector_config.get('store_type')}")
            logger.info(f"   Max Docs: {config.max_docs_for_context}")
            logger.info(f"   Temperature: {config.temperature}")

            return self.create_chain(config=config)

        except Exception as e:
            logger.error(f"❌ Failed to create optimized chain: {e}")
            logger.info("🔄 Falling back to default configuration...")

            # 폴백: 기본 설정
            return self.create_chain()

    def health_check(self) -> Dict[str, Any]:
        """QA 체인 빌더 상태 확인"""

        health_status = {
            "overall_status": "unknown",
            "components": {},
            "dependencies": {
                "langchain": _langchain_available,
                "unified_graph_exists": self.unified_graph_path.exists(),
                "vector_store_exists": self.vector_store_path.exists(),
            },
            "config_manager": self.config_manager is not None,
            "errors": [],
        }

        try:
            # 컴포넌트별 상태 확인
            if self._llm:
                try:
                    llm_health = self._llm.health_check()
                    health_status["components"]["llm"] = llm_health
                except Exception as e:
                    health_status["components"]["llm"] = {
                        "status": "error",
                        "error": str(e),
                    }
                    health_status["errors"].append(f"LLM health check failed: {e}")

            if self._retriever:
                health_status["components"]["retriever"] = {"status": "loaded"}

            if self._query_analyzer:
                health_status["components"]["query_analyzer"] = {"status": "loaded"}

            # 전체 상태 결정
            error_count = len(health_status["errors"])
            if error_count == 0:
                health_status["overall_status"] = "healthy"
            elif error_count <= 2:
                health_status["overall_status"] = "degraded"
            else:
                health_status["overall_status"] = "unhealthy"

        except Exception as e:
            health_status["overall_status"] = "error"
            health_status["errors"].append(f"Health check failed: {e}")

        return health_status


# 편의 함수들 - Pipeline 통합 지원
def create_qa_chain(
    unified_graph_path: str,
    vector_store_path: str,
    chain_type: Union[ChainType, str] = ChainType.GRAPH_ENHANCED_QA,
    config_manager: Optional[object] = None,
    **kwargs,
) -> GraphRAGQAChain:
    """QA 체인 생성 편의 함수 - YAML 설정 지원"""

    builder = QAChainBuilder(
        unified_graph_path=unified_graph_path,
        vector_store_path=vector_store_path,
        config_manager=config_manager,
    )

    return builder.create_chain(chain_type=chain_type, **kwargs)


def create_conversational_qa_chain(
    unified_graph_path: str,
    vector_store_path: str,
    config_manager: Optional[object] = None,
    **kwargs,
) -> GraphRAGQAChain:
    """대화형 QA 체인 생성 편의 함수"""

    builder = QAChainBuilder(
        unified_graph_path=unified_graph_path,
        vector_store_path=vector_store_path,
        config_manager=config_manager,
    )

    return builder.create_conversational_chain(**kwargs)


def create_qa_chain_from_pipeline(pipeline: "GraphRAGPipeline") -> GraphRAGQAChain:
    """GraphRAG Pipeline으로부터 QA 체인 생성 (성능 최적화)"""

    if not hasattr(pipeline, "config_manager") or not pipeline.config_manager:
        raise ValueError("Pipeline must have a valid config_manager")

    config = pipeline.config_manager.config

    builder = QAChainBuilder(
        unified_graph_path=config.graph.unified_graph_path,
        vector_store_path=config.graph.vector_store_path,
        config_manager=pipeline.config_manager,
    )

    # Pipeline과 통합
    qa_chain = builder.integrate_with_pipeline(pipeline)

    logger.info("🔗 QA Chain created from GraphRAG Pipeline")
    return qa_chain


def create_optimized_qa_chain(
    config_manager: "GraphRAGConfigManager",
) -> GraphRAGQAChain:
    """YAML 설정 최적화된 QA 체인 생성"""

    config = config_manager.config

    builder = QAChainBuilder(
        unified_graph_path=config.graph.unified_graph_path,
        vector_store_path=config.graph.vector_store_path,
        config_manager=config_manager,
    )

    return builder.create_optimized_chain_for_pipeline(config_manager)


def replace_pipeline_llm_with_qa_chain(
    pipeline: "GraphRAGPipeline",
) -> "GraphRAGPipeline":
    """Pipeline의 LLM 호출을 QA Chain으로 교체 (성능 개선)"""

    logger.info("🔄 Replacing Pipeline LLM with optimized QA Chain...")

    # QA 체인 생성
    qa_chain = create_qa_chain_from_pipeline(pipeline)

    # Pipeline의 ask 메서드를 QA 체인으로 교체
    original_ask = pipeline.ask

    def optimized_ask(query: str, return_context: bool = False):
        """최적화된 ask 메서드 (QA Chain 사용)"""

        try:
            logger.info(f"🚀 Processing query with optimized QA Chain: {query[:50]}...")

            # QA 체인으로 처리
            result = qa_chain._call({"question": query})

            if return_context:
                # QAResult 형태로 변환
                from ..graphrag_pipeline import QAResult

                return QAResult(
                    query=query,
                    answer=result.get("answer", ""),
                    subgraph_result=None,  # QA Chain에서는 직접 제공하지 않음
                    serialized_context=None,
                    query_analysis=result.get("query_analysis"),
                    processing_time=0.0,  # QA Chain에서 측정됨
                    confidence_score=result.get("query_analysis", {}).get(
                        "confidence", 0.0
                    ),
                    source_nodes=[
                        doc.get("metadata", {}).get("node_id", "")
                        for doc in result.get("source_documents", [])
                    ],
                )
            else:
                return result.get("answer", "")

        except Exception as e:
            logger.error(f"❌ QA Chain failed: {e}")
            logger.info("🔄 Falling back to original Pipeline method...")

            # 폴백: 원래 메서드 사용
            return original_ask(query, return_context)

    # 메서드 교체
    pipeline.ask = optimized_ask
    pipeline._qa_chain = qa_chain  # 참조 보관

    logger.info("✅ Pipeline LLM replaced with QA Chain successfully")
    logger.info("💡 Use pipeline.ask() as usual - now with LangChain optimization!")

    return pipeline


# 검증 및 테스트 함수들
def validate_qa_chain_integration(
    config_manager: "GraphRAGConfigManager",
) -> Dict[str, Any]:
    """QA Chain 통합 검증 - HuggingFace ID 지원"""

    validation_result = {
        "status": "unknown",
        "checks": {},
        "recommendations": [],
        "errors": [],
    }

    try:
        # 1. 설정 검증
        validation_result["checks"]["config_valid"] = config_manager is not None

        if config_manager:
            config = config_manager.config

            # 2. 경로 검증
            graph_path = Path(config.graph.unified_graph_path)
            vector_path = Path(config.graph.vector_store_path)

            validation_result["checks"]["graph_exists"] = graph_path.exists()
            validation_result["checks"]["vector_store_exists"] = vector_path.exists()

            if not graph_path.exists():
                validation_result["errors"].append(
                    f"Unified graph not found: {graph_path}"
                )
                validation_result["recommendations"].append(
                    "Run unified graph builder first"
                )

            if not vector_path.exists():
                validation_result["errors"].append(
                    f"Vector store not found: {vector_path}"
                )
                validation_result["recommendations"].append(
                    "Run embedding generation first"
                )

            # 3. LLM 설정 검증 (수정된 부분)
            try:
                llm_config = config_manager.get_llm_config()
                validation_result["checks"]["llm_config_valid"] = True
                validation_result["checks"]["llm_provider"] = llm_config.get("provider")

                if llm_config.get("provider") == "huggingface_local":
                    model_path = llm_config.get("model_path")

                    # ✅ HuggingFace ID vs 로컬 경로 구분
                    if model_path:
                        # HuggingFace ID 패턴 (org/model-name)
                        if "/" in model_path and not model_path.startswith("/"):
                            validation_result["checks"][
                                "model_source"
                            ] = "huggingface_hub"
                            validation_result["checks"]["model_id"] = model_path
                            logger.info(f"✅ Using HuggingFace model ID: {model_path}")
                        else:
                            # 로컬 경로
                            validation_result["checks"]["model_source"] = "local_path"
                            if not Path(model_path).exists():
                                validation_result["errors"].append(
                                    f"Local model path not found: {model_path}"
                                )
                                validation_result["recommendations"].append(
                                    f"Download model to {model_path} or use HuggingFace ID"
                                )
                            else:
                                validation_result["checks"]["local_model_exists"] = True

            except Exception as e:
                validation_result["checks"]["llm_config_valid"] = False
                validation_result["errors"].append(f"LLM config error: {e}")

            # 4. 의존성 검증 (더 정확한 체크)
            try:
                import langchain_core
                import langchain

                validation_result["checks"]["langchain_available"] = True
                validation_result["checks"]["langchain_version"] = langchain.__version__
            except ImportError:
                validation_result["checks"]["langchain_available"] = False
                validation_result["errors"].append("LangChain not available")
                validation_result["recommendations"].append(
                    "Install LangChain: pip install langchain langchain-core"
                )

        # 5. 전체 상태 결정 (수정된 로직)
        error_count = len(validation_result["errors"])

        # 중요한 에러와 경고 구분
        critical_errors = []
        for error in validation_result["errors"]:
            if "not found" in error and "graph" in error:
                critical_errors.append(error)
            elif "LangChain not available" in error:
                critical_errors.append(error)

        if len(critical_errors) == 0:
            validation_result["status"] = "ready"
        elif len(critical_errors) <= 1:
            validation_result["status"] = "partial"
        else:
            validation_result["status"] = "not_ready"

        # 상태별 메시지 추가
        if validation_result["status"] == "ready":
            validation_result["message"] = (
                "All components ready for QA Chain integration"
            )
        elif validation_result["status"] == "partial":
            validation_result["message"] = "QA Chain can work with some limitations"
        else:
            validation_result["message"] = "Critical components missing"

    except Exception as e:
        validation_result["status"] = "error"
        validation_result["errors"].append(f"Validation failed: {e}")

    return validation_result


def print_qa_chain_integration_guide():
    """QA Chain 통합 가이드 출력"""

    guide = """
🔗 GraphRAG QA Chain Integration Guide
=====================================

1. 기본 사용법:
   ```python
   from graphrag.langchain import create_qa_chain_from_pipeline
   
   # Pipeline에서 QA Chain 생성
   qa_chain = create_qa_chain_from_pipeline(pipeline)
   
   # 질문하기
   result = qa_chain.invoke({"question": "your question"})
   print(result["answer"])
   ```

2. Pipeline 최적화:
   ```python
   from graphrag.langchain import replace_pipeline_llm_with_qa_chain
   
   # Pipeline의 ask() 메서드를 QA Chain으로 교체
   optimized_pipeline = replace_pipeline_llm_with_qa_chain(pipeline)
   
   # 기존과 동일하게 사용 (내부적으로 LangChain 최적화 적용)
   answer = optimized_pipeline.ask("your question")
   ```

3. 설정 검증:
   ```python
   from graphrag.langchain import validate_qa_chain_integration
   
   validation = validate_qa_chain_integration(config_manager)
   print(f"Status: {validation['status']}")
   ```

4. 성능 모니터링:
   ```python
   # LLM 사용 통계
   stats = qa_chain._llm.get_usage_stats()
   print(f"Total calls: {stats['total_calls']}")
   print(f"Cache hit ratio: {stats['cache_hit_ratio']:.2%}")
   ```

⚡ Performance Benefits:
- 🚀 LangChain 체인 최적화
- 💾 자동 응답 캐싱
- 🔄 실패 시 자동 재시도
- 📊 사용량 통계 추적
- 🧠 대화 히스토리 관리
"""

    print(guide)


def main():
    """QA Chain Builder 테스트"""

    if not _langchain_available:
        print("❌ LangChain not available for testing")
        return

    print("🧪 Testing QA Chain Builder...")

    try:
        # 기본 경로 설정 (테스트용)
        from pathlib import Path

        base_dir = Path(__file__).parent.parent.parent.parent
        unified_graph_path = (
            base_dir
            / "data"
            / "processed"
            / "graphs"
            / "unified"
            / "unified_knowledge_graph.json"
        )
        vector_store_path = base_dir / "data" / "processed" / "vector_store"

        # QA 체인 빌더 생성
        builder = QAChainBuilder(
            unified_graph_path=str(unified_graph_path),
            vector_store_path=str(vector_store_path),
        )

        # 빌더 정보 출력
        info = builder.get_chain_info()
        print(f"📊 Builder info:")
        print(f"   Graph path: {info['unified_graph_path']}")
        print(f"   Vector store: {info['vector_store_path']}")
        print(f"   Available chain types: {info['available_chain_types']}")

        # 기본 QA 체인 생성 테스트
        print(f"\n🔧 Testing basic QA chain creation...")

        # 실제 LLM 없이 테스트 (모킹)
        class MockLLM(BaseLanguageModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                return "테스트 답변입니다."

            def _llm_type(self):
                return "mock_llm"

        try:
            # 체인 설정
            config = QAChainConfig(
                chain_type=ChainType.BASIC_QA,
                enable_memory=False,
                max_docs_for_context=5,
                verbose=True,
            )

            # 기본 체인 생성 (실제로는 컴포넌트들이 필요)
            print(f"✅ QA chain configuration created")
            print(f"   Chain type: {config.chain_type.value}")
            print(f"   Memory enabled: {config.enable_memory}")
            print(f"   Max docs: {config.max_docs_for_context}")

        except Exception as e:
            print(f"⚠️ Chain creation test skipped: {e}")

        # 사용 가능한 체인 타입들 테스트
        print(f"\n📋 Available chain types:")
        for chain_type in ChainType:
            print(f"   • {chain_type.value}: {chain_type.name}")

        print(f"\n✅ QA Chain Builder test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
