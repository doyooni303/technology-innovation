"""
GraphRAG 메모리 관리 모듈
Memory Manager for GraphRAG System

대화형 GraphRAG 시스템의 메모리 및 컨텍스트 관리
- 대화 히스토리 관리 및 압축
- GraphRAG 특화 컨텍스트 캐싱
- 세션별 메모리 관리
- 관련성 기반 히스토리 필터링
- 토큰 제한 고려한 스마트 압축
"""

import json
import pickle
import hashlib
import logging
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta

# LangChain imports
try:
    from langchain_core.memory import BaseMemory
    from langchain.memory import (
        ConversationBufferMemory,
        ConversationBufferWindowMemory,
        ConversationSummaryMemory,
        ConversationSummaryBufferMemory,
    )
    from langchain_core.messages import (
        BaseMessage,
        HumanMessage,
        AIMessage,
        SystemMessage,
    )
    from langchain_core.language_models import BaseLanguageModel

    _langchain_available = True
except ImportError:
    _langchain_available = False
    warnings.warn("LangChain not available. Install with: pip install langchain")

    # Placeholder classes
    class BaseMemory:
        def __init__(self, *args, **kwargs):
            pass

    class BaseMessage:
        pass

    class BaseLanguageModel:
        pass


# GraphRAG imports
try:
    from ..query_analyzer import QueryAnalysisResult, QueryType, QueryComplexity
    from ..embeddings.subgraph_extractor import SubgraphResult
    from ..embeddings.context_serializer import SerializedContext
except ImportError as e:
    warnings.warn(f"Some GraphRAG components not available: {e}")

# 로깅 설정
logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """메모리 타입"""

    BUFFER = "buffer"  # 단순 버퍼 (최근 N개)
    WINDOW = "window"  # 슬라이딩 윈도우
    SUMMARY = "summary"  # 요약 기반
    SUMMARY_BUFFER = "summary_buffer"  # 요약 + 버퍼 하이브리드
    GRAPHRAG_ENHANCED = "graphrag_enhanced"  # GraphRAG 특화


class CompressionStrategy(Enum):
    """압축 전략"""

    FIFO = "fifo"  # First In First Out
    RELEVANCE = "relevance"  # 관련성 기반
    FREQUENCY = "frequency"  # 언급 빈도 기반
    SEMANTIC = "semantic"  # 의미적 유사도 기반
    HYBRID = "hybrid"  # 복합 전략


@dataclass
class MemoryConfig:
    """메모리 설정"""

    # 기본 설정
    memory_type: MemoryType = MemoryType.GRAPHRAG_ENHANCED
    memory_key: str = "chat_history"
    return_messages: bool = True

    # 용량 제한
    max_token_limit: int = 4000
    max_interactions: int = 20
    buffer_size: int = 10

    # 압축 설정
    compression_strategy: CompressionStrategy = CompressionStrategy.HYBRID
    compression_ratio: float = 0.3  # 압축시 유지할 비율

    # GraphRAG 특화 설정
    cache_subgraph_results: bool = True
    cache_ttl_hours: int = 24
    max_cached_contexts: int = 50

    # 관련성 필터링
    relevance_threshold: float = 0.3
    enable_semantic_filtering: bool = True

    # 요약 설정
    enable_summarization: bool = True
    summary_max_tokens: int = 500
    summary_overlap_tokens: int = 100


@dataclass
class ConversationTurn:
    """대화 턴 정보"""

    timestamp: datetime
    human_message: str
    ai_message: str
    query_analysis: Optional[QueryAnalysisResult] = None
    subgraph_result: Optional[SubgraphResult] = None
    serialized_context: Optional[SerializedContext] = None
    relevance_score: float = 1.0
    token_count: int = 0
    session_id: str = "default"


class GraphRAGMemoryManager:
    """GraphRAG 메모리 매니저"""

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        llm: Optional[BaseLanguageModel] = None,
        persist_directory: Optional[str] = None,
    ):
        """
        Args:
            config: 메모리 설정
            llm: 요약용 LLM (선택적)
            persist_directory: 영구 저장 디렉토리
        """
        if not _langchain_available:
            raise ImportError("LangChain is required for MemoryManager")

        self.config = config or MemoryConfig()
        self.llm = llm
        self.persist_directory = Path(persist_directory) if persist_directory else None

        # 세션별 메모리 저장소
        self.session_memories: Dict[str, BaseMemory] = {}
        self.conversation_histories: Dict[str, List[ConversationTurn]] = defaultdict(
            list
        )

        # GraphRAG 특화 캐시
        self.subgraph_cache: Dict[str, SubgraphResult] = {}
        self.context_cache: Dict[str, SerializedContext] = {}
        self.cache_timestamps: Dict[str, datetime] = {}

        # 토큰 카운터 (대략적)
        self.token_counter = self._create_token_counter()

        # 영구 저장소 설정
        if self.persist_directory:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._load_persisted_data()

        logger.info("✅ GraphRAGMemoryManager initialized")
        logger.info(f"   🧠 Memory type: {self.config.memory_type.value}")
        logger.info(f"   💾 Persist: {'Yes' if self.persist_directory else 'No'}")

    def _create_token_counter(self) -> Callable[[str], int]:
        """토큰 카운터 함수 생성"""
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 기본 인코딩
            return lambda text: len(encoding.encode(text))
        except ImportError:
            # tiktoken이 없으면 단순 추정
            logger.warning("⚠️ tiktoken not available, using simple token estimation")
            return lambda text: len(text.split()) * 1.3  # 단어수 × 1.3 (대략적)

    def get_or_create_memory(self, session_id: str = "default") -> BaseMemory:
        """세션별 메모리 조회 또는 생성"""

        if session_id not in self.session_memories:
            logger.info(f"🧠 Creating new memory for session: {session_id}")
            self.session_memories[session_id] = self._create_memory_instance()

        return self.session_memories[session_id]

    def _create_memory_instance(self) -> BaseMemory:
        """메모리 인스턴스 생성"""

        memory_type = self.config.memory_type

        if memory_type == MemoryType.BUFFER:
            return ConversationBufferMemory(
                memory_key=self.config.memory_key,
                return_messages=self.config.return_messages,
            )

        elif memory_type == MemoryType.WINDOW:
            return ConversationBufferWindowMemory(
                k=self.config.buffer_size,
                memory_key=self.config.memory_key,
                return_messages=self.config.return_messages,
            )

        elif memory_type == MemoryType.SUMMARY:
            if not self.llm:
                logger.warning(
                    "⚠️ LLM required for summary memory, falling back to buffer"
                )
                return self._create_fallback_memory()

            return ConversationSummaryMemory(
                llm=self.llm,
                memory_key=self.config.memory_key,
                return_messages=self.config.return_messages,
                max_token_limit=self.config.summary_max_tokens,
            )

        elif memory_type == MemoryType.SUMMARY_BUFFER:
            if not self.llm:
                logger.warning(
                    "⚠️ LLM required for summary buffer memory, falling back to window"
                )
                return ConversationBufferWindowMemory(
                    k=self.config.buffer_size,
                    memory_key=self.config.memory_key,
                    return_messages=self.config.return_messages,
                )

            return ConversationSummaryBufferMemory(
                llm=self.llm,
                memory_key=self.config.memory_key,
                return_messages=self.config.return_messages,
                max_token_limit=self.config.max_token_limit,
            )

        elif memory_type == MemoryType.GRAPHRAG_ENHANCED:
            return self._create_graphrag_enhanced_memory()

        else:
            logger.warning(f"⚠️ Unknown memory type: {memory_type}, using buffer")
            return self._create_fallback_memory()

    def _create_fallback_memory(self) -> BaseMemory:
        """폴백 메모리 생성"""
        return ConversationBufferWindowMemory(
            k=5,  # 최근 5개만
            memory_key=self.config.memory_key,
            return_messages=self.config.return_messages,
        )

    def _create_graphrag_enhanced_memory(self) -> BaseMemory:
        """GraphRAG 특화 메모리 생성"""
        # 기본적으로는 SummaryBuffer를 베이스로 하되, 추가 기능 구현
        if self.llm:
            base_memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                memory_key=self.config.memory_key,
                return_messages=self.config.return_messages,
                max_token_limit=self.config.max_token_limit,
            )
        else:
            base_memory = ConversationBufferWindowMemory(
                k=self.config.buffer_size,
                memory_key=self.config.memory_key,
                return_messages=self.config.return_messages,
            )

        return base_memory

    def add_conversation_turn(
        self,
        human_message: str,
        ai_message: str,
        session_id: str = "default",
        query_analysis: Optional[QueryAnalysisResult] = None,
        subgraph_result: Optional[SubgraphResult] = None,
        serialized_context: Optional[SerializedContext] = None,
    ) -> None:
        """대화 턴 추가 (GraphRAG 특화 정보 포함)"""

        # 기본 메모리에 추가
        memory = self.get_or_create_memory(session_id)
        memory.save_context({"input": human_message}, {"output": ai_message})

        # GraphRAG 특화 정보 저장
        turn = ConversationTurn(
            timestamp=datetime.now(),
            human_message=human_message,
            ai_message=ai_message,
            query_analysis=query_analysis,
            subgraph_result=subgraph_result,
            serialized_context=serialized_context,
            token_count=self.token_counter(human_message + ai_message),
            session_id=session_id,
        )

        self.conversation_histories[session_id].append(turn)

        # 캐시 저장
        if self.config.cache_subgraph_results and subgraph_result:
            self._cache_subgraph_result(human_message, subgraph_result)

        if self.config.cache_subgraph_results and serialized_context:
            self._cache_serialized_context(human_message, serialized_context)

        # 메모리 압축 검사
        self._check_and_compress_memory(session_id)

        logger.info(f"💭 Added conversation turn to session: {session_id}")

    def _cache_subgraph_result(self, query: str, result: SubgraphResult) -> None:
        """서브그래프 결과 캐싱"""
        cache_key = self._generate_cache_key(query)
        self.subgraph_cache[cache_key] = result
        self.cache_timestamps[cache_key] = datetime.now()

        # 캐시 크기 제한
        if len(self.subgraph_cache) > self.config.max_cached_contexts:
            self._clean_old_cache()

    def _cache_serialized_context(self, query: str, context: SerializedContext) -> None:
        """직렬화된 컨텍스트 캐싱"""
        cache_key = self._generate_cache_key(query)
        self.context_cache[cache_key] = context
        self.cache_timestamps[cache_key] = datetime.now()

    def _generate_cache_key(self, query: str) -> str:
        """캐시 키 생성"""
        # 쿼리 정규화 후 해시
        normalized_query = query.lower().strip()
        return hashlib.md5(normalized_query.encode()).hexdigest()

    def _clean_old_cache(self) -> None:
        """오래된 캐시 정리"""
        now = datetime.now()
        ttl_threshold = now - timedelta(hours=self.config.cache_ttl_hours)

        expired_keys = [
            key
            for key, timestamp in self.cache_timestamps.items()
            if timestamp < ttl_threshold
        ]

        for key in expired_keys:
            self.subgraph_cache.pop(key, None)
            self.context_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)

        # 크기 제한
        if len(self.subgraph_cache) > self.config.max_cached_contexts:
            # 오래된 순으로 정렬하여 제거
            sorted_keys = sorted(self.cache_timestamps.items(), key=lambda x: x[1])

            excess_count = len(self.subgraph_cache) - self.config.max_cached_contexts
            for key, _ in sorted_keys[:excess_count]:
                self.subgraph_cache.pop(key, None)
                self.context_cache.pop(key, None)
                self.cache_timestamps.pop(key, None)

        if expired_keys:
            logger.info(f"🗑️ Cleaned {len(expired_keys)} expired cache entries")

    def get_cached_result(
        self, query: str
    ) -> Optional[Tuple[SubgraphResult, SerializedContext]]:
        """캐시된 결과 조회"""
        cache_key = self._generate_cache_key(query)

        # TTL 검사
        if cache_key in self.cache_timestamps:
            cache_time = self.cache_timestamps[cache_key]
            now = datetime.now()

            if now - cache_time > timedelta(hours=self.config.cache_ttl_hours):
                # 만료된 캐시 제거
                self.subgraph_cache.pop(cache_key, None)
                self.context_cache.pop(cache_key, None)
                self.cache_timestamps.pop(cache_key, None)
                return None

        subgraph_result = self.subgraph_cache.get(cache_key)
        context_result = self.context_cache.get(cache_key)

        if subgraph_result and context_result:
            logger.info(f"✅ Cache hit for query: {query[:50]}...")
            return subgraph_result, context_result

        return None

    def _check_and_compress_memory(self, session_id: str) -> None:
        """메모리 압축 검사 및 실행"""

        history = self.conversation_histories[session_id]

        # 토큰 수 계산
        total_tokens = sum(turn.token_count for turn in history)

        # 압축 필요성 확인
        if (
            total_tokens > self.config.max_token_limit
            or len(history) > self.config.max_interactions
        ):

            logger.info(f"🗜️ Compressing memory for session {session_id}")
            self._compress_memory(session_id)

    def _compress_memory(self, session_id: str) -> None:
        """메모리 압축 실행"""

        history = self.conversation_histories[session_id]
        strategy = self.config.compression_strategy

        if strategy == CompressionStrategy.FIFO:
            compressed_history = self._compress_fifo(history)
        elif strategy == CompressionStrategy.RELEVANCE:
            compressed_history = self._compress_by_relevance(history)
        elif strategy == CompressionStrategy.FREQUENCY:
            compressed_history = self._compress_by_frequency(history)
        elif strategy == CompressionStrategy.SEMANTIC:
            compressed_history = self._compress_by_semantic_similarity(history)
        elif strategy == CompressionStrategy.HYBRID:
            compressed_history = self._compress_hybrid(history)
        else:
            compressed_history = self._compress_fifo(history)  # 폴백

        # 압축된 히스토리로 교체
        self.conversation_histories[session_id] = compressed_history

        # 기본 메모리도 재구성
        self._rebuild_base_memory(session_id, compressed_history)

        logger.info(f"✅ Compressed {len(history)} → {len(compressed_history)} turns")

    def _compress_fifo(self, history: List[ConversationTurn]) -> List[ConversationTurn]:
        """FIFO 압축 (최근 것만 유지)"""
        target_size = int(len(history) * self.config.compression_ratio)
        target_size = max(target_size, 3)  # 최소 3개는 유지
        return history[-target_size:]

    def _compress_by_relevance(
        self, history: List[ConversationTurn]
    ) -> List[ConversationTurn]:
        """관련성 기반 압축"""
        # 관련성 점수 기반 정렬
        scored_history = []
        for turn in history:
            score = turn.relevance_score

            # 최근성 보너스
            age_hours = (datetime.now() - turn.timestamp).total_seconds() / 3600
            recency_bonus = max(0, 1.0 - age_hours / 24)  # 24시간내 보너스

            total_score = score + recency_bonus * 0.3
            scored_history.append((total_score, turn))

        # 상위 N개 선택
        scored_history.sort(key=lambda x: x[0], reverse=True)
        target_size = int(len(history) * self.config.compression_ratio)
        target_size = max(target_size, 3)

        selected_turns = [turn for _, turn in scored_history[:target_size]]

        # 시간순 정렬
        selected_turns.sort(key=lambda x: x.timestamp)
        return selected_turns

    def _compress_by_frequency(
        self, history: List[ConversationTurn]
    ) -> List[ConversationTurn]:
        """빈도 기반 압축 (자주 언급되는 주제 우선)"""
        # 간단히 키워드 빈도로 계산
        keyword_freq = defaultdict(int)

        for turn in history:
            if turn.query_analysis and turn.query_analysis.keywords:
                for keyword in turn.query_analysis.keywords:
                    keyword_freq[keyword.lower()] += 1

        # 각 턴의 빈도 점수 계산
        scored_turns = []
        for turn in history:
            score = 0
            if turn.query_analysis and turn.query_analysis.keywords:
                for keyword in turn.query_analysis.keywords:
                    score += keyword_freq[keyword.lower()]

            scored_turns.append((score, turn))

        # 상위 N개 선택
        scored_turns.sort(key=lambda x: x[0], reverse=True)
        target_size = int(len(history) * self.config.compression_ratio)
        target_size = max(target_size, 3)

        selected_turns = [turn for _, turn in scored_turns[:target_size]]
        selected_turns.sort(key=lambda x: x.timestamp)
        return selected_turns

    def _compress_by_semantic_similarity(
        self, history: List[ConversationTurn]
    ) -> List[ConversationTurn]:
        """의미적 유사도 기반 압축 (유사한 대화는 대표적인 것만)"""
        # 간단한 구현: 최근 것들 위주로 선택
        return self._compress_fifo(history)

    def _compress_hybrid(
        self, history: List[ConversationTurn]
    ) -> List[ConversationTurn]:
        """하이브리드 압축 (여러 전략 조합)"""
        # 1단계: 관련성 기반 필터링
        relevant_turns = [
            turn
            for turn in history
            if turn.relevance_score >= self.config.relevance_threshold
        ]

        if not relevant_turns:
            relevant_turns = history  # 폴백

        # 2단계: 최근성 고려
        if len(relevant_turns) > self.config.buffer_size:
            # 최근 절반 + 관련성 높은 나머지
            recent_count = len(relevant_turns) // 2
            recent_turns = relevant_turns[-recent_count:]

            older_turns = relevant_turns[:-recent_count]
            older_turns.sort(key=lambda x: x.relevance_score, reverse=True)

            target_older = self.config.buffer_size - recent_count
            selected_older = older_turns[:target_older] if target_older > 0 else []

            final_turns = selected_older + recent_turns
            final_turns.sort(key=lambda x: x.timestamp)
            return final_turns

        return relevant_turns

    def _rebuild_base_memory(
        self, session_id: str, compressed_history: List[ConversationTurn]
    ) -> None:
        """압축된 히스토리로 기본 메모리 재구성"""

        # 새로운 메모리 인스턴스 생성
        new_memory = self._create_memory_instance()

        # 압축된 히스토리로 메모리 재구성
        for turn in compressed_history:
            new_memory.save_context(
                {"input": turn.human_message}, {"output": turn.ai_message}
            )

        # 기존 메모리 교체
        self.session_memories[session_id] = new_memory

    def get_conversation_summary(self, session_id: str = "default") -> str:
        """대화 요약 생성"""

        if session_id not in self.conversation_histories:
            return "No conversation history found."

        history = self.conversation_histories[session_id]

        if not history:
            return "No conversations yet."

        # 간단한 요약 생성
        total_turns = len(history)
        recent_topics = []

        # 최근 5개 대화의 주요 키워드
        for turn in history[-5:]:
            if turn.query_analysis and turn.query_analysis.keywords:
                recent_topics.extend(turn.query_analysis.keywords[:3])

        topic_counts = defaultdict(int)
        for topic in recent_topics:
            topic_counts[topic.lower()] += 1

        top_topics = [
            topic
            for topic, _ in sorted(
                topic_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]
        ]

        summary = f"총 {total_turns}개 대화 진행됨."
        if top_topics:
            summary += f" 주요 관심 주제: {', '.join(top_topics)}"

        return summary

    def clear_session(self, session_id: str) -> None:
        """세션 메모리 초기화"""
        self.session_memories.pop(session_id, None)
        self.conversation_histories.pop(session_id, None)
        logger.info(f"🗑️ Cleared session: {session_id}")

    def get_memory_stats(self, session_id: str = "default") -> Dict[str, Any]:
        """메모리 통계 정보"""

        stats = {
            "session_id": session_id,
            "total_turns": 0,
            "total_tokens": 0,
            "cache_hits": len(self.subgraph_cache),
            "memory_type": self.config.memory_type.value,
        }

        if session_id in self.conversation_histories:
            history = self.conversation_histories[session_id]
            stats["total_turns"] = len(history)
            stats["total_tokens"] = sum(turn.token_count for turn in history)

            if history:
                stats["oldest_conversation"] = history[0].timestamp.isoformat()
                stats["newest_conversation"] = history[-1].timestamp.isoformat()

        return stats

    def _load_persisted_data(self) -> None:
        """영구 저장된 데이터 로드"""
        if not self.persist_directory:
            return

        try:
            # 대화 히스토리 로드
            history_file = self.persist_directory / "conversation_histories.pkl"
            if history_file.exists():
                with open(history_file, "rb") as f:
                    self.conversation_histories = pickle.load(f)
                logger.info(f"📂 Loaded conversation histories")

            # 캐시 로드
            cache_file = self.persist_directory / "cache_data.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                    self.subgraph_cache = cache_data.get("subgraph_cache", {})
                    self.context_cache = cache_data.get("context_cache", {})
                    self.cache_timestamps = cache_data.get("cache_timestamps", {})
                logger.info(f"📂 Loaded cache data")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load persisted data: {e}")

    def save_to_disk(self) -> None:
        """데이터를 디스크에 저장"""
        if not self.persist_directory:
            return

        try:
            # 대화 히스토리 저장
            history_file = self.persist_directory / "conversation_histories.pkl"
            with open(history_file, "wb") as f:
                pickle.dump(dict(self.conversation_histories), f)

            # 캐시 저장
            cache_file = self.persist_directory / "cache_data.pkl"
            cache_data = {
                "subgraph_cache": self.subgraph_cache,
                "context_cache": self.context_cache,
                "cache_timestamps": self.cache_timestamps,
            }
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)

            logger.info(f"💾 Saved memory data to disk")

        except Exception as e:
            logger.error(f"❌ Failed to save memory data: {e}")


# 편의 함수들
def create_memory_manager(
    memory_type: str = "graphrag_enhanced",
    max_token_limit: int = 4000,
    llm: Optional[BaseLanguageModel] = None,
    persist_directory: Optional[str] = None,
    **kwargs,
) -> GraphRAGMemoryManager:
    """메모리 매니저 생성 편의 함수"""

    config = MemoryConfig(
        memory_type=MemoryType(memory_type), max_token_limit=max_token_limit, **kwargs
    )

    return GraphRAGMemoryManager(
        config=config, llm=llm, persist_directory=persist_directory
    )


def main():
    """GraphRAGMemoryManager 테스트"""

    if not _langchain_available:
        print("❌ LangChain not available for testing")
        return

    print("🧪 Testing GraphRAGMemoryManager...")

    try:
        # 메모리 매니저 생성
        memory_manager = create_memory_manager(
            memory_type="buffer", max_token_limit=1000, max_interactions=5
        )

        # 테스트 대화 추가
        test_conversations = [
            (
                "배터리 SoC 예측 기법에 대해 알려주세요",
                "SoC 예측에는 칼만 필터, LSTM 등이 사용됩니다.",
            ),
            (
                "김철수 교수의 연구 분야는?",
                "김철수 교수는 배터리 관리 시스템을 연구합니다.",
            ),
            (
                "전기차 충전 기술 동향은?",
                "무선 충전, 고속 충전 기술이 발전하고 있습니다.",
            ),
        ]

        session_id = "test_session"

        for human_msg, ai_msg in test_conversations:
            memory_manager.add_conversation_turn(
                human_message=human_msg, ai_message=ai_msg, session_id=session_id
            )
            print(f"💭 Added: {human_msg[:30]}...")

        # 메모리 조회
        memory = memory_manager.get_or_create_memory(session_id)
        print(f"🧠 Memory type: {type(memory).__name__}")

        # 통계 확인
        stats = memory_manager.get_memory_stats(session_id)
        print(f"📊 Memory stats: {stats}")

        # 대화 요약
        summary = memory_manager.get_conversation_summary(session_id)
        print(f"📝 Summary: {summary}")

        # 캐시 테스트
        cached_result = memory_manager.get_cached_result("배터리 SoC")
        print(f"💾 Cache result: {'Found' if cached_result else 'Not found'}")

        print(f"\n✅ GraphRAGMemoryManager test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
