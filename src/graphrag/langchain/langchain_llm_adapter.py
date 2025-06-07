"""
GraphRAG LangChain LLM 어댑터
LangChain LLM Adapter for GraphRAG System

GraphRAG의 LocalLLMManager를 LangChain BaseLanguageModel로 변환
- YAML 설정 완전 호환
- 스트리밍 지원
- 배치 처리 최적화
- 에러 핸들링 및 재시도
- 토큰 사용량 추적
"""

import asyncio
import logging
import warnings
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator, Mapping
from dataclasses import dataclass
from enum import Enum
import time

# LangChain imports
try:
    from langchain_core.language_models.llms import LLM
    from langchain_core.language_models.base import BaseLanguageModel
    from langchain_core.outputs import Generation, LLMResult, GenerationChunk
    from langchain_core.callbacks.manager import (
        CallbackManagerForLLMRun,
        AsyncCallbackManagerForLLMRun,
    )
    from langchain_core.pydantic_v1 import Field, root_validator

    _langchain_available = True
except ImportError:
    _langchain_available = False
    warnings.warn("LangChain not available for LLM adapter")

# GraphRAG imports
try:
    from ..graphrag_pipeline import LocalLLMManager
    from ..config_manager import GraphRAGConfigManager
except ImportError as e:
    warnings.warn(f"GraphRAG pipeline components not available: {e}")

# 로깅 설정
logger = logging.getLogger(__name__)


class AdapterMode(Enum):
    """어댑터 모드"""

    DIRECT = "direct"  # 직접 호출
    BATCHED = "batched"  # 배치 처리
    STREAMING = "streaming"  # 스트리밍
    CACHED = "cached"  # 캐싱 활용


@dataclass
class LLMUsageStats:
    """LLM 사용 통계"""

    total_calls: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    average_time: float = 0.0
    cache_hits: int = 0
    failed_calls: int = 0

    def update_call(self, tokens: int, call_time: float, from_cache: bool = False):
        """호출 통계 업데이트"""
        self.total_calls += 1
        if not from_cache:
            self.total_tokens += tokens
        self.total_time += call_time
        self.average_time = self.total_time / self.total_calls
        if from_cache:
            self.cache_hits += 1

    def update_failure(self):
        """실패 통계 업데이트"""
        self.failed_calls += 1


class GraphRAGLLMAdapter(LLM):
    """수정된 GraphRAG LLM Adapter - FieldInfo 오류 해결"""

    # LangChain Pydantic 필드들
    llm_manager: Any = Field(description="GraphRAG LocalLLMManager instance")
    temperature: float = Field(default=0.1, description="Generation temperature")
    max_tokens: int = Field(default=1000, description="Maximum tokens to generate")

    # 어댑터 설정
    mode: AdapterMode = Field(default=AdapterMode.DIRECT, description="Adapter mode")
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries")

    # 내부 상태 (exclude=True로 직렬화에서 제외)
    _cache: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    _cache_timestamps: Dict[str, float] = Field(default_factory=dict, exclude=True)
    _stats: LLMUsageStats = Field(default_factory=LLMUsageStats, exclude=True)
    _is_loaded: bool = Field(default=False, exclude=True)

    class Config:
        """Pydantic 설정"""

        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def validate_llm_manager(cls, values):
        """LLM 매니저 검증"""
        llm_manager = values.get("llm_manager")
        if llm_manager is None:
            raise ValueError("llm_manager is required")
        return values

    def __init__(self, **kwargs):
        """어댑터 초기화"""
        super().__init__(**kwargs)

        if not _langchain_available:
            raise ImportError("LangChain is required for LLM adapter")

        # ✅ Pydantic Field 값들을 안전하게 검증
        self._validate_field_values()

        # 통계 초기화
        self._stats = LLMUsageStats()
        self._cache = {}
        self._cache_timestamps = {}
        self._is_loaded = False

        logger.info("✅ GraphRAGLLMAdapter initialized")
        logger.info(f"   🌡️ Temperature: {self._get_safe_temperature()}")
        logger.info(f"   📏 Max tokens: {self._get_safe_max_tokens()}")
        logger.info(
            f"   🔧 Mode: {self.mode.value if hasattr(self.mode, 'value') else self.mode}"
        )
        logger.info(f"   💾 Caching: {self.enable_caching}")

    def _validate_field_values(self):
        """Pydantic Field 값들을 검증하고 수정"""

        # temperature 검증
        if not isinstance(getattr(self, "temperature", None), (int, float)):
            logger.warning("⚠️ Invalid temperature field, setting to 0.1")
            object.__setattr__(self, "temperature", 0.1)

        # max_tokens 검증
        if not isinstance(getattr(self, "max_tokens", None), (int, float)):
            logger.warning("⚠️ Invalid max_tokens field, setting to 1000")
            object.__setattr__(self, "max_tokens", 1000)

        # mode 검증
        if not hasattr(getattr(self, "mode", None), "value"):
            logger.warning("⚠️ Invalid mode field, setting to DIRECT")
            from enum import Enum

            if hasattr(self, "mode") and isinstance(self.mode, str):
                # 문자열을 Enum으로 변환 시도
                try:
                    object.__setattr__(self, "mode", AdapterMode(self.mode))
                except (ValueError, AttributeError):
                    object.__setattr__(self, "mode", AdapterMode.DIRECT)
            else:
                object.__setattr__(self, "mode", AdapterMode.DIRECT)

    def _get_safe_temperature(self) -> float:
        """안전한 temperature 값 반환"""
        temp = getattr(self, "temperature", 0.1)
        return float(temp) if isinstance(temp, (int, float)) else 0.1

    def _get_safe_max_tokens(self) -> int:
        """안전한 max_tokens 값 반환"""
        tokens = getattr(self, "max_tokens", 1000)
        return int(tokens) if isinstance(tokens, (int, float)) else 1000

    @property
    def _llm_type(self) -> str:
        """LangChain LLM 타입 식별자"""
        return "graphrag_llm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """LangChain이 사용하는 식별 파라미터 - 안전한 값 추출"""
        return {
            "temperature": self._get_safe_temperature(),
            "max_tokens": self._get_safe_max_tokens(),
            "mode": self.mode.value if hasattr(self.mode, "value") else str(self.mode),
            "model_path": getattr(
                getattr(self.llm_manager, "config", None), "model_path", "unknown"
            ),
        }

    def _ensure_loaded(self) -> None:
        """LLM 로드 확인 및 지연 로딩"""
        if not self._is_loaded and not self.llm_manager.is_loaded:
            logger.info("📥 Loading LLM model (lazy loading)...")
            try:
                self.llm_manager.load_model()
                self._is_loaded = True
                logger.info("✅ LLM model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load LLM model: {e}")
                raise

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """동기 LLM 호출 (LangChain 인터페이스)"""

        start_time = time.time()

        try:
            # 1. 캐시 확인
            if self.enable_caching:
                cached_response = self._get_from_cache(prompt)
                if cached_response is not None:
                    call_time = time.time() - start_time
                    self._stats.update_call(
                        tokens=len(cached_response.split()),
                        call_time=call_time,
                        from_cache=True,
                    )
                    logger.debug(f"💾 Cache hit for prompt: {prompt[:50]}...")
                    return cached_response

            # 2. LLM 로드 확인
            self._ensure_loaded()

            # 3. 파라미터 준비
            generation_kwargs = self._prepare_generation_kwargs(stop, **kwargs)

            # 4. 모드별 처리
            if self.mode == AdapterMode.BATCHED:
                response = self._call_batched([prompt], **generation_kwargs)[0]
            elif self.mode == AdapterMode.STREAMING:
                # 스트리밍을 동기로 변환
                response = self._call_streaming_sync(prompt, **generation_kwargs)
            else:  # DIRECT 또는 CACHED
                response = self._call_direct(prompt, **generation_kwargs)

            # 5. 응답 후처리
            response = self._post_process_response(response)

            # 6. 캐시 저장
            if self.enable_caching:
                self._save_to_cache(prompt, response)

            # 7. 통계 업데이트
            call_time = time.time() - start_time
            self._stats.update_call(tokens=len(response.split()), call_time=call_time)

            # 8. 콜백 호출 (LangChain)
            if run_manager:
                run_manager.on_llm_end(
                    LLMResult(
                        generations=[[Generation(text=response)]],
                        llm_output={"model_name": self._llm_type},
                    )
                )

            logger.debug(f"✅ LLM call completed in {call_time:.2f}s")
            return response

        except Exception as e:
            self._stats.update_failure()
            logger.error(f"❌ LLM call failed: {e}")

            # 재시도 로직
            if self.max_retries > 0:
                return self._retry_call(prompt, stop, run_manager, **kwargs)
            else:
                raise

    def _call_direct(self, prompt: str, **kwargs) -> str:
        """직접 LLM 호출"""
        try:
            response = self.llm_manager.generate(
                prompt=prompt, max_length=kwargs.get("max_length", self.max_tokens)
            )

            if not isinstance(response, str):
                logger.warning(f"⚠️ LLM returned non-string: {type(response)}")
                response = str(response)

            return response

        except Exception as e:
            logger.error(f"❌ Direct LLM call failed: {e}")
            raise

    def _call_batched(self, prompts: List[str], **kwargs) -> List[str]:
        """배치 LLM 호출"""
        responses = []

        for prompt in prompts:
            try:
                response = self._call_direct(prompt, **kwargs)
                responses.append(response)
            except Exception as e:
                logger.warning(f"⚠️ Batch item failed: {e}")
                responses.append(f"배치 처리 중 오류가 발생했습니다: {str(e)}")

        return responses

    def _call_streaming_sync(self, prompt: str, **kwargs) -> str:
        """스트리밍을 동기적으로 처리"""
        # 현재 LocalLLMManager는 스트리밍을 지원하지 않으므로
        # 일반 호출로 폴백
        logger.debug("📡 Streaming not supported, falling back to direct call")
        return self._call_direct(prompt, **kwargs)

    def _retry_call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """재시도 로직"""

        last_error = None

        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔄 Retry attempt {attempt + 1}/{self.max_retries}")

                # 재시도 지연
                if attempt > 0:
                    time.sleep(self.retry_delay * attempt)

                # 간단한 직접 호출로 재시도
                response = self._call_direct(prompt, max_length=self.max_tokens)

                logger.info(f"✅ Retry successful on attempt {attempt + 1}")
                return response

            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ Retry attempt {attempt + 1} failed: {e}")

        # 모든 재시도 실패 시 폴백 응답
        logger.error(f"❌ All retry attempts failed. Last error: {last_error}")
        return self._generate_fallback_response(prompt, str(last_error))

    def _generate_fallback_response(self, prompt: str, error_msg: str) -> str:
        """폴백 응답 생성"""
        return (
            f"죄송합니다. 현재 응답을 생성할 수 없습니다.\n"
            f"오류: {error_msg}\n\n"
            f"잠시 후 다시 시도해주세요."
        )

    def _prepare_generation_kwargs(
        self, stop: Optional[List[str]], **kwargs
    ) -> Dict[str, Any]:
        """생성 파라미터 준비 - FieldInfo 오류 수정"""

        # ✅ Pydantic Field 값을 안전하게 추출
        max_tokens_value = getattr(self, "max_tokens", 1000)
        temperature_value = getattr(self, "temperature", 0.1)

        # FieldInfo 객체인 경우 기본값 사용
        if not isinstance(max_tokens_value, (int, float)):
            max_tokens_value = 1000
            logger.warning("⚠️ max_tokens is not a numeric value, using default 1000")

        if not isinstance(temperature_value, (int, float)):
            temperature_value = 0.1
            logger.warning("⚠️ temperature is not a numeric value, using default 0.1")

        generation_kwargs = {
            "max_length": kwargs.get("max_tokens", max_tokens_value),
            "temperature": kwargs.get("temperature", temperature_value),
        }

        # stop 토큰은 현재 LocalLLMManager에서 지원하지 않음
        if stop:
            logger.debug(f"⚠️ Stop tokens not supported: {stop}")

        logger.debug(f"🔧 Generation kwargs: {generation_kwargs}")
        return generation_kwargs

    def _post_process_response(self, response: str) -> str:
        """응답 후처리"""
        if not isinstance(response, str):
            response = str(response)

        # 기본 정리
        response = response.strip()

        # 빈 응답 처리
        if not response:
            response = "죄송합니다. 응답을 생성할 수 없었습니다."

        # 길이 제한
        max_chars = self.max_tokens * 4  # 대략적 추정
        if len(response) > max_chars:
            response = response[:max_chars] + "..."
            logger.debug(f"📏 Response truncated to {max_chars} characters")

        return response

    def _get_cache_key(self, prompt: str) -> str:
        """캐시 키 생성"""
        import hashlib

        key_data = f"{prompt}_{self.temperature}_{self.max_tokens}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_from_cache(self, prompt: str) -> Optional[str]:
        """캐시에서 응답 조회"""
        if not self.enable_caching:
            return None

        cache_key = self._get_cache_key(prompt)

        if cache_key in self._cache:
            # TTL 확인
            cache_time = self._cache_timestamps.get(cache_key, 0)
            current_time = time.time()

            if current_time - cache_time < self.cache_ttl:
                return self._cache[cache_key]
            else:
                # 만료된 캐시 제거
                del self._cache[cache_key]
                del self._cache_timestamps[cache_key]

        return None

    def _save_to_cache(self, prompt: str, response: str) -> None:
        """캐시에 응답 저장"""
        if not self.enable_caching:
            return

        cache_key = self._get_cache_key(prompt)

        self._cache[cache_key] = response
        self._cache_timestamps[cache_key] = time.time()

        # 캐시 크기 제한
        if len(self._cache) > 100:
            # 가장 오래된 캐시 제거
            oldest_key = min(self._cache_timestamps, key=self._cache_timestamps.get)
            del self._cache[oldest_key]
            del self._cache_timestamps[oldest_key]

    # ========================================================================
    # 비동기 메서드들 (LangChain 호환성)
    # ========================================================================

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """비동기 LLM 호출"""

        # 현재는 동기 메서드를 비동기로 래핑
        loop = asyncio.get_event_loop()

        # 동기 콜백 매니저로 변환
        sync_run_manager = None

        return await loop.run_in_executor(
            None, self._call, prompt, stop, sync_run_manager, **kwargs
        )

    # ========================================================================
    # 스트리밍 지원 (향후 확장용)
    # ========================================================================

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """스트리밍 생성 (현재는 미지원)"""

        # 현재 LocalLLMManager는 스트리밍을 지원하지 않으므로
        # 일반 응답을 청크로 나눠서 반환
        logger.debug("📡 Streaming not fully supported, simulating...")

        response = self._call(prompt, stop, run_manager, **kwargs)

        # 응답을 청크로 나누기
        chunk_size = 50
        for i in range(0, len(response), chunk_size):
            chunk = response[i : i + chunk_size]
            yield GenerationChunk(text=chunk)

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """비동기 스트리밍 생성"""

        # 동기 스트리밍을 비동기로 래핑
        loop = asyncio.get_event_loop()

        for chunk in self._stream(prompt, stop, None, **kwargs):
            yield chunk
            await asyncio.sleep(0)  # 이벤트 루프 양보

    # ========================================================================
    # 유틸리티 메서드들
    # ========================================================================

    def get_usage_stats(self) -> Dict[str, Any]:
        """사용 통계 반환"""
        return {
            "total_calls": self._stats.total_calls,
            "total_tokens": self._stats.total_tokens,
            "total_time": round(self._stats.total_time, 2),
            "average_time": round(self._stats.average_time, 2),
            "cache_hits": self._stats.cache_hits,
            "failed_calls": self._stats.failed_calls,
            "cache_hit_ratio": (
                self._stats.cache_hits / max(1, self._stats.total_calls)
            ),
            "success_rate": (
                (self._stats.total_calls - self._stats.failed_calls)
                / max(1, self._stats.total_calls)
            ),
        }

    def clear_cache(self) -> None:
        """캐시 초기화"""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("🗑️ LLM adapter cache cleared")

    def update_config(self, **kwargs) -> None:
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"📝 Updated {key} = {value}")

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환 - 안전한 값 추출"""
        return {
            "adapter_type": self._llm_type,
            "mode": self.mode.value if hasattr(self.mode, "value") else str(self.mode),
            "temperature": self._get_safe_temperature(),
            "max_tokens": self._get_safe_max_tokens(),
            "caching_enabled": getattr(self, "enable_caching", True),
            "is_loaded": getattr(self, "_is_loaded", False),
            "llm_manager_loaded": (
                self.llm_manager.is_loaded if self.llm_manager else False
            ),
            "model_path": (
                getattr(
                    getattr(self.llm_manager, "config", None), "model_path", "unknown"
                )
                if self.llm_manager
                else "unknown"
            ),
        }

    def health_check(self) -> Dict[str, Any]:
        """어댑터 상태 확인"""
        try:
            # 간단한 테스트 호출
            test_response = self._call("Test", max_tokens=10)
            is_healthy = len(test_response) > 0

            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "model_loaded": self._is_loaded,
                "llm_manager_available": self.llm_manager is not None,
                "test_response_length": len(test_response),
                "last_check": time.time(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": self._is_loaded,
                "llm_manager_available": self.llm_manager is not None,
                "last_check": time.time(),
            }


# ============================================================================
# 팩토리 함수들
# ============================================================================


def create_llm_adapter(
    config_manager: "GraphRAGConfigManager",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    mode: Union[AdapterMode, str] = AdapterMode.DIRECT,
    **kwargs,
) -> GraphRAGLLMAdapter:
    """LLM 어댑터 팩토리 함수 (YAML 설정 호환) - 안전한 값 전달"""

    # LLM 설정 가져오기
    llm_config = config_manager.get_llm_config()

    # LocalLLMManager 생성
    from ..graphrag_pipeline import LocalLLMManager

    llm_manager = LocalLLMManager(llm_config)

    # 모드 변환
    if isinstance(mode, str):
        try:
            mode = AdapterMode(mode)
        except ValueError:
            logger.warning(f"⚠️ Unknown mode: {mode}, using DIRECT")
            mode = AdapterMode.DIRECT

    # ✅ 안전한 값 추출 및 검증
    safe_temperature = (
        temperature if temperature is not None else llm_config.get("temperature", 0.1)
    )
    safe_max_tokens = (
        max_tokens if max_tokens is not None else llm_config.get("max_new_tokens", 1000)
    )

    # 값 타입 검증
    if not isinstance(safe_temperature, (int, float)):
        safe_temperature = 0.1
        logger.warning("⚠️ Invalid temperature in config, using 0.1")

    if not isinstance(safe_max_tokens, (int, float)):
        safe_max_tokens = 1000
        logger.warning("⚠️ Invalid max_tokens in config, using 1000")

    # 어댑터 설정
    adapter_config = {
        "llm_manager": llm_manager,
        "temperature": float(safe_temperature),
        "max_tokens": int(safe_max_tokens),
        "mode": mode,
        **kwargs,
    }

    adapter = GraphRAGLLMAdapter(**adapter_config)

    logger.info(f"✅ LLM adapter created from YAML config")
    logger.info(f"   Provider: {llm_config.get('provider')}")
    logger.info(f"   Temperature: {safe_temperature}")
    logger.info(f"   Max tokens: {safe_max_tokens}")
    logger.info(f"   Mode: {mode.value}")

    return adapter


def create_llm_adapter_from_manager(
    llm_manager: LocalLLMManager,
    temperature: float = 0.1,
    max_tokens: int = 1000,
    mode: Union[AdapterMode, str] = AdapterMode.DIRECT,
    **kwargs,
) -> GraphRAGLLMAdapter:
    """기존 LLM 매니저로부터 어댑터 생성"""

    if isinstance(mode, str):
        mode = AdapterMode(mode)

    return GraphRAGLLMAdapter(
        llm_manager=llm_manager,
        temperature=temperature,
        max_tokens=max_tokens,
        mode=mode,
        **kwargs,
    )


def main():
    """LLM 어댑터 테스트"""

    if not _langchain_available:
        print("❌ LangChain not available for testing")
        return

    print("🧪 Testing GraphRAG LLM Adapter...")

    try:
        # Mock LLM Manager 생성 (테스트용)
        class MockLLMManager:
            def __init__(self):
                self.config = type(
                    "Config",
                    (),
                    {
                        "model_path": "test-model",
                        "temperature": 0.1,
                        "max_new_tokens": 100,
                    },
                )()
                self.is_loaded = False

            def load_model(self):
                self.is_loaded = True
                print("📥 Mock LLM loaded")

            def generate(self, prompt, max_length=100):
                return f"Mock response to: {prompt[:30]}..."

        # 어댑터 생성
        mock_manager = MockLLMManager()
        adapter = GraphRAGLLMAdapter(
            llm_manager=mock_manager,
            temperature=0.1,
            max_tokens=100,
            enable_caching=True,
        )

        print(f"✅ Adapter created")

        # 모델 정보 확인
        model_info = adapter.get_model_info()
        print(f"📊 Model info:")
        for key, value in model_info.items():
            print(f"   {key}: {value}")

        # 테스트 호출
        test_prompt = "안녕하세요. 테스트 질문입니다."
        print(f"\n🔍 Testing prompt: {test_prompt}")

        try:
            response = adapter._call(test_prompt)
            print(f"✅ Response: {response}")

            # 사용 통계 확인
            stats = adapter.get_usage_stats()
            print(f"📈 Usage stats:")
            for key, value in stats.items():
                print(f"   {key}: {value}")

        except Exception as e:
            print(f"⚠️ Call test skipped: {e}")

        # 상태 확인
        health = adapter.health_check()
        print(f"\n🏥 Health check:")
        for key, value in health.items():
            print(f"   {key}: {value}")

        print(f"\n✅ GraphRAG LLM Adapter test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
