"""
임베딩 모델 추상화 및 구현
Embedding Models for GraphRAG System

다양한 임베딩 모델을 통일된 인터페이스로 제공
- BaseEmbeddingModel: 추상 기본 클래스
- SentenceTransformerModel: sentence-transformers 래퍼
- HuggingFaceModel: transformers 라이브러리 래퍼
- 모델 자동 선택 및 설정 관리
"""

import os
import json
import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass

# 로깅 설정
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """임베딩 설정 클래스"""

    model_name: str
    model_type: str  # "sentence-transformers", "huggingface", "openai", etc.
    dimension: int
    max_length: int
    batch_size: int = 32
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    cache_dir: Optional[str] = None
    model_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}


class BaseEmbeddingModel(ABC):
    """임베딩 모델 기본 추상 클래스"""

    def __init__(self, config: EmbeddingConfig):
        """
        Args:
            config: 임베딩 설정
        """
        self.config = config
        self.model = None
        self._is_loaded = False

        # 디바이스 설정
        self.device = self._determine_device(config.device)
        logger.info(f"🔧 Embedding model will use device: {self.device}")

    def _determine_device(self, device_preference: str) -> str:
        """최적 디바이스 결정"""
        if device_preference == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    return "mps"  # Apple Silicon
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        else:
            return device_preference

    @abstractmethod
    def load_model(self) -> None:
        """모델 로드 (지연 로딩)"""
        pass

    @abstractmethod
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """텍스트를 임베딩으로 변환

        Args:
            texts: 인코딩할 텍스트(들)
            batch_size: 배치 크기 (None이면 기본값 사용)
            show_progress: 진행률 표시 여부

        Returns:
            임베딩 배열 (n_texts, embedding_dim)
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """임베딩 차원 수 반환"""
        pass

    def encode_single(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩 (편의 함수)"""
        return self.encode([text])[0]

    def is_loaded(self) -> bool:
        """모델 로드 상태 확인"""
        return self._is_loaded

    def unload_model(self) -> None:
        """모델 언로드 (메모리 절약)"""
        self.model = None
        self._is_loaded = False
        logger.info("🗑️ Model unloaded")


class SentenceTransformerModel(BaseEmbeddingModel):
    """sentence-transformers 기반 임베딩 모델"""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.model_name = config.model_name

        # sentence-transformers 의존성 체크
        try:
            import sentence_transformers

            self._st_available = True
        except ImportError:
            self._st_available = False
            logger.warning("sentence-transformers not available")

    def load_model(self) -> None:
        """SentenceTransformer 모델 로드"""
        if not self._st_available:
            raise ImportError(
                "sentence-transformers is required for this model.\n"
                "Install with: pip install sentence-transformers"
            )

        if self._is_loaded:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"📥 Loading SentenceTransformer: {self.model_name}")

            # 모델 로드 설정
            load_kwargs = {"device": self.device, **self.config.model_kwargs}

            if self.config.cache_dir:
                load_kwargs["cache_folder"] = self.config.cache_dir

            self.model = SentenceTransformer(self.model_name, **load_kwargs)
            self._is_loaded = True

            logger.info(f"✅ Model loaded successfully")
            logger.info(f"   Dimension: {self.get_embedding_dimension()}")
            logger.info(f"   Max length: {self.model.max_seq_length}")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """텍스트들을 임베딩으로 변환"""
        if not self._is_loaded:
            self.load_model()

        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or self.config.batch_size

        try:
            # 경고 억제 (tqdm과 충돌 방지)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # 코사인 유사도를 위해 정규화
                )

            return embeddings

        except Exception as e:
            logger.error(f"❌ Encoding failed: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """임베딩 차원 수 반환"""
        if not self._is_loaded:
            # 모델 로드 없이 차원 수를 알기 위한 임시 로드
            temp_model = self.model
            self.load_model()
            dim = self.model.get_sentence_embedding_dimension()
            if temp_model is None:  # 원래 로드되지 않았던 경우 다시 언로드
                self.unload_model()
            return dim
        else:
            return self.model.get_sentence_embedding_dimension()


class HuggingFaceModel(BaseEmbeddingModel):
    """HuggingFace transformers 기반 임베딩 모델"""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.model_name = config.model_name

        # transformers 의존성 체크
        try:
            import transformers
            import torch

            self._hf_available = True
        except ImportError:
            self._hf_available = False
            logger.warning("transformers or torch not available")

    def load_model(self) -> None:
        """HuggingFace 모델 로드"""
        if not self._hf_available:
            raise ImportError(
                "transformers and torch are required for this model.\n"
                "Install with: pip install transformers torch"
            )

        if self._is_loaded:
            return

        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            logger.info(f"📥 Loading HuggingFace model: {self.model_name}")

            # 토크나이저와 모델 로드
            load_kwargs = {**self.config.model_kwargs}

            if self.config.cache_dir:
                load_kwargs["cache_dir"] = self.config.cache_dir

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, **load_kwargs
            )
            self.model = AutoModel.from_pretrained(self.model_name, **load_kwargs)

            # 디바이스로 이동
            if self.device != "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()  # 평가 모드
            self._is_loaded = True

            logger.info(f"✅ HuggingFace model loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load HuggingFace model: {e}")
            raise

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """텍스트들을 임베딩으로 변환"""
        if not self._is_loaded:
            self.load_model()

        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or self.config.batch_size

        try:
            import torch
            from tqdm import tqdm

            all_embeddings = []

            # 배치 처리
            for i in tqdm(
                range(0, len(texts), batch_size),
                desc="Encoding",
                disable=not show_progress,
            ):
                batch_texts = texts[i : i + batch_size]

                # 토크나이징
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )

                # 디바이스로 이동
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 추론
                with torch.no_grad():
                    outputs = self.model(**inputs)

                    # [CLS] 토큰 또는 평균 풀링
                    if (
                        hasattr(outputs, "pooler_output")
                        and outputs.pooler_output is not None
                    ):
                        # BERT 스타일 (CLS 토큰)
                        embeddings = outputs.pooler_output
                    else:
                        # 평균 풀링
                        token_embeddings = outputs.last_hidden_state
                        attention_mask = inputs["attention_mask"]

                        # 마스크를 고려한 평균
                        input_mask_expanded = (
                            attention_mask.unsqueeze(-1)
                            .expand(token_embeddings.size())
                            .float()
                        )
                        embeddings = torch.sum(
                            token_embeddings * input_mask_expanded, 1
                        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

                    # 정규화
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    # CPU로 이동 및 numpy 변환
                    batch_embeddings = embeddings.cpu().numpy()
                    all_embeddings.append(batch_embeddings)

            return np.vstack(all_embeddings)

        except Exception as e:
            logger.error(f"❌ HuggingFace encoding failed: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """임베딩 차원 수 반환"""
        if not self._is_loaded:
            self.load_model()

        return self.model.config.hidden_size


# 모델 레지스트리
EMBEDDING_MODELS = {
    # Sentence Transformers 모델들
    "sentence-transformers": {
        # 다국어 지원 모델들
        "all-MiniLM-L6-v2": {
            "dimension": 384,
            "max_length": 256,
            "description": "Fast and good quality, English focused",
        },
        "all-mpnet-base-v2": {
            "dimension": 768,
            "max_length": 384,
            "description": "High quality, English focused",
        },
        "paraphrase-multilingual-MiniLM-L12-v2": {
            "dimension": 384,
            "max_length": 128,
            "description": "Multilingual, good for Korean/English",
        },
        "paraphrase-multilingual-mpnet-base-v2": {
            "dimension": 768,
            "max_length": 128,
            "description": "High quality multilingual",
        },
        # 과학 논문 특화
        "allenai/specter": {
            "dimension": 768,
            "max_length": 512,
            "description": "Scientific papers specialist",
        },
        "allenai/specter2_base": {
            "dimension": 768,
            "max_length": 512,
            "description": "Updated scientific papers model",
        },
    },
    # HuggingFace 모델들
    "huggingface": {
        "sentence-transformers/all-MiniLM-L6-v2": {
            "dimension": 384,
            "max_length": 256,
            "description": "Same as ST version but via HF",
        },
        "klue/bert-base": {
            "dimension": 768,
            "max_length": 512,
            "description": "Korean BERT",
        },
        "microsoft/DialoGPT-medium": {
            "dimension": 1024,
            "max_length": 1024,
            "description": "Dialog focused",
        },
    },
}


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """사용 가능한 모델 목록 반환"""
    return EMBEDDING_MODELS


def get_model_info(
    model_name: str, model_type: str = "auto"
) -> Optional[Dict[str, Any]]:
    """특정 모델 정보 반환"""
    if model_type == "auto":
        # 모든 타입에서 검색
        for mtype, models in EMBEDDING_MODELS.items():
            if model_name in models:
                info = models[model_name].copy()
                info["model_type"] = mtype
                return info
        return None
    else:
        return EMBEDDING_MODELS.get(model_type, {}).get(model_name)


def recommend_model(
    language: str = "mixed", quality: str = "balanced", use_case: str = "academic"
) -> Tuple[str, str]:
    """사용 케이스에 따른 모델 추천

    Args:
        language: "korean", "english", "mixed"
        quality: "fast", "balanced", "high"
        use_case: "academic", "general", "dialog"

    Returns:
        (model_name, model_type) 튜플
    """

    # 학술 논문 특화
    if use_case == "academic":
        if quality == "high":
            return "allenai/specter2_base", "sentence-transformers"
        else:
            return "allenai/specter", "sentence-transformers"

    # 다국어 (한국어/영어 혼용)
    if language in ["mixed", "korean"]:
        if quality == "fast":
            return "paraphrase-multilingual-MiniLM-L12-v2", "sentence-transformers"
        else:
            return "paraphrase-multilingual-mpnet-base-v2", "sentence-transformers"

    # 영어 전용
    if language == "english":
        if quality == "fast":
            return "all-MiniLM-L6-v2", "sentence-transformers"
        else:
            return "all-mpnet-base-v2", "sentence-transformers"

    # 기본값
    return "paraphrase-multilingual-mpnet-base-v2", "sentence-transformers"


def create_embedding_model(
    model_name: str = "auto", model_type: str = "auto", device: str = "auto", **kwargs
) -> BaseEmbeddingModel:
    """임베딩 모델 팩토리 함수

    Args:
        model_name: 모델명 ("auto"면 자동 추천)
        model_type: 모델 타입 ("auto"면 자동 감지)
        device: 디바이스 ("auto"면 자동 선택)
        **kwargs: 추가 설정

    Returns:
        BaseEmbeddingModel 인스턴스
    """

    # 자동 모델 선택
    if model_name == "auto":
        # 기본 추천 (학술 논문용 다국어 모델)
        model_name, model_type = recommend_model(
            language="mixed", quality="balanced", use_case="academic"
        )
        logger.info(f"🎯 Auto-selected model: {model_name} ({model_type})")

    # 모델 타입 자동 감지
    if model_type == "auto":
        model_info = get_model_info(model_name, "auto")
        if model_info:
            model_type = model_info["model_type"]
        else:
            # 기본값: sentence-transformers
            model_type = "sentence-transformers"

    # 모델 정보 가져오기
    model_info = get_model_info(model_name, model_type)
    if not model_info:
        logger.warning(f"⚠️ Model {model_name} not in registry, using defaults")
        model_info = {"dimension": 768, "max_length": 512}

    # 설정 생성
    config = EmbeddingConfig(
        model_name=model_name,
        model_type=model_type,
        dimension=model_info.get("dimension", 768),
        max_length=model_info.get("max_length", 512),
        device=device,
        **kwargs,
    )

    # 모델 인스턴스 생성
    if model_type == "sentence-transformers":
        return SentenceTransformerModel(config)
    elif model_type == "huggingface":
        return HuggingFaceModel(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def list_models():
    """사용 가능한 모델들 출력"""
    print("🤖 Available Embedding Models:")
    print("=" * 50)

    for model_type, models in EMBEDDING_MODELS.items():
        print(f"\n📦 {model_type.upper()}:")
        for name, info in models.items():
            dim = info["dimension"]
            desc = info["description"]
            print(f"  • {name} (dim:{dim}) - {desc}")

    print(f"\n💡 Usage:")
    print(f"  model = create_embedding_model('auto')  # Auto selection")
    print(f"  model = create_embedding_model('all-MiniLM-L6-v2')  # Specific model")


if __name__ == "__main__":
    # 테스트 실행
    print("🧪 Testing Embedding Models...")

    # 사용 가능한 모델 목록
    list_models()

    # 추천 시스템 테스트
    print(f"\n🎯 Model Recommendations:")
    cases = [
        ("mixed", "fast", "academic"),
        ("english", "high", "general"),
        ("korean", "balanced", "academic"),
    ]

    for lang, qual, use in cases:
        rec_model, rec_type = recommend_model(lang, qual, use)
        print(f"  {lang}/{qual}/{use}: {rec_model} ({rec_type})")
