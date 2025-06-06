"""
GraphRAG 임베딩 모듈
Embedding Module for GraphRAG System

노드별 최적화된 임베딩 생성 및 관리
- MultiNodeEmbedder: 통합 노드 임베딩 생성기
- 다양한 임베딩 모델 지원
- 노드 타입별 텍스트 처리 최적화
- 벡터 저장소 연동
"""

__version__ = "1.0.0"
__author__ = "GraphRAG Team"

# 임베딩 관련 의존성 체크
import sys
import warnings
from typing import Dict, Any, Optional


def check_embedding_dependencies() -> Dict[str, bool]:
    """임베딩 관련 의존성 체크"""
    dependencies = {
        "sentence_transformers": False,
        "chromadb": False,
        "faiss": False,
        "torch": False,
        "transformers": False,
        "numpy": False,
        "scikit-learn": False,
        "tqdm": False,  # 👈 추가
        "pandas": False,  # 👈 추가
    }

    for package in dependencies:
        try:
            if package == "sentence_transformers":
                import sentence_transformers
            elif package == "chromadb":
                import chromadb
            elif package == "faiss":
                import faiss
            elif package == "torch":
                import torch
            elif package == "transformers":
                import transformers
            elif package == "numpy":
                import numpy
            elif package == "scikit-learn":
                import sklearn

            dependencies[package] = True
        except ImportError:
            dependencies[package] = False

    return dependencies


# 의존성 체크 실행
_embedding_deps = check_embedding_dependencies()

# 필수 의존성 확인
_required_deps = ["numpy", "scikit-learn"]
_missing_required = [dep for dep in _required_deps if not _embedding_deps[dep]]

if _missing_required:
    raise ImportError(
        f"Missing required embedding dependencies: {_missing_required}\n"
        f"Install with: pip install {' '.join(_missing_required)}"
    )

# 선택적 의존성 경고
if not _embedding_deps["sentence_transformers"]:
    warnings.warn(
        "sentence-transformers not found. Some embedding models may not work.\n"
        "Install with: pip install sentence-transformers"
    )

if not _embedding_deps["torch"]:
    warnings.warn(
        "PyTorch not found. GPU acceleration will not be available.\n"
        "Install with: pip install torch"
    )

# 메인 클래스들 import
try:
    from .embedding_models import (
        BaseEmbeddingModel,
        SentenceTransformerModel,
        HuggingFaceModel,
        get_available_models,
        create_embedding_model,
    )

    from .node_text_processors import (
        BaseNodeTextProcessor,
        PaperTextProcessor,
        AuthorTextProcessor,
        KeywordTextProcessor,
        JournalTextProcessor,
        create_text_processor,
    )

    from .multi_node_embedder import MultiNodeEmbedder
    from .vector_store_manager import (
        VectorStoreManager,
        create_vector_store,
        SearchResult,
    )

    _components_available = True

except ImportError as e:
    warnings.warn(f"Some embedding components not available: {e}")
    _components_available = False

    # Placeholder 클래스들
    class MultiNodeEmbedder:
        def __init__(self, *args, **kwargs):
            raise ImportError("MultiNodeEmbedder requires additional dependencies")

    class VectorStoreManager:
        def __init__(self, *args, **kwargs):
            raise ImportError("VectorStoreManager requires additional dependencies")

    class EmbeddingResult:
        def __init__(self, *args, **kwargs):
            raise ImportError("EmbeddingResult requires additional dependencies")

    BaseEmbeddingModel = None
    BaseNodeTextProcessor = None
    SearchResult = None  # 👈 추가


# 편의 함수들
def print_embedding_dependencies():
    """임베딩 의존성 정보 출력"""
    print("🔧 GraphRAG Embedding Dependencies:")
    for dep, available in _embedding_deps.items():
        status = "✅" if available else "❌"
        print(f"  {status} {dep}")

    if not _components_available:
        print("\n⚠️  Some components are not available due to missing dependencies")


def get_embedding_dependencies() -> Dict[str, bool]:
    """임베딩 의존성 정보 반환"""
    return _embedding_deps.copy()


def is_ready() -> bool:
    """임베딩 시스템이 준비되었는지 확인"""
    return _components_available and _embedding_deps["sentence_transformers"]


# 빠른 시작 함수
def create_embedder(
    unified_graph_path: str, embedding_model: str = "auto", **kwargs
) -> Optional["MultiNodeEmbedder"]:
    """빠른 임베더 생성

    Args:
        unified_graph_path: 통합 그래프 파일 경로
        embedding_model: 사용할 임베딩 모델 ("auto", "sentence-transformers", etc.)
        **kwargs: 추가 설정

    Returns:
        MultiNodeEmbedder 인스턴스 또는 None
    """
    if not is_ready():
        print("❌ Embedding system is not ready. Check dependencies.")
        return None

    try:
        embedder = MultiNodeEmbedder(
            unified_graph_path=unified_graph_path,
            embedding_model=embedding_model,
            **kwargs,
        )
        return embedder
    except Exception as e:
        print(f"❌ Failed to create embedder: {e}")
        return None


# 패키지 레벨 exports
__all__ = [
    # 메인 클래스
    "MultiNodeEmbedder",
    # 기본 클래스들
    "BaseEmbeddingModel",
    "BaseNodeTextProcessor",
    "VectorStoreManager",
    "EmbeddingResult",
    "SearchResult",
    # 구체적 구현들 (사용 가능한 경우)
    "SentenceTransformerModel",
    "HuggingFaceModel",
    "PaperTextProcessor",
    "AuthorTextProcessor",
    "KeywordTextProcessor",
    "JournalTextProcessor",
    # 팩토리 함수들
    "create_embedding_model",
    "create_text_processor",
    "create_embedder",
    "create_vector_store",
    # 유틸리티
    "get_available_models",
    "print_embedding_dependencies",
    "get_embedding_dependencies",
    "is_ready",
]

print(f"🚀 GraphRAG Embeddings v{__version__} loaded")
if _components_available:
    print(f"✅ All components available")
else:
    print(f"⚠️  Some components unavailable - check dependencies")
