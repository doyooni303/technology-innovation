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
        "tqdm": False,
        "pandas": False,
    }

    for package in dependencies:
        try:
            if package == "sentence_transformers":
                import sentence_transformers
            elif package == "chromadb":
                # 🆕 ChromaDB는 SQLite 버전 체크로 인해 RuntimeError 발생 가능
                # import 대신 패키지 존재 여부만 확인
                import importlib.util

                spec = importlib.util.find_spec("chromadb")
                dependencies[package] = spec is not None
                continue  # import 시도하지 않고 건너뛰기
            elif package == "faiss":
                try:
                    import faiss
                except ImportError:
                    # faiss-cpu 또는 faiss-gpu 시도
                    try:
                        import faiss_gpu as faiss
                    except ImportError:
                        import faiss_cpu as faiss
            elif package == "torch":
                import torch
            elif package == "transformers":
                import transformers
            elif package == "numpy":
                import numpy
            elif package == "scikit-learn":
                import sklearn
            elif package == "tqdm":
                import tqdm
            elif package == "pandas":
                import pandas

            dependencies[package] = True
        except (ImportError, RuntimeError) as e:
            # RuntimeError도 잡아서 ChromaDB SQLite 에러 처리
            dependencies[package] = False

    return dependencies


# 의존성 체크 실행
_embedding_deps = check_embedding_dependencies()

# 필수 의존성 확인
_required_deps = ["numpy", "scikit-learn", "tqdm", "pandas"]  # 🆕 tqdm, pandas 추가
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

# FAISS 우선, ChromaDB는 SQLite 호환성 문제로 비활성화
if not _embedding_deps["faiss"]:
    warnings.warn(
        "FAISS not found. Vector store may not work optimally.\n"
        "Install with: pip install faiss-cpu  # or faiss-gpu for GPU support"
    )

# ChromaDB는 SQLite 호환성 문제로 경고만 표시
if not _embedding_deps["chromadb"]:
    # SQLite 호환성 문제가 있을 수 있으므로 조용히 처리
    pass

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

    from .multi_node_embedder import (
        MultiNodeEmbedder,
        EmbeddingResult,
    )  # 🆕 EmbeddingResult 추가

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

    class SearchResult:
        def __init__(self, *args, **kwargs):
            raise ImportError("SearchResult requires additional dependencies")

    BaseEmbeddingModel = None
    BaseNodeTextProcessor = None


# 편의 함수들
def print_embedding_dependencies():
    """임베딩 의존성 정보 출력"""
    print("🔧 GraphRAG Embedding Dependencies:")
    for dep, available in _embedding_deps.items():
        status = "✅" if available else "❌"
        print(f"  {status} {dep}")

    if not _components_available:
        print("\n⚠️  Some components are not available due to missing dependencies")
        print("💡 Try: pip install sentence-transformers faiss-cpu tqdm pandas")


def get_embedding_dependencies() -> Dict[str, bool]:
    """임베딩 의존성 정보 반환"""
    return _embedding_deps.copy()


def is_ready() -> bool:
    """임베딩 시스템이 준비되었는지 확인"""
    # 핵심 의존성들 체크 (ChromaDB 제외, FAISS 우선)
    required_for_basic = ["sentence_transformers", "numpy", "tqdm", "pandas"]
    vector_store_ready = _embedding_deps["faiss"]  # FAISS만 사용

    basic_ready = all(_embedding_deps[dep] for dep in required_for_basic)

    return _components_available and basic_ready and vector_store_ready


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
        print_embedding_dependencies()
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

# 🆕 개선된 로딩 메시지
print(f"🚀 GraphRAG Embeddings v{__version__} loaded")
if _components_available and is_ready():
    print(f"✅ All components ready")
    print(f"   Vector store: FAISS")
elif _components_available:
    print(f"⚠️  Components loaded but missing dependencies")
    missing = [
        dep for dep, avail in _embedding_deps.items() if not avail and dep != "chromadb"
    ]  # ChromaDB 제외
    if missing:
        print(f"   Missing: {', '.join(missing)}")
    if not _embedding_deps["faiss"]:
        print(f"   💡 Install FAISS: pip install faiss-cpu")
else:
    print(f"❌ Components unavailable - check dependencies")
    print(f"💡 Run: pip install sentence-transformers faiss-cpu tqdm pandas")
