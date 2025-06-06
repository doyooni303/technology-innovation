"""
GraphRAG: Graph-based Retrieval Augmented Generation
논문 데이터를 위한 지식 그래프 기반 질의응답 시스템

이 패키지는 다음 컴포넌트들을 제공합니다:
- UnifiedKnowledgeGraphBuilder: 여러 그래프를 통합된 지식 그래프로 구축
- QueryAnalyzer: 사용자 쿼리 분석 및 분류
- SubgraphExtractor: 관련 서브그래프 추출
- ContextSerializer: 그래프 정보를 LLM용 텍스트로 변환
- GraphRAGPipeline: 전체 파이프라인 통합

사용 예시:
    from graphrag import GraphRAGPipeline

    # 기본 사용법
    rag = GraphRAGPipeline()
    answer = rag.query("배터리 SoC 예측에 사용된 머신러닝 기법은?")

    # 고급 사용법
    from graphrag import UnifiedKnowledgeGraphBuilder, QueryAnalyzer

    builder = UnifiedKnowledgeGraphBuilder(graphs_dir="./graphs")
    unified_graph = builder.build_unified_graph()

    analyzer = QueryAnalyzer()
    query_info = analyzer.analyze("연구 동향은?")
"""

__version__ = "1.0.0"
__author__ = "Technology Innovation Research Team"
__email__ = "research@company.com"
__description__ = "Graph-based Retrieval Augmented Generation for Academic Literature"

import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List

# Python 버전 체크
if sys.version_info < (3, 8):
    raise RuntimeError("GraphRAG requires Python 3.8 or higher")

# 경고 필터링 (개발 중 불필요한 경고 억제)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")


# 로깅 설정
def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """GraphRAG 로깅 설정

    Args:
        level: 로깅 레벨 (DEBUG, INFO, WARNING, ERROR)
        format_string: 로그 포맷 문자열
        log_file: 로그 파일 경로 (None이면 콘솔만)

    Returns:
        Logger: 설정된 로거
    """
    if format_string is None:
        format_string = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    # 루트 로거 설정
    logging.basicConfig(
        level=getattr(logging, level.upper()), format=format_string, handlers=[]
    )

    logger = logging.getLogger("graphrag")
    logger.handlers.clear()  # 기존 핸들러 제거

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)

    # 파일 핸들러 (선택적)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    logger.setLevel(getattr(logging, level.upper()))
    return logger


# 기본 로깅 설정
_logger = setup_logging()


# 의존성 체크
def check_dependencies() -> Dict[str, bool]:
    """필수 의존성 체크

    Returns:
        Dict[str, bool]: 각 의존성의 설치 여부
    """
    dependencies = {
        "networkx": False,
        "pandas": False,
        "numpy": False,
        "transformers": False,
        "torch": False,
        "sklearn": False,
        "tqdm": False,
        "matplotlib": False,
        "seaborn": False,
    }

    for package in dependencies:
        try:
            __import__(package)
            dependencies[package] = True
        except ImportError:
            _logger.warning(f"Optional dependency '{package}' not found")
            dependencies[package] = False

    # 필수 의존성 체크
    required = ["networkx", "pandas", "numpy"]
    missing_required = [pkg for pkg in required if not dependencies[pkg]]

    if missing_required:
        raise ImportError(
            f"Missing required dependencies: {missing_required}\n"
            f"Install with: pip install {' '.join(missing_required)}"
        )

    return dependencies


# 의존성 체크 실행
_deps = check_dependencies()


# 설정 관리
class GraphRAGConfig:
    """GraphRAG 전역 설정 관리"""

    def __init__(self):
        self.reset_to_defaults()

    def reset_to_defaults(self):
        """기본 설정으로 리셋"""
        self.settings = {
            # 그래프 구축 설정
            "graph_builder": {
                "enable_cross_connections": True,
                "temporal_connections": True,
                "min_keyword_frequency": 2,
                "max_temporal_connections": 50,
            },
            # 쿼리 분석 설정
            "query_analyzer": {
                "complexity_threshold": 0.5,
                "auto_mode_enabled": True,
                "supported_languages": ["en", "ko"],
            },
            # 서브그래프 추출 설정
            "subgraph_extractor": {
                "max_nodes": 1000,
                "max_edges": 5000,
                "hop_limit": 3,
                "similarity_threshold": 0.5,
            },
            # 컨텍스트 직렬화 설정
            "context_serializer": {
                "max_context_length": 8000,
                "include_metadata": True,
                "format_style": "structured",
            },
            # 성능 설정
            "performance": {
                "cache_enabled": True,
                "cache_size_mb": 500,
                "parallel_processing": True,
                "max_workers": 4,
            },
            # 출력 설정
            "output": {
                "save_intermediate_results": False,
                "output_format": ["json"],
                "verbose": False,
            },
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        """중첩된 설정값 가져오기

        Args:
            key_path: 점으로 구분된 키 경로 (예: "graph_builder.enable_cross_connections")
            default: 기본값

        Returns:
            설정값 또는 기본값
        """
        keys = key_path.split(".")
        value = self.settings

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any):
        """중첩된 설정값 설정

        Args:
            key_path: 점으로 구분된 키 경로
            value: 설정할 값
        """
        keys = key_path.split(".")
        setting = self.settings

        for key in keys[:-1]:
            if key not in setting:
                setting[key] = {}
            setting = setting[key]

        setting[keys[-1]] = value

    def update(self, new_settings: Dict[str, Any]):
        """설정 업데이트 (딕셔너리 병합)

        Args:
            new_settings: 새로운 설정 딕셔너리
        """

        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if (
                    key in base_dict
                    and isinstance(base_dict[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_update(self.settings, new_settings)


# 전역 설정 인스턴스
config = GraphRAGConfig()


# 편의 함수들
def set_log_level(level: str):
    """로깅 레벨 설정

    Args:
        level: DEBUG, INFO, WARNING, ERROR 중 하나
    """
    _logger.setLevel(getattr(logging, level.upper()))


def get_version() -> str:
    """GraphRAG 버전 반환"""
    return __version__


def get_dependencies() -> Dict[str, bool]:
    """설치된 의존성 정보 반환"""
    return _deps.copy()


def print_system_info():
    """시스템 정보 출력"""
    print(f"GraphRAG v{__version__}")
    print(f"Python {sys.version}")
    print(f"Dependencies:")
    for pkg, installed in _deps.items():
        status = "✅" if installed else "❌"
        print(f"  {status} {pkg}")


# 주요 클래스들 Import (현재 구현된 것들)
try:
    from .unified_graph_builder import UnifiedKnowledgeGraphBuilder

    _logger.info("✅ UnifiedKnowledgeGraphBuilder loaded")
except ImportError as e:
    _logger.warning(f"⚠️ UnifiedKnowledgeGraphBuilder not available: {e}")
    UnifiedKnowledgeGraphBuilder = None

# 향후 구현될 클래스들 (placeholder)
QueryAnalyzer = None
SubgraphExtractor = None
ContextSerializer = None
GraphRAGPipeline = None


# 동적 import 함수들 (lazy loading)
def _lazy_import_query_analyzer():
    """QueryAnalyzer 지연 로딩"""
    global QueryAnalyzer
    if QueryAnalyzer is None:
        try:
            from .query_analyzer import QueryAnalyzer

            _logger.info("✅ QueryAnalyzer loaded")
        except ImportError as e:
            _logger.warning(f"⚠️ QueryAnalyzer not available: {e}")
            raise ImportError(f"QueryAnalyzer not implemented yet: {e}")
    return QueryAnalyzer


def _lazy_import_subgraph_extractor():
    """SubgraphExtractor 지연 로딩"""
    global SubgraphExtractor
    if SubgraphExtractor is None:
        try:
            from .subgraph_extractor import SubgraphExtractor

            _logger.info("✅ SubgraphExtractor loaded")
        except ImportError as e:
            _logger.warning(f"⚠️ SubgraphExtractor not available: {e}")
            raise ImportError(f"SubgraphExtractor not implemented yet: {e}")
    return SubgraphExtractor


def _lazy_import_context_serializer():
    """ContextSerializer 지연 로딩"""
    global ContextSerializer
    if ContextSerializer is None:
        try:
            from .context_serializer import ContextSerializer

            _logger.info("✅ ContextSerializer loaded")
        except ImportError as e:
            _logger.warning(f"⚠️ ContextSerializer not available: {e}")
            raise ImportError(f"ContextSerializer not implemented yet: {e}")
    return ContextSerializer


def _lazy_import_pipeline():
    """GraphRAGPipeline 지연 로딩"""
    global GraphRAGPipeline
    if GraphRAGPipeline is None:
        try:
            from .graphrag_pipeline import GraphRAGPipeline

            _logger.info("✅ GraphRAGPipeline loaded")
        except ImportError as e:
            _logger.warning(f"⚠️ GraphRAGPipeline not available: {e}")
            raise ImportError(f"GraphRAGPipeline not implemented yet: {e}")
    return GraphRAGPipeline


# Quick Start 함수
def build_unified_graph(
    graphs_dir: str, output_dir: Optional[str] = None, **kwargs
) -> "UnifiedKnowledgeGraphBuilder":
    """빠른 통합 그래프 구축

    Args:
        graphs_dir: 개별 그래프 파일들이 있는 디렉토리
        output_dir: 출력 디렉토리 (None이면 graphs_dir/unified)
        **kwargs: 추가 설정

    Returns:
        UnifiedKnowledgeGraphBuilder: 구축된 그래프 빌더
    """
    if UnifiedKnowledgeGraphBuilder is None:
        raise ImportError("UnifiedKnowledgeGraphBuilder is not available")

    builder = UnifiedKnowledgeGraphBuilder(graphs_dir)

    # 설정 업데이트
    if kwargs:
        config.update({"graph_builder": kwargs})

    unified_graph = builder.build_unified_graph(
        save_output=True, output_dir=Path(output_dir) if output_dir else None
    )

    return builder


def quick_query(query: str, graphs_dir: str, mode: str = "auto") -> str:
    """빠른 쿼리 실행 (개발 중)

    Args:
        query: 사용자 질문
        graphs_dir: 그래프 디렉토리
        mode: 쿼리 모드 ("auto", "selective", "comprehensive")

    Returns:
        str: 답변
    """
    try:
        pipeline_class = _lazy_import_pipeline()
        pipeline = pipeline_class(graphs_dir=graphs_dir)
        return pipeline.query(query, mode=mode)
    except ImportError:
        raise NotImplementedError(
            "GraphRAGPipeline is not implemented yet. "
            "Please use individual components or wait for the next update."
        )


# 패키지 레벨에서 사용 가능한 모든 것들
__all__ = [
    # 버전 정보
    "__version__",
    "__author__",
    "__description__",
    # 설정
    "config",
    "setup_logging",
    "set_log_level",
    # 유틸리티
    "get_version",
    "get_dependencies",
    "print_system_info",
    "check_dependencies",
    # 주요 클래스들 (구현된 것들)
    "UnifiedKnowledgeGraphBuilder",
    # 지연 로딩 함수들
    "_lazy_import_query_analyzer",
    "_lazy_import_subgraph_extractor",
    "_lazy_import_context_serializer",
    "_lazy_import_pipeline",
    # 편의 함수들
    "build_unified_graph",
    "quick_query",
]

# 패키지 초기화 완료 로깅
_logger.info(f"🚀 GraphRAG v{__version__} initialized successfully")
_logger.info(
    f"📊 Available components: {[name for name in __all__ if not name.startswith('_')]}"
)

# 개발 중 알림
if any(
    cls is None
    for cls in [QueryAnalyzer, SubgraphExtractor, ContextSerializer, GraphRAGPipeline]
):
    _logger.info("🔧 Some components are still under development")
    _logger.info("✅ Available: UnifiedKnowledgeGraphBuilder")
    _logger.info(
        "🚧 Coming soon: QueryAnalyzer, SubgraphExtractor, ContextSerializer, GraphRAGPipeline"
    )
