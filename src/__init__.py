"""
GraphRAG 기반 문헌분석 프레임워크
Technology Innovation Research Project
"""

__version__ = "1.0.0"
__author__ = "Technology Innovation Team"

# 프로젝트 루트 경로 설정
import os
import sys
from pathlib import Path

# 현재 파일의 상위 디렉토리를 프로젝트 루트로 설정
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
BIBS_DIR = DATA_DIR / "bibs"
PDFS_DIR = DATA_DIR / "pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_EXTRACTIONS_DIR = PROCESSED_DIR / "raw_extractions"
GRAPHS_DIR = PROCESSED_DIR / "graphs"

# 경로를 sys.path에 추가하여 모듈 임포트 가능하게 함
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 필요한 디렉토리 생성
for directory in [DATA_DIR, PROCESSED_DIR, RAW_EXTRACTIONS_DIR, GRAPHS_DIR]:
    directory.mkdir(exist_ok=True)

print(f"📁 Project initialized - Root: {PROJECT_ROOT}")
print(f"📁 Data directory: {DATA_DIR}")
print(f"📁 Bibtex files: {BIBS_DIR}")
print(f"📁 PDF files: {PDFS_DIR}")
