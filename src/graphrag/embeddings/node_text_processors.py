"""
노드 타입별 텍스트 처리 모듈
Node Text Processors for GraphRAG Embeddings

각 노드 타입의 특성에 맞는 최적화된 텍스트 생성
- PaperTextProcessor: 논문 제목, 초록, 키워드 결합
- AuthorTextProcessor: 저자명, 연구 분야, 논문 제목들 결합
- KeywordTextProcessor: 키워드와 관련 컨텍스트 생성
- JournalTextProcessor: 저널명과 연구 영역 정보 결합
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Union
from collections import Counter

# 로깅 설정
logger = logging.getLogger(__name__)


class BaseNodeTextProcessor(ABC):
    """노드 텍스트 처리 기본 클래스"""

    def __init__(self, max_length: int = 512, language: str = "mixed"):
        """
        Args:
            max_length: 최대 텍스트 길이 (토큰 기준 대략적)
            language: 주요 언어 ("ko", "en", "mixed")
        """
        self.max_length = max_length
        self.language = language

        # 언어별 불용어 설정
        self.stopwords = self._get_stopwords(language)

    def _get_stopwords(self, language: str) -> Set[str]:
        """언어별 불용어 세트"""
        korean_stopwords = {
            "이",
            "가",
            "을",
            "를",
            "의",
            "에",
            "에서",
            "와",
            "과",
            "도",
            "는",
            "은",
            "하다",
            "있다",
            "되다",
            "이다",
            "그",
            "것",
            "수",
            "등",
            "및",
            "또한",
            "그리고",
        }

        english_stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "this",
            "that",
            "these",
            "those",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
        }

        if language == "ko":
            return korean_stopwords
        elif language == "en":
            return english_stopwords
        else:  # mixed
            return korean_stopwords | english_stopwords

    def clean_text(self, text: str) -> str:
        """텍스트 기본 정제"""
        if not text:
            return ""

        # 기본 정제
        text = str(text).strip()

        # 연속 공백 제거
        text = re.sub(r"\s+", " ", text)

        # 특수문자 정리 (의미있는 것들은 보존)
        text = re.sub(r"[^\w\s\-\.\,\(\)]", " ", text)

        # 다시 연속 공백 제거
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def truncate_text(self, text: str, max_length: Optional[int] = None) -> str:
        """텍스트 길이 제한 (단어 기준)"""
        if not text:
            return ""

        max_length = max_length or self.max_length
        words = text.split()

        if len(words) <= max_length:
            return text

        # 중요한 단어들을 우선 보존
        truncated_words = words[:max_length]
        return " ".join(truncated_words)

    def remove_stopwords(self, text: str) -> str:
        """불용어 제거 (선택적)"""
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords]
        return " ".join(filtered_words)

    @abstractmethod
    def process_node(self, node_data: Dict[str, Any]) -> str:
        """노드 데이터를 임베딩용 텍스트로 변환"""
        pass

    def batch_process(self, nodes_data: List[Dict[str, Any]]) -> List[str]:
        """여러 노드를 배치 처리"""
        return [self.process_node(node_data) for node_data in nodes_data]


class PaperTextProcessor(BaseNodeTextProcessor):
    """논문 노드 텍스트 처리기"""

    def __init__(
        self,
        max_length: int = 512,
        language: str = "mixed",
        include_authors: bool = True,
        include_journal: bool = True,
    ):
        """
        Args:
            max_length: 최대 텍스트 길이
            language: 주요 언어
            include_authors: 저자 정보 포함 여부
            include_journal: 저널 정보 포함 여부
        """
        super().__init__(max_length, language)
        self.include_authors = include_authors
        self.include_journal = include_journal

    def process_node(self, node_data: Dict[str, Any]) -> str:
        """논문 노드 처리

        Format: "Title: [title] Abstract: [abstract] Keywords: [keywords] Authors: [authors] Journal: [journal]"
        """
        parts = []

        # 1. 제목 (가장 중요)
        title = self.clean_text(node_data.get("title", ""))
        if title:
            parts.append(f"Title: {title}")

        # 2. 초록 (매우 중요)
        abstract = self.clean_text(node_data.get("abstract", ""))
        if abstract:
            # 초록이 너무 길면 앞부분만 사용
            abstract = self.truncate_text(abstract, max_length=150)
            parts.append(f"Abstract: {abstract}")

        # 3. 키워드 (중요)
        keywords = node_data.get("keywords", [])
        if keywords:
            if isinstance(keywords, str):
                keywords = [kw.strip() for kw in keywords.split(";") if kw.strip()]
            elif isinstance(keywords, list):
                keywords = [str(kw).strip() for kw in keywords if str(kw).strip()]

            # 중복 제거 및 정제
            clean_keywords = []
            seen = set()
            for kw in keywords[:10]:  # 최대 10개
                kw_clean = self.clean_text(kw)
                if kw_clean and kw_clean.lower() not in seen:
                    clean_keywords.append(kw_clean)
                    seen.add(kw_clean.lower())

            if clean_keywords:
                parts.append(f"Keywords: {', '.join(clean_keywords)}")

        # 4. 저자 (선택적)
        if self.include_authors:
            authors = node_data.get("authors", [])
            if authors:
                if isinstance(authors, str):
                    authors = [authors]
                elif isinstance(authors, list):
                    authors = [
                        str(author).strip() for author in authors if str(author).strip()
                    ]

                # 처음 5명만 포함
                author_names = []
                for author in authors[:5]:
                    author_clean = self.clean_text(author)
                    if author_clean:
                        author_names.append(author_clean)

                if author_names:
                    parts.append(f"Authors: {', '.join(author_names)}")

        # 5. 저널 (선택적)
        if self.include_journal:
            journal = self.clean_text(node_data.get("journal", ""))
            if journal:
                parts.append(f"Journal: {journal}")

        # 6. 연도 (추가 컨텍스트)
        year = node_data.get("year", "")
        if year:
            parts.append(f"Year: {year}")

        # 결합 및 길이 제한
        combined_text = " ".join(parts)
        return self.truncate_text(combined_text)


class AuthorTextProcessor(BaseNodeTextProcessor):
    """저자 노드 텍스트 처리기"""

    def __init__(
        self,
        max_length: int = 512,
        language: str = "mixed",
        include_papers: bool = True,
        max_papers: int = 10,
    ):
        """
        Args:
            max_length: 최대 텍스트 길이
            language: 주요 언어
            include_papers: 논문 정보 포함 여부
            max_papers: 포함할 최대 논문 수
        """
        super().__init__(max_length, language)
        self.include_papers = include_papers
        self.max_papers = max_papers

    def process_node(self, node_data: Dict[str, Any]) -> str:
        """저자 노드 처리

        Format: "Author: [name] Research Areas: [inferred areas] Papers: [paper titles] Journals: [journals]"
        """
        parts = []

        # 1. 저자명 (필수)
        name = self.clean_text(node_data.get("name", ""))
        if not name:
            name = self.clean_text(node_data.get("id", ""))  # fallback to ID

        if name:
            parts.append(f"Author: {name}")

        # 2. 연구 생산성 정보
        paper_count = node_data.get("paper_count", 0)
        if paper_count:
            parts.append(f"Publications: {paper_count} papers")

        # 3. 활동 기간
        first_year = node_data.get("first_year", "")
        last_year = node_data.get("last_year", "")
        if first_year and last_year:
            if first_year == last_year:
                parts.append(f"Active: {first_year}")
            else:
                parts.append(f"Active: {first_year}-{last_year}")

        # 4. 주요 키워드/연구 분야 (top_keywords에서 추출)
        top_keywords = node_data.get("top_keywords", [])
        if top_keywords:
            if isinstance(top_keywords, list) and len(top_keywords) > 0:
                # 튜플 형태인 경우 키워드만 추출
                keywords = []
                for item in top_keywords[:5]:  # 상위 5개
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        keyword = str(item[0]).strip()
                    else:
                        keyword = str(item).strip()

                    keyword_clean = self.clean_text(keyword)
                    if keyword_clean:
                        keywords.append(keyword_clean)

                if keywords:
                    parts.append(f"Research Areas: {', '.join(keywords)}")

        # 5. 주요 저널
        most_frequent_journal = self.clean_text(
            node_data.get("most_frequent_journal", "")
        )
        if most_frequent_journal:
            parts.append(f"Main Journal: {most_frequent_journal}")

        # 6. 생산성 타입
        productivity_type = node_data.get("productivity_type", "")
        if productivity_type:
            parts.append(f"Type: {productivity_type}")

        # 7. 논문 제목들 (선택적, 연구 분야 추론용)
        if self.include_papers:
            papers = node_data.get("papers", [])
            if papers and isinstance(papers, list):
                paper_titles = []
                for paper in papers[: self.max_papers]:
                    if isinstance(paper, dict):
                        title = self.clean_text(paper.get("title", ""))
                    else:
                        title = self.clean_text(str(paper))

                    if title:
                        # 너무 긴 제목은 줄이기
                        title = self.truncate_text(title, max_length=20)
                        paper_titles.append(title)

                if paper_titles:
                    parts.append(f"Paper Titles: {' | '.join(paper_titles)}")

        # 결합 및 길이 제한
        combined_text = " ".join(parts)
        return self.truncate_text(combined_text)


class KeywordTextProcessor(BaseNodeTextProcessor):
    """키워드 노드 텍스트 처리기"""

    def __init__(
        self,
        max_length: int = 512,
        language: str = "mixed",
        include_context: bool = True,
    ):
        """
        Args:
            max_length: 최대 텍스트 길이
            language: 주요 언어
            include_context: 관련 컨텍스트 포함 여부
        """
        super().__init__(max_length, language)
        self.include_context = include_context

    def _generate_keyword_context(self, keyword: str) -> str:
        """키워드에 대한 추가 컨텍스트 생성"""
        keyword_lower = keyword.lower()

        # 도메인별 컨텍스트 생성
        contexts = {
            # AI/ML 키워드들
            "machine learning": "artificial intelligence algorithm data analysis prediction model training",
            "deep learning": "neural network artificial intelligence machine learning algorithm",
            "neural network": "deep learning artificial intelligence machine learning algorithm",
            "artificial intelligence": "machine learning deep learning neural network algorithm",
            "reinforcement learning": "machine learning artificial intelligence agent reward algorithm",
            # 배터리/전기차 키워드들
            "battery": "energy storage lithium ion electric vehicle power management",
            "lithium": "battery energy storage electric vehicle ion cell",
            "soc": "state of charge battery energy management system estimation",
            "electric vehicle": "battery charging power management energy storage automotive",
            "charging": "battery electric vehicle power energy management station",
            # 일반 기술 키워드들
            "optimization": "algorithm performance improvement efficiency solution",
            "control": "system management regulation automation process",
            "system": "architecture design implementation framework platform",
            "algorithm": "computation method procedure optimization solution",
            "model": "simulation representation framework algorithm analysis",
        }

        # 키워드 매칭 (부분 매칭 포함)
        for key, context in contexts.items():
            if key in keyword_lower or keyword_lower in key:
                return context

        return ""

    def process_node(self, node_data: Dict[str, Any]) -> str:
        """키워드 노드 처리

        Format: "Keyword: [keyword] Frequency: [freq] Context: [related terms] Papers: [paper count]"
        """
        parts = []

        # 1. 키워드 자체 (필수)
        keyword = self.clean_text(node_data.get("id", ""))
        if not keyword:
            keyword = self.clean_text(node_data.get("name", ""))

        if keyword:
            parts.append(f"Keyword: {keyword}")

        # 2. 빈도 정보
        frequency = node_data.get("frequency", 0)
        if frequency:
            parts.append(f"Frequency: {frequency} papers")

        # 3. 관련 논문 수 (papers 필드에서)
        papers = node_data.get("papers", [])
        if papers and isinstance(papers, list):
            parts.append(f"Used in: {len(papers)} publications")

        # 4. 추가 컨텍스트 (선택적)
        if self.include_context and keyword:
            context = self._generate_keyword_context(keyword)
            if context:
                parts.append(f"Related: {context}")

        # 5. 동시 출현 키워드 정보 (있다면)
        # UnifiedGraph에서 연결된 다른 키워드들을 찾을 수 있음
        # 이는 SubgraphExtractor에서 활용될 예정

        # 결합 및 길이 제한
        combined_text = " ".join(parts)
        return self.truncate_text(combined_text)


class JournalTextProcessor(BaseNodeTextProcessor):
    """저널 노드 텍스트 처리기"""

    def __init__(
        self, max_length: int = 512, language: str = "mixed", include_stats: bool = True
    ):
        """
        Args:
            max_length: 최대 텍스트 길이
            language: 주요 언어
            include_stats: 통계 정보 포함 여부
        """
        super().__init__(max_length, language)
        self.include_stats = include_stats

    def _classify_journal_domain(self, journal_name: str, keywords: List[str]) -> str:
        """저널 도메인 분류"""
        name_lower = journal_name.lower()

        # 저널명 기반 분류
        if any(term in name_lower for term in ["ieee", "acm", "computer", "software"]):
            return "Computer Science"
        elif any(term in name_lower for term in ["nature", "science", "cell"]):
            return "High-Impact General Science"
        elif any(
            term in name_lower for term in ["energy", "power", "electric", "battery"]
        ):
            return "Energy and Power Systems"
        elif any(
            term in name_lower
            for term in ["artificial", "intelligence", "neural", "machine"]
        ):
            return "Artificial Intelligence"
        elif any(
            term in name_lower for term in ["engineering", "industrial", "automation"]
        ):
            return "Engineering"
        elif any(
            term in name_lower for term in ["applied", "international", "journal"]
        ):
            return "Applied Sciences"

        # 키워드 기반 분류
        if keywords:
            keyword_text = " ".join(str(kw).lower() for kw in keywords)
            if any(
                term in keyword_text for term in ["ai", "machine learning", "neural"]
            ):
                return "Artificial Intelligence"
            elif any(term in keyword_text for term in ["battery", "energy", "power"]):
                return "Energy Systems"
            elif any(
                term in keyword_text for term in ["computer", "algorithm", "software"]
            ):
                return "Computer Science"

        return "General"

    def process_node(self, node_data: Dict[str, Any]) -> str:
        """저널 노드 처리

        Format: "Journal: [name] Domain: [domain] Papers: [count] Period: [years] Focus: [keywords]"
        """
        parts = []

        # 1. 저널명 (필수)
        journal_name = self.clean_text(node_data.get("name", ""))
        if not journal_name:
            journal_name = self.clean_text(node_data.get("id", ""))

        if journal_name:
            parts.append(f"Journal: {journal_name}")

        # 2. 논문 수 통계
        if self.include_stats:
            paper_count = node_data.get("paper_count", 0)
            if paper_count:
                parts.append(f"Publications: {paper_count} papers")

        # 3. 활동 기간
        first_year = node_data.get("first_year", "")
        last_year = node_data.get("last_year", "")
        if first_year and last_year:
            if first_year == last_year:
                parts.append(f"Period: {first_year}")
            else:
                parts.append(f"Period: {first_year}-{last_year}")

                # 활동 년수 계산
                try:
                    years_active = int(last_year) - int(first_year) + 1
                    parts.append(f"Active: {years_active} years")
                except:
                    pass

        # 4. 저널 타입/도메인
        journal_type = node_data.get("journal_type", "")
        if journal_type:
            parts.append(f"Type: {journal_type}")

        # 5. 주요 키워드/연구 분야
        top_keywords = node_data.get("top_keywords", [])
        if top_keywords:
            if isinstance(top_keywords, list) and len(top_keywords) > 0:
                keywords = []
                for item in top_keywords[:8]:  # 상위 8개
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        keyword = str(item[0]).strip()
                    else:
                        keyword = str(item).strip()

                    keyword_clean = self.clean_text(keyword)
                    if keyword_clean:
                        keywords.append(keyword_clean)

                if keywords:
                    parts.append(f"Focus Areas: {', '.join(keywords)}")

                    # 도메인 자동 분류
                    domain = self._classify_journal_domain(journal_name, keywords)
                    parts.append(f"Domain: {domain}")

        # 6. 기타 통계 (선택적)
        if self.include_stats:
            unique_authors = node_data.get("unique_authors", 0)
            if unique_authors:
                parts.append(f"Authors: {unique_authors} contributors")

        # 결합 및 길이 제한
        combined_text = " ".join(parts)
        return self.truncate_text(combined_text)


# 프로세서 레지스트리
NODE_PROCESSORS = {
    "paper": PaperTextProcessor,
    "author": AuthorTextProcessor,
    "keyword": KeywordTextProcessor,
    "journal": JournalTextProcessor,
}


def create_text_processor(
    node_type: str, max_length: int = 512, language: str = "mixed", **kwargs
) -> BaseNodeTextProcessor:
    """텍스트 프로세서 팩토리 함수

    Args:
        node_type: 노드 타입 ("paper", "author", "keyword", "journal")
        max_length: 최대 텍스트 길이
        language: 주요 언어
        **kwargs: 각 프로세서별 추가 설정

    Returns:
        BaseNodeTextProcessor 인스턴스
    """
    if node_type not in NODE_PROCESSORS:
        raise ValueError(f"Unsupported node type: {node_type}")

    processor_class = NODE_PROCESSORS[node_type]
    return processor_class(max_length=max_length, language=language, **kwargs)


def get_supported_node_types() -> List[str]:
    """지원되는 노드 타입 목록 반환"""
    return list(NODE_PROCESSORS.keys())


if __name__ == "__main__":
    # 테스트 실행
    print("🧪 Testing Node Text Processors...")

    # 테스트 데이터
    test_data = {
        "paper": {
            "id": "paper_1",
            "title": "Deep Learning for Battery State of Charge Prediction",
            "abstract": "This paper presents a novel deep learning approach for accurate battery SoC prediction using LSTM networks.",
            "keywords": ["deep learning", "battery", "SoC", "LSTM", "prediction"],
            "authors": ["김철수", "John Smith", "박영희"],
            "journal": "IEEE Transactions on Power Electronics",
            "year": "2023",
        },
        "author": {
            "id": "김철수",
            "name": "김철수",
            "paper_count": 15,
            "first_year": 2018,
            "last_year": 2023,
            "top_keywords": [
                ("machine learning", 8),
                ("battery", 6),
                ("optimization", 4),
            ],
            "most_frequent_journal": "IEEE Transactions on Industrial Electronics",
            "productivity_type": "Leading Researcher",
        },
        "keyword": {
            "id": "machine learning",
            "name": "machine learning",
            "frequency": 25,
            "papers": ["paper_1", "paper_2", "paper_3"],
        },
        "journal": {
            "id": "IEEE Transactions on Power Electronics",
            "name": "IEEE Transactions on Power Electronics",
            "paper_count": 120,
            "first_year": 2015,
            "last_year": 2023,
            "journal_type": "IEEE",
            "top_keywords": [
                ("power electronics", 45),
                ("battery", 32),
                ("control", 28),
            ],
            "unique_authors": 180,
        },
    }

    print("=" * 60)

    for node_type, data in test_data.items():
        print(f"\n📝 Testing {node_type.upper()} processor:")
        print("-" * 40)

        processor = create_text_processor(node_type, max_length=200, language="mixed")
        result = processor.process_node(data)

        print(f"Input: {data}")
        print(f"Output: {result}")
        print(f"Length: {len(result.split())} words")

    print(f"\n✅ All processors tested successfully!")
    print(f"📋 Supported node types: {get_supported_node_types()}")
