"""
GraphRAG 쿼리 분석 모듈
Query Analyzer for GraphRAG System

사용자 질문을 분석하여 최적의 검색 전략을 결정합니다.
- 쿼리 복잡도 자동 판단
- 필요한 그래프 노드 타입 식별
- 검색 모드 추천
- 한국어/영어 지원
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import Counter
import numpy as np

# 로깅 설정
logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """쿼리 복잡도 레벨"""

    SIMPLE = "simple"  # 단순 조회 (특정 저자, 논문 등)
    MEDIUM = "medium"  # 중간 복잡도 (트렌드, 패턴 분석)
    COMPLEX = "complex"  # 복잡한 분석 (다중 홉, 종합 분석)
    EXPLORATORY = "exploratory"  # 탐색적 분석 (전체 구조, 숨겨진 패턴)


class QueryType(Enum):
    """쿼리 타입 분류"""

    CITATION_ANALYSIS = "citation_analysis"  # 인용 분석
    AUTHOR_ANALYSIS = "author_analysis"  # 연구자 분석
    KEYWORD_ANALYSIS = "keyword_analysis"  # 키워드/주제 분석
    JOURNAL_ANALYSIS = "journal_analysis"  # 저널 분석
    TREND_ANALYSIS = "trend_analysis"  # 트렌드 분석
    COLLABORATION_ANALYSIS = "collaboration_analysis"  # 협업 분석
    SIMILARITY_ANALYSIS = "similarity_analysis"  # 유사도 분석
    COMPREHENSIVE_ANALYSIS = "comprehensive_analysis"  # 종합 분석
    FACTUAL_LOOKUP = "factual_lookup"  # 단순 조회
    COMPARISON = "comparison"  # 비교 분석


class SearchMode(Enum):
    """검색 모드"""

    LOCAL = "local"  # 특정 엔티티 중심 검색
    GLOBAL = "global"  # 전역 패턴 분석
    HYBRID = "hybrid"  # 하이브리드 접근


class NodeType(Enum):
    """그래프 노드 타입"""

    PAPER = "paper"
    AUTHOR = "author"
    KEYWORD = "keyword"
    JOURNAL = "journal"


@dataclass
class QueryAnalysisResult:
    """쿼리 분석 결과"""

    # 기본 정보
    original_query: str
    processed_query: str
    language: str

    # 분류 결과
    complexity: QueryComplexity
    query_type: QueryType
    search_mode: SearchMode

    # 필요 리소스
    required_node_types: Set[NodeType]
    required_edge_types: Set[str]
    estimated_scope: str  # "narrow", "medium", "broad"

    # 추출된 엔티티
    entities: Dict[str, List[str]]  # 타입별 엔티티 리스트
    keywords: List[str]
    temporal_indicators: List[str]

    # 메타데이터
    confidence_score: float
    processing_hints: List[str]
    estimated_complexity_score: float
    suggested_timeout: int  # 초 단위

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (JSON 직렬화 용)"""
        result = asdict(self)

        # Enum들을 문자열로 변환
        result["complexity"] = self.complexity.value
        result["query_type"] = self.query_type.value
        result["search_mode"] = self.search_mode.value
        result["required_node_types"] = [nt.value for nt in self.required_node_types]

        return result


class QueryAnalyzer:
    """GraphRAG 쿼리 분석기"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 분석기 설정
        """
        self.config = config or self._get_default_config()

        # 언어별 패턴 로드
        self._load_language_patterns()

        # 도메인 특화 키워드 로드
        self._load_domain_keywords()

        # 복잡도 분석 가중치
        self._load_complexity_weights()

        logger.info("✅ QueryAnalyzer initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정"""
        return {
            "supported_languages": ["ko", "en"],
            "default_language": "ko",
            "complexity_threshold": {
                "simple_max": 0.3,
                "medium_max": 0.6,
                "complex_max": 0.8,
            },
            "entity_extraction": {"max_entities_per_type": 10, "min_confidence": 0.5},
            "timeout_settings": {
                "simple": 10,
                "medium": 30,
                "complex": 120,
                "exploratory": 300,
            },
        }

    def _load_language_patterns(self):
        """언어별 패턴 정의"""
        self.language_patterns = {
            "ko": {
                # 질문 패턴
                "question_patterns": [
                    r"무엇",
                    r"누구",
                    r"언제",
                    r"어디",
                    r"어떻게",
                    r"왜",
                    r"얼마나",
                    r"어떤",
                    r"몇",
                    r"어느",
                    r"뭐",
                    r"누가",
                ],
                # 복잡도 지시어
                "complexity_indicators": {
                    "simple": [
                        r"누구",
                        r"언제",
                        r"어디",
                        r"몇",
                        r"리스트",
                        r"목록",
                        r"찾아",
                        r"알려",
                        r"보여",
                    ],
                    "medium": [
                        r"동향",
                        r"트렌드",
                        r"패턴",
                        r"변화",
                        r"분석",
                        r"비교",
                        r"관계",
                        r"영향",
                        r"차이",
                    ],
                    "complex": [
                        r"종합",
                        r"전체적",
                        r"포괄적",
                        r"상관관계",
                        r"인과관계",
                        r"예측",
                        r"모델",
                        r"시뮬레이션",
                    ],
                    "exploratory": [
                        r"숨겨진",
                        r"발견",
                        r"탐색",
                        r"새로운",
                        r"혁신적",
                        r"예상치 못한",
                        r"놀라운",
                        r"특이한",
                    ],
                },
                # 쿼리 타입 키워드
                "query_type_keywords": {
                    "citation_analysis": [
                        r"인용",
                        r"참조",
                        r"영향력",
                        r"피인용",
                        r"h-index",
                    ],
                    "author_analysis": [
                        r"저자",
                        r"연구자",
                        r"교수",
                        r"박사",
                        r"연구진",
                        r"팀",
                    ],
                    "keyword_analysis": [r"키워드", r"주제", r"토픽", r"용어", r"개념"],
                    "collaboration_analysis": [
                        r"협업",
                        r"공동연구",
                        r"협력",
                        r"파트너십",
                        r"네트워크",
                    ],
                    "trend_analysis": [
                        r"동향",
                        r"트렌드",
                        r"변화",
                        r"발전",
                        r"진화",
                        r"성장",
                    ],
                },
            },
            "en": {
                "question_patterns": [
                    r"what",
                    r"who",
                    r"when",
                    r"where",
                    r"how",
                    r"why",
                    r"which",
                    r"whose",
                    r"whom",
                ],
                "complexity_indicators": {
                    "simple": [
                        r"who",
                        r"when",
                        r"where",
                        r"list",
                        r"show",
                        r"find",
                        r"tell",
                        r"give",
                    ],
                    "medium": [
                        r"trend",
                        r"pattern",
                        r"change",
                        r"analyze",
                        r"compare",
                        r"relationship",
                        r"influence",
                        r"impact",
                        r"difference",
                    ],
                    "complex": [
                        r"comprehensive",
                        r"overall",
                        r"correlation",
                        r"causation",
                        r"predict",
                        r"model",
                        r"simulate",
                        r"synthesize",
                    ],
                    "exploratory": [
                        r"hidden",
                        r"discover",
                        r"explore",
                        r"novel",
                        r"innovative",
                        r"unexpected",
                        r"surprising",
                        r"unusual",
                        r"emerging",
                    ],
                },
                "query_type_keywords": {
                    "citation_analysis": [
                        r"citation",
                        r"reference",
                        r"impact",
                        r"cited",
                        r"h-index",
                    ],
                    "author_analysis": [
                        r"author",
                        r"researcher",
                        r"professor",
                        r"scientist",
                        r"team",
                    ],
                    "keyword_analysis": [
                        r"keyword",
                        r"topic",
                        r"subject",
                        r"term",
                        r"concept",
                    ],
                    "collaboration_analysis": [
                        r"collaboration",
                        r"cooperation",
                        r"partnership",
                        r"network",
                    ],
                    "trend_analysis": [
                        r"trend",
                        r"change",
                        r"development",
                        r"evolution",
                        r"growth",
                    ],
                },
            },
        }

    def _load_domain_keywords(self):
        """학술 도메인 특화 키워드"""
        self.domain_keywords = {
            # 배터리/전기차 도메인
            "battery_ev": [
                "battery",
                "배터리",
                "lithium",
                "리튬",
                "soc",
                "상태",
                "electric vehicle",
                "전기차",
                "ev",
                "charging",
                "충전",
            ],
            # AI/ML 도메인
            "ai_ml": [
                "machine learning",
                "머신러닝",
                "deep learning",
                "딥러닝",
                "artificial intelligence",
                "인공지능",
                "neural network",
                "신경망",
            ],
            # 학술 용어
            "academic": [
                "research",
                "연구",
                "paper",
                "논문",
                "journal",
                "저널",
                "conference",
                "학회",
                "publication",
                "출판",
            ],
        }

    def _load_complexity_weights(self):
        """복잡도 계산 가중치"""
        self.complexity_weights = {
            "query_length": 0.1,  # 쿼리 길이
            "question_words": 0.15,  # 의문사 개수
            "complexity_terms": 0.25,  # 복잡도 지시어
            "entity_count": 0.2,  # 엔티티 개수
            "logical_operators": 0.15,  # 논리 연산자 (그리고, 또는 등)
            "temporal_scope": 0.15,  # 시간적 범위
        }

    def detect_language(self, query: str) -> str:
        """개선된 쿼리 언어 감지 (혼용 텍스트 지원)"""
        # 1. 기본 문자 통계
        korean_chars = len(re.findall(r"[가-힣]", query))
        english_chars = len(re.findall(r"[a-zA-Z]", query))
        total_chars = korean_chars + english_chars

        if total_chars == 0:
            return self.config["default_language"]

        korean_ratio = korean_chars / total_chars

        # 2. 언어별 핵심 패턴 매칭
        korean_patterns = [
            r"[가-힣]+(?:교수|박사|연구원|저자)",  # 한국어 직책
            r"[가-힣]+(?:의|이|가|을|를|에서)",  # 한국어 조사
            r"(?:무엇|누구|언제|어디|어떻게|왜)",  # 한국어 의문사
            r"(?:동향|트렌드|분석|연구|논문)",  # 한국어 학술용어
        ]

        english_patterns = [
            r"\b(?:what|who|when|where|how|why)\b",  # 영어 의문사
            r"\b(?:analysis|research|trend|paper)\b",  # 영어 학술용어
            r"\b(?:Dr|Prof|Professor)\s+[A-Z][a-z]+",  # 영어 직책
            r"\b[A-Z][a-z]+\s+(?:et\s+al|and\s+[A-Z])",  # 영어 저자 패턴
        ]

        korean_pattern_score = 0
        english_pattern_score = 0

        for pattern in korean_patterns:
            korean_pattern_score += len(re.findall(pattern, query))

        for pattern in english_patterns:
            english_pattern_score += len(re.findall(pattern, query, re.IGNORECASE))

        # 3. 도메인 전문용어 고려
        domain_terms = {
            "technical_english": [
                "machine learning",
                "deep learning",
                "neural network",
                "SoC",
                "IoT",
                "AI",
                "ML",
                "CNN",
                "LSTM",
                "RNN",
                "battery",
                "lithium",
                "charging",
                "electric vehicle",
            ],
            "korean_academic": [
                "연구",
                "논문",
                "분석",
                "개발",
                "기술",
                "시스템",
                "알고리즘",
                "모델",
                "데이터",
                "성능",
            ],
        }

        technical_english_count = 0
        for term in domain_terms["technical_english"]:
            if term.lower() in query.lower():
                technical_english_count += 1

        korean_academic_count = 0
        for term in domain_terms["korean_academic"]:
            if term in query:
                korean_academic_count += 1

        # 4. 종합 판단 로직
        # 혼용 텍스트 감지
        is_mixed = (
            korean_ratio > 0.1
            and korean_ratio < 0.9
            and technical_english_count > 0
            and korean_academic_count > 0
        )

        if is_mixed:
            # 혼용인 경우 주요 언어 패턴으로 결정
            if korean_pattern_score > english_pattern_score:
                return "ko"  # 한국어 주도
            elif english_pattern_score > korean_pattern_score:
                return "en"  # 영어 주도
            else:
                # 패턴 점수가 같으면 문자 비율로 결정
                return "ko" if korean_ratio >= 0.5 else "en"

        # 5. 순수 언어인 경우
        if korean_ratio > 0.7:
            return "ko"
        elif korean_ratio < 0.3:
            return "en"
        else:
            # 애매한 경우 패턴 점수로 결정
            total_korean_score = korean_pattern_score + korean_academic_count
            total_english_score = english_pattern_score + technical_english_count

            if total_korean_score > total_english_score:
                return "ko"
            elif total_english_score > total_korean_score:
                return "en"
            else:
                # 최종적으로 문자 비율로 결정
                return "ko" if korean_ratio >= 0.5 else "en"

    def preprocess_query(self, query: str) -> str:
        """쿼리 전처리"""
        # 기본 정제
        processed = query.strip()

        # 연속 공백 제거
        processed = re.sub(r"\s+", " ", processed)

        # 특수문자 정리 (의미있는 것들은 보존)
        processed = re.sub(r"[^\w\s\?\!\.\,\(\)\-]", " ", processed)

        return processed

    def extract_entities(self, query: str, language: str) -> Dict[str, List[str]]:
        """혼용 텍스트 지원 엔티티 추출"""
        entities = {
            "authors": [],
            "papers": [],
            "keywords": [],
            "journals": [],
            "years": [],
            "institutions": [],
        }

        # 년도 추출 (언어 무관)
        years = re.findall(r"\b(19|20)\d{2}\b", query)
        entities["years"] = years

        # 혼용 텍스트를 위한 다중 패턴 적용
        korean_chars = len(re.findall(r"[가-힣]", query))
        english_chars = len(re.findall(r"[a-zA-Z]", query))
        is_mixed = korean_chars > 0 and english_chars > 0

        # 저자명 패턴 (혼용 고려)
        author_patterns = []

        if language == "ko" or is_mixed:
            # 한국어 저자 패턴
            author_patterns.extend(
                [
                    r"([가-힣]{2,4})\s*(?:교수|박사|연구원|저자)",
                    r"([가-힣]{2,4})\s*(?:등|외)",
                    r"([가-힣]{2,4})\s*(?:의|이|가)\s*연구",
                ]
            )

        if language == "en" or is_mixed:
            # 영어 저자 패턴
            author_patterns.extend(
                [
                    r"([A-Z][a-z]+\s+[A-Z][a-z]+)",
                    r"([A-Z]\.\s*[A-Z][a-z]+)",
                    r"Dr\.\s*([A-Z][a-z]+\s+[A-Z][a-z]+)",
                    r"Prof\.\s*([A-Z][a-z]+\s+[A-Z][a-z]+)",
                    r"Professor\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
                ]
            )

        # 혼용 패턴 (한영 조합)
        if is_mixed:
            author_patterns.extend(
                [
                    r"([가-힣]{2,4})\s+(?:Dr|Prof|Professor)",  # 김철수 Dr
                    r"(?:Dr|Prof|Professor)\s+([가-힣]{2,4})",  # Dr 김철수
                    r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s*(?:교수|박사)",  # John Smith 교수
                ]
            )

        for pattern in author_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities["authors"].extend(matches)

        # 저널명 패턴 (혼용 고려)
        journal_patterns = [
            r"(IEEE\s+[A-Za-z\s]+)",
            r"(Nature\s+[A-Za-z\s]+)",
            r"(Journal\s+of\s+[A-Za-z\s]+)",
            r"([가-힣\s]*학회지)",
            r"([가-힣\s]*저널)",
        ]

        for pattern in journal_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities["journals"].extend(matches)

        # 기관명 패턴
        institution_patterns = [
            r"([가-힣]+대학교?)",
            r"([가-힣]+연구소)",
            r"([A-Z][a-z]+\s+University)",
            r"([A-Z][a-z]+\s+Institute)",
            r"(MIT|Stanford|Harvard|KAIST|서울대|연세대|고려대)",
        ]

        for pattern in institution_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities["institutions"].extend(matches)

        # 도메인 키워드 매칭 (다국어)
        query_lower = query.lower()
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    entities["keywords"].append(keyword)

        # 추가 기술 키워드 추출 (혼용 고려)
        technical_terms = [
            # 영어 기술용어
            r"\b(machine\s+learning|deep\s+learning|neural\s+network)\b",
            r"\b(artificial\s+intelligence|reinforcement\s+learning)\b",
            r"\b(battery|lithium|charging|electric\s+vehicle)\b",
            r"\b(SoC|IoT|AI|ML|CNN|LSTM|RNN|GPU)\b",
            # 한국어 기술용어
            r"(머신\s*러닝|딥\s*러닝|신경망)",
            r"(인공지능|강화학습|전기차)",
            r"(배터리|리튬|충전|자율주행)",
        ]

        for pattern in technical_terms:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities["keywords"].extend([match for match in matches if match])

        # 중복 제거 및 정제
        for key in entities:
            if key == "keywords":
                # 키워드는 소문자로 정규화
                entities[key] = list(
                    set([kw.lower().strip() for kw in entities[key] if kw.strip()])
                )
            else:
                entities[key] = list(
                    set([item.strip() for item in entities[key] if item.strip()])
                )

        return entities

    def calculate_complexity_score(
        self, query: str, language: str, entities: Dict[str, List[str]]
    ) -> float:
        """혼용 텍스트 지원 쿼리 복잡도 점수 계산 (0-1 범위)"""
        scores = {}

        # 혼용 텍스트 감지
        korean_chars = len(re.findall(r"[가-힣]", query))
        english_chars = len(re.findall(r"[a-zA-Z]", query))
        is_mixed = korean_chars > 0 and english_chars > 0

        # 1. 쿼리 길이 점수
        length_score = min(1.0, len(query.split()) / 20)
        scores["query_length"] = length_score

        # 2. 의문사 개수 (다국어 지원)
        question_words = 0

        # 기본 언어 패턴
        patterns = self.language_patterns[language]
        for pattern in patterns["question_patterns"]:
            question_words += len(re.findall(pattern, query, re.IGNORECASE))

        # 혼용인 경우 다른 언어 패턴도 적용
        if is_mixed:
            other_language = "en" if language == "ko" else "ko"
            if other_language in self.language_patterns:
                other_patterns = self.language_patterns[other_language]
                for pattern in other_patterns["question_patterns"]:
                    question_words += len(re.findall(pattern, query, re.IGNORECASE))

        scores["question_words"] = min(1.0, question_words / 3)

        # 3. 복잡도 지시어 점수 (다국어)
        complexity_score = 0

        # 기본 언어의 복잡도 지시어
        for level, terms in patterns["complexity_indicators"].items():
            weight = {"simple": 0.2, "medium": 0.5, "complex": 0.8, "exploratory": 1.0}[
                level
            ]
            for term in terms:
                if re.search(term, query, re.IGNORECASE):
                    complexity_score = max(complexity_score, weight)

        # 혼용인 경우 다른 언어 패턴도 확인
        if is_mixed:
            other_language = "en" if language == "ko" else "ko"
            if other_language in self.language_patterns:
                other_patterns = self.language_patterns[other_language]
                for level, terms in other_patterns["complexity_indicators"].items():
                    weight = {
                        "simple": 0.2,
                        "medium": 0.5,
                        "complex": 0.8,
                        "exploratory": 1.0,
                    }[level]
                    for term in terms:
                        if re.search(term, query, re.IGNORECASE):
                            complexity_score = max(complexity_score, weight)

        # 추가 복잡도 지시어 (도메인 특화)
        advanced_terms = [
            # 고급 분석 용어
            r"(?:comprehensive|전체적|overall|종합)",
            r"(?:correlation|상관관계|causation|인과관계)",
            r"(?:prediction|예측|forecasting|전망)",
            r"(?:network\s+analysis|네트워크\s*분석)",
            r"(?:hidden\s+pattern|숨겨진\s*패턴)",
            r"(?:deep\s+analysis|심층\s*분석)",
            # 다중 개념 조합
            r"(?:and|그리고|또한|뿐만\s*아니라)",
            r"(?:both|둘\s*다|모두)",
            r"(?:relationship\s+between|관계|사이)",
        ]

        for term_pattern in advanced_terms:
            if re.search(term_pattern, query, re.IGNORECASE):
                complexity_score = max(complexity_score, 0.7)

        scores["complexity_terms"] = complexity_score

        # 4. 엔티티 개수 (다양성 고려)
        total_entities = sum(len(ents) for ents in entities.values())
        entity_types = sum(
            1 for ents in entities.values() if ents
        )  # 엔티티 타입 다양성

        entity_score = min(1.0, total_entities / 10)
        # 엔티티 타입이 다양하면 복잡도 증가
        if entity_types >= 3:
            entity_score = min(1.0, entity_score * 1.3)

        scores["entity_count"] = entity_score

        # 5. 논리 연산자 (다국어)
        logical_ops = [
            # 한국어
            r"(?:그리고|또한|또|더불어|뿐만\s*아니라)",
            r"(?:또는|혹은|아니면)",
            r"(?:하지만|그러나|그런데|반면)",
            r"(?:따라서|그러므로|결과적으로)",
            # 영어
            r"\b(?:and|also|furthermore|moreover)\b",
            r"\b(?:or|either|alternatively)\b",
            r"\b(?:but|however|nevertheless|whereas)\b",
            r"\b(?:therefore|thus|consequently)\b",
        ]

        logical_count = 0
        for op_pattern in logical_ops:
            logical_count += len(re.findall(op_pattern, query, re.IGNORECASE))

        scores["logical_operators"] = min(1.0, logical_count / 3)

        # 6. 시간적 범위 (다국어)
        temporal_indicators = [
            # 한국어 시간 지시어
            r"(?:최근|요즘|근래|지금)",
            r"(?:과거|예전|이전|전에)",
            r"(?:미래|앞으로|향후|장래)",
            r"(?:변화|발전|진화|추이)",
            r"(?:동향|트렌드|경향)",
            r"(?:역사|발달과정|변천)",
            # 영어 시간 지시어
            r"\b(?:recent|recently|current|now|today)\b",
            r"\b(?:past|previous|former|before|ago)\b",
            r"\b(?:future|upcoming|next|coming)\b",
            r"\b(?:change|development|evolution|progress)\b",
            r"\b(?:trend|tendency|pattern)\b",
            r"\b(?:history|historical|timeline)\b",
        ]

        temporal_score = 0
        temporal_count = 0
        for indicator_pattern in temporal_indicators:
            matches = re.findall(indicator_pattern, query, re.IGNORECASE)
            if matches:
                temporal_count += len(matches)

        if temporal_count > 0:
            temporal_score = min(1.0, 0.4 + (temporal_count * 0.2))

        scores["temporal_scope"] = temporal_score

        # 7. 혼용 텍스트 보너스 (혼용 자체가 복잡성을 나타냄)
        if is_mixed:
            # 혼용 텍스트는 일반적으로 더 복잡한 개념을 다룸
            mixed_bonus = 0.1
            for key in scores:
                scores[key] = min(1.0, scores[key] + mixed_bonus)

        # 가중 평균 계산
        total_score = sum(scores[key] * self.complexity_weights[key] for key in scores)

        return min(1.0, total_score)

    def classify_query_type(
        self, query: str, language: str, entities: Dict[str, List[str]]
    ) -> QueryType:
        """쿼리 타입 분류"""
        query_lower = query.lower()
        patterns = self.language_patterns[language]

        # 각 타입별 점수 계산
        type_scores = {}

        for query_type, keywords in patterns["query_type_keywords"].items():
            score = 0
            for keyword in keywords:
                if re.search(keyword, query_lower):
                    score += 1
            type_scores[query_type] = score

        # 엔티티 기반 추가 점수
        if entities["authors"]:
            type_scores["author_analysis"] = type_scores.get("author_analysis", 0) + 2

        if entities["keywords"]:
            type_scores["keyword_analysis"] = type_scores.get("keyword_analysis", 0) + 2

        if entities["years"]:
            type_scores["trend_analysis"] = type_scores.get("trend_analysis", 0) + 1

        # 특별 패턴 검사
        comprehensive_patterns = [
            r"종합",
            r"전체",
            r"모든",
            r"전반적",
            r"overall",
            r"comprehensive",
            r"all",
        ]

        for pattern in comprehensive_patterns:
            if re.search(pattern, query_lower):
                type_scores["comprehensive_analysis"] = (
                    type_scores.get("comprehensive_analysis", 0) + 3
                )

        # 최고 점수 타입 선택
        if not type_scores or max(type_scores.values()) == 0:
            return QueryType.FACTUAL_LOOKUP

        best_type = max(type_scores.items(), key=lambda x: x[1])[0]
        return QueryType(best_type)

    def determine_complexity(self, complexity_score: float) -> QueryComplexity:
        """복잡도 점수로부터 복잡도 레벨 결정"""
        thresholds = self.config["complexity_threshold"]

        if complexity_score <= thresholds["simple_max"]:
            return QueryComplexity.SIMPLE
        elif complexity_score <= thresholds["medium_max"]:
            return QueryComplexity.MEDIUM
        elif complexity_score <= thresholds["complex_max"]:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.EXPLORATORY

    def determine_required_resources(
        self, query_type: QueryType, entities: Dict[str, List[str]]
    ) -> Tuple[Set[NodeType], Set[str]]:
        """필요한 노드/엣지 타입 결정"""

        # 쿼리 타입별 기본 리소스
        type_resources = {
            QueryType.CITATION_ANALYSIS: {
                "nodes": {NodeType.PAPER},
                "edges": {"cites", "semantically_similar_to"},
            },
            QueryType.AUTHOR_ANALYSIS: {
                "nodes": {NodeType.AUTHOR, NodeType.PAPER},
                "edges": {"authored_by", "collaborates_with"},
            },
            QueryType.KEYWORD_ANALYSIS: {
                "nodes": {NodeType.KEYWORD, NodeType.PAPER},
                "edges": {"has_keyword", "co_occurs_with"},
            },
            QueryType.COLLABORATION_ANALYSIS: {
                "nodes": {NodeType.AUTHOR, NodeType.PAPER},
                "edges": {"collaborates_with", "authored_by"},
            },
            QueryType.TREND_ANALYSIS: {
                "nodes": {NodeType.KEYWORD, NodeType.PAPER, NodeType.AUTHOR},
                "edges": {"has_keyword", "temporal_proximity", "authored_by"},
            },
            QueryType.COMPREHENSIVE_ANALYSIS: {
                "nodes": {
                    NodeType.PAPER,
                    NodeType.AUTHOR,
                    NodeType.KEYWORD,
                    NodeType.JOURNAL,
                },
                "edges": {
                    "cites",
                    "authored_by",
                    "has_keyword",
                    "collaborates_with",
                    "co_occurs_with",
                    "published_in",
                },
            },
        }

        # 기본 리소스
        resources = type_resources.get(
            query_type, {"nodes": {NodeType.PAPER}, "edges": {"cites"}}
        )

        required_nodes = set(resources["nodes"])
        required_edges = set(resources["edges"])

        # 엔티티 기반 추가 리소스
        if entities["authors"]:
            required_nodes.add(NodeType.AUTHOR)
            required_edges.update(["authored_by", "collaborates_with"])

        if entities["keywords"]:
            required_nodes.add(NodeType.KEYWORD)
            required_edges.update(["has_keyword", "co_occurs_with"])

        if entities["years"]:
            required_edges.add("temporal_proximity")

        return required_nodes, required_edges

    def determine_search_mode(
        self,
        complexity: QueryComplexity,
        query_type: QueryType,
        entities: Dict[str, List[str]],
    ) -> SearchMode:
        """검색 모드 결정"""

        # 특정 엔티티가 많이 언급되면 LOCAL
        total_specific_entities = len(entities["authors"]) + len(entities["papers"])

        if total_specific_entities >= 2:
            return SearchMode.LOCAL

        # 탐색적이거나 종합적 분석이면 GLOBAL
        if (
            complexity == QueryComplexity.EXPLORATORY
            or query_type == QueryType.COMPREHENSIVE_ANALYSIS
        ):
            return SearchMode.GLOBAL

        # 트렌드나 패턴 분석이면 HYBRID
        if query_type in [QueryType.TREND_ANALYSIS, QueryType.COLLABORATION_ANALYSIS]:
            return SearchMode.HYBRID

        # 기본값
        return SearchMode.LOCAL if total_specific_entities > 0 else SearchMode.GLOBAL

    def generate_processing_hints(self, result: QueryAnalysisResult) -> List[str]:
        """처리 힌트 생성"""
        hints = []

        # 복잡도별 힌트
        if result.complexity == QueryComplexity.SIMPLE:
            hints.append("Direct entity lookup recommended")
        elif result.complexity == QueryComplexity.EXPLORATORY:
            hints.append("Consider community detection algorithms")
            hints.append("Enable broad graph traversal")

        # 쿼리 타입별 힌트
        if result.query_type == QueryType.TREND_ANALYSIS:
            hints.append("Include temporal analysis")
            hints.append("Consider time-series aggregation")

        if result.query_type == QueryType.COLLABORATION_ANALYSIS:
            hints.append("Focus on author-author relationships")
            hints.append("Calculate network centrality metrics")

        # 리소스별 힌트
        if NodeType.AUTHOR in result.required_node_types:
            hints.append("Include author disambiguation")

        if len(result.required_node_types) > 2:
            hints.append("Use multi-type graph traversal")

        return hints

    def analyze(self, query: str) -> QueryAnalysisResult:
        """메인 분석 함수"""
        logger.info(f"🔍 Analyzing query: {query[:50]}...")

        # 1. 기본 전처리
        language = self.detect_language(query)
        processed_query = self.preprocess_query(query)

        # 2. 엔티티 추출
        entities = self.extract_entities(processed_query, language)

        # 3. 복잡도 계산
        complexity_score = self.calculate_complexity_score(
            processed_query, language, entities
        )
        complexity = self.determine_complexity(complexity_score)

        # 4. 쿼리 타입 분류
        query_type = self.classify_query_type(processed_query, language, entities)

        # 5. 필요 리소스 결정
        required_nodes, required_edges = self.determine_required_resources(
            query_type, entities
        )

        # 6. 검색 모드 결정
        search_mode = self.determine_search_mode(complexity, query_type, entities)

        # 7. 범위 추정
        if len(entities["authors"]) + len(entities["papers"]) > 3:
            estimated_scope = "narrow"
        elif complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPLORATORY]:
            estimated_scope = "broad"
        else:
            estimated_scope = "medium"

        # 8. 키워드 추출 (단순화)
        keywords = (
            entities["keywords"]
            + [
                word
                for word in processed_query.split()
                if len(word) > 3 and word.isalpha()
            ][:10]
        )  # 최대 10개

        # 9. 시간 지시어
        temporal_indicators = entities["years"] + [
            word
            for word in ["최근", "과거", "미래", "recent", "past", "future"]
            if word in processed_query.lower()
        ]

        # 10. 신뢰도 및 타임아웃 계산
        confidence_score = min(1.0, 0.5 + complexity_score * 0.5)
        suggested_timeout = self.config["timeout_settings"][complexity.value]

        # 11. 결과 구성
        result = QueryAnalysisResult(
            original_query=query,
            processed_query=processed_query,
            language=language,
            complexity=complexity,
            query_type=query_type,
            search_mode=search_mode,
            required_node_types=required_nodes,
            required_edge_types=required_edges,
            estimated_scope=estimated_scope,
            entities=entities,
            keywords=keywords,
            temporal_indicators=temporal_indicators,
            confidence_score=confidence_score,
            processing_hints=[],
            estimated_complexity_score=complexity_score,
            suggested_timeout=suggested_timeout,
        )

        # 12. 처리 힌트 생성
        result.processing_hints = self.generate_processing_hints(result)

        logger.info(f"✅ Analysis complete: {complexity.value} {query_type.value}")
        return result

    def batch_analyze(self, queries: List[str]) -> List[QueryAnalysisResult]:
        """여러 쿼리 일괄 분석"""
        logger.info(f"📊 Batch analyzing {len(queries)} queries...")

        results = []
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}")
            result = self.analyze(query)
            results.append(result)

        return results


def main():
    """테스트 실행 (혼용 텍스트 포함)"""
    analyzer = QueryAnalyzer()

    # 테스트 쿼리들 (혼용 텍스트 포함)
    test_queries = [
        # 순수 한국어
        "김철수 교수의 연구 논문 목록을 보여줘",
        "배터리 SoC 예측에 사용된 머신러닝 기법들의 동향은?",
        # 순수 영어
        "Who are the most cited authors in battery research?",
        "What are the recent trends in electric vehicle charging technology?",
        # 혼용 텍스트 (한글 주도)
        "김철수 교수의 machine learning 연구 실적은?",
        "battery SoC prediction에 대한 한국 연구자들의 동향은?",
        "IEEE journal에 발표된 딥러닝 논문들을 분석해줘",
        # 혼용 텍스트 (영어 주도)
        "What are the trends in 배터리 연구 by Korean researchers?",
        "Machine learning applications in 전기차 charging optimization",
        "Dr. Smith and 김철수 교수의 collaboration network",
        # 복잡한 혼용 분석
        "전기차 배터리 분야에서 가장 영향력 있는 international researchers들과 그들의 collaboration network를 comprehensive하게 분석해줘",
    ]

    print("🧪 Testing QueryAnalyzer with Mixed Language Support...")
    print("=" * 70)

    for i, query in enumerate(test_queries):
        print(f"\n{i+1}. Query: {query}")
        print("-" * 50)

        result = analyzer.analyze(query)

        # 언어 감지 결과
        korean_chars = len(re.findall(r"[가-힣]", query))
        english_chars = len(re.findall(r"[a-zA-Z]", query))
        total_chars = korean_chars + english_chars
        korean_ratio = korean_chars / total_chars if total_chars > 0 else 0

        print(f"📝 Language Analysis:")
        print(f"   Detected: {result.language}")
        print(f"   Korean ratio: {korean_ratio:.1%} ({korean_chars}/{total_chars})")
        print(
            f"   Mixed text: {'Yes' if korean_chars > 0 and english_chars > 0 else 'No'}"
        )

        print(f"🔍 Analysis Results:")
        print(
            f"   Complexity: {result.complexity.value} (score: {result.estimated_complexity_score:.3f})"
        )
        print(f"   Type: {result.query_type.value}")
        print(f"   Search Mode: {result.search_mode.value}")
        print(f"   Required Nodes: {[nt.value for nt in result.required_node_types]}")

        # 엔티티 추출 결과 (혼용 텍스트 특별히 확인)
        if any(result.entities.values()):
            print(f"🎯 Extracted Entities:")
            for entity_type, entities in result.entities.items():
                if entities:
                    print(f"   {entity_type}: {entities}")

        print(f"⚡ Performance:")
        print(f"   Confidence: {result.confidence_score:.3f}")
        print(f"   Timeout: {result.suggested_timeout}s")

        # 혼용 텍스트 특별 힌트
        if korean_chars > 0 and english_chars > 0:
            print(f"🌐 Mixed Language Hints:")
            if "machine learning" in query.lower() or "deep learning" in query.lower():
                print(f"   - Technical English terms detected")
            if any(term in query for term in ["교수", "연구", "분석"]):
                print(f"   - Korean academic terms detected")
            print(f"   - Applied multi-language pattern matching")

    print(f"\n✅ QueryAnalyzer testing completed!")
    print(f"🌐 Mixed language support validated!")


if __name__ == "__main__":
    main()
