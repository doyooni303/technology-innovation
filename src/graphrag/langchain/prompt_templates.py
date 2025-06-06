"""
GraphRAG 프롬프트 템플릿 모듈
Prompt Templates for GraphRAG System

쿼리 타입별 최적화된 프롬프트 템플릿 제공
- 쿼리 분석 결과 기반 동적 프롬프트 생성
- 한국어/영어/혼용 언어 지원
- 복잡도별 프롬프트 최적화
- LangChain PromptTemplate 완전 호환
"""

import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import warnings

# LangChain imports
try:
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    from langchain_core.prompts.base import BasePromptTemplate
    from langchain_core.messages import SystemMessage, HumanMessage

    _langchain_available = True
except ImportError:
    _langchain_available = False
    warnings.warn(
        "LangChain not available. Install with: pip install langchain langchain-core"
    )

    # Placeholder classes
    class PromptTemplate:
        def __init__(self, *args, **kwargs):
            pass

    class ChatPromptTemplate:
        def __init__(self, *args, **kwargs):
            pass

    class BasePromptTemplate:
        def __init__(self, *args, **kwargs):
            pass


# GraphRAG imports
try:
    from ..query_analyzer import QueryType, QueryComplexity, QueryAnalysisResult
except ImportError as e:
    warnings.warn(f"GraphRAG QueryAnalyzer not available: {e}")

    # Placeholder enums
    class QueryType(Enum):
        CITATION_ANALYSIS = "citation_analysis"
        AUTHOR_ANALYSIS = "author_analysis"
        KEYWORD_ANALYSIS = "keyword_analysis"
        TREND_ANALYSIS = "trend_analysis"
        COLLABORATION_ANALYSIS = "collaboration_analysis"
        COMPREHENSIVE_ANALYSIS = "comprehensive_analysis"
        FACTUAL_LOOKUP = "factual_lookup"

    class QueryComplexity(Enum):
        SIMPLE = "simple"
        MEDIUM = "medium"
        COMPLEX = "complex"
        EXPLORATORY = "exploratory"


# 로깅 설정
logger = logging.getLogger(__name__)


@dataclass
class PromptConfig:
    """프롬프트 설정 클래스"""

    language: str = "mixed"  # "ko", "en", "mixed"
    style: str = "academic"  # "academic", "conversational", "technical"
    include_metadata: bool = True
    include_confidence: bool = True
    max_context_length: int = 8000
    citation_style: str = "detailed"  # "minimal", "detailed", "academic"


class PromptStyle(Enum):
    """프롬프트 스타일"""

    ACADEMIC = "academic"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    CONCISE = "concise"


class GraphRAGPromptTemplates:
    """GraphRAG 프롬프트 템플릿 관리자"""

    def __init__(self, config: Optional[PromptConfig] = None):
        """
        Args:
            config: 프롬프트 설정
        """
        self.config = config or PromptConfig()

        # 언어별 기본 지시문
        self._load_base_instructions()

        # 쿼리 타입별 템플릿
        self._load_query_type_templates()

        # 복잡도별 템플릿
        self._load_complexity_templates()

        logger.info("✅ GraphRAGPromptTemplates initialized")
        logger.info(f"   🌐 Language: {self.config.language}")
        logger.info(f"   🎨 Style: {self.config.style}")

    def _load_base_instructions(self) -> None:
        """기본 지시문 로드"""
        self.base_instructions = {
            "ko": {
                "system_role": "당신은 과학 논문과 연구 데이터를 분석하는 전문 AI 어시스턴트입니다.",
                "task_description": "제공된 지식 그래프 정보를 바탕으로 사용자의 질문에 정확하고 유용한 답변을 제공해주세요.",
                "context_explanation": "다음은 질문과 관련된 논문, 저자, 키워드, 저널 정보가 포함된 지식 그래프 컨텍스트입니다:",
                "answer_guidelines": [
                    "제공된 컨텍스트 정보를 기반으로 답변하세요",
                    "구체적인 논문 제목, 저자명, 연도 등을 포함하여 답변하세요",
                    "불확실한 정보는 추측하지 말고 알 수 없다고 명시하세요",
                    "관련성이 높은 정보를 우선적으로 활용하세요",
                ],
                "citation_instruction": "답변에는 관련 논문이나 저자를 구체적으로 언급해주세요.",
            },
            "en": {
                "system_role": "You are an expert AI assistant specialized in analyzing scientific papers and research data.",
                "task_description": "Please provide accurate and helpful answers to user questions based on the provided knowledge graph information.",
                "context_explanation": "The following is knowledge graph context containing papers, authors, keywords, and journal information related to the question:",
                "answer_guidelines": [
                    "Base your answer on the provided context information",
                    "Include specific paper titles, author names, years, etc. in your response",
                    "If information is uncertain, explicitly state that it's unknown rather than guessing",
                    "Prioritize highly relevant information",
                ],
                "citation_instruction": "Please specifically mention relevant papers or authors in your answer.",
            },
            "mixed": {
                "system_role": "You are an expert AI assistant specialized in analyzing scientific papers and research data. 당신은 과학 논문과 연구 데이터 분석을 전문으로 하는 AI 어시스턴트입니다.",
                "task_description": "Please provide accurate and helpful answers based on the knowledge graph information. 지식 그래프 정보를 바탕으로 정확하고 유용한 답변을 제공해주세요.",
                "context_explanation": "다음은 질문과 관련된 지식 그래프 컨텍스트입니다 (The following is knowledge graph context related to the question):",
                "answer_guidelines": [
                    "제공된 컨텍스트 정보를 기반으로 답변하세요 (Base your answer on the provided context)",
                    "구체적인 논문 제목, 저자명을 포함하세요 (Include specific paper titles and author names)",
                    "불확실한 정보는 명시하세요 (Explicitly state uncertain information)",
                    "관련성 높은 정보를 우선하세요 (Prioritize highly relevant information)",
                ],
                "citation_instruction": "관련 논문이나 저자를 구체적으로 언급해주세요 (Please mention relevant papers or authors specifically).",
            },
        }

    def _load_query_type_templates(self) -> None:
        """쿼리 타입별 템플릿 로드"""
        self.query_type_templates = {
            QueryType.CITATION_ANALYSIS: {
                "ko": {
                    "specific_instruction": "인용 관계와 영향력을 중심으로 분석해주세요.",
                    "focus_areas": [
                        "피인용 수",
                        "인용 패턴",
                        "영향력 있는 논문",
                        "연구 영향도",
                    ],
                    "output_format": "인용 분석 결과를 구체적인 수치와 함께 제시해주세요.",
                },
                "en": {
                    "specific_instruction": "Focus on citation relationships and impact analysis.",
                    "focus_areas": [
                        "Citation counts",
                        "Citation patterns",
                        "Influential papers",
                        "Research impact",
                    ],
                    "output_format": "Present citation analysis results with specific metrics.",
                },
            },
            QueryType.AUTHOR_ANALYSIS: {
                "ko": {
                    "specific_instruction": "연구자의 연구 분야, 협업 네트워크, 연구 생산성을 분석해주세요.",
                    "focus_areas": [
                        "주요 연구 분야",
                        "공동 연구자",
                        "논문 수",
                        "연구 활동 기간",
                    ],
                    "output_format": "연구자별로 구분하여 상세한 프로필을 제공해주세요.",
                },
                "en": {
                    "specific_instruction": "Analyze researchers' fields, collaboration networks, and productivity.",
                    "focus_areas": [
                        "Research areas",
                        "Collaborators",
                        "Publication count",
                        "Research period",
                    ],
                    "output_format": "Provide detailed profiles for each researcher.",
                },
            },
            QueryType.KEYWORD_ANALYSIS: {
                "ko": {
                    "specific_instruction": "키워드와 연구 주제의 관계를 분석해주세요.",
                    "focus_areas": [
                        "키워드 빈도",
                        "관련 주제",
                        "연구 트렌드",
                        "주제 연관성",
                    ],
                    "output_format": "키워드별 사용 빈도와 관련 연구를 정리해주세요.",
                },
                "en": {
                    "specific_instruction": "Analyze relationships between keywords and research topics.",
                    "focus_areas": [
                        "Keyword frequency",
                        "Related topics",
                        "Research trends",
                        "Topic associations",
                    ],
                    "output_format": "Organize keyword frequencies and related research.",
                },
            },
            QueryType.TREND_ANALYSIS: {
                "ko": {
                    "specific_instruction": "시간의 흐름에 따른 연구 동향과 변화를 분석해주세요.",
                    "focus_areas": [
                        "연도별 연구량",
                        "주제 변화",
                        "새로운 트렌드",
                        "기술 발전",
                    ],
                    "output_format": "시간순으로 정리하여 트렌드 변화를 명확히 보여주세요.",
                },
                "en": {
                    "specific_instruction": "Analyze research trends and changes over time.",
                    "focus_areas": [
                        "Research volume by year",
                        "Topic evolution",
                        "Emerging trends",
                        "Technology development",
                    ],
                    "output_format": "Present trend changes clearly in chronological order.",
                },
            },
            QueryType.COLLABORATION_ANALYSIS: {
                "ko": {
                    "specific_instruction": "연구자 간 협업 관계와 네트워크를 분석해주세요.",
                    "focus_areas": [
                        "공동 연구",
                        "협업 빈도",
                        "연구 네트워크",
                        "기관 간 협력",
                    ],
                    "output_format": "협업 관계를 네트워크 형태로 설명해주세요.",
                },
                "en": {
                    "specific_instruction": "Analyze collaboration relationships and networks among researchers.",
                    "focus_areas": [
                        "Joint research",
                        "Collaboration frequency",
                        "Research networks",
                        "Institutional cooperation",
                    ],
                    "output_format": "Explain collaboration relationships in network format.",
                },
            },
            QueryType.COMPREHENSIVE_ANALYSIS: {
                "ko": {
                    "specific_instruction": "다양한 관점에서 종합적으로 분석해주세요.",
                    "focus_areas": [
                        "전체적 개요",
                        "주요 연구자",
                        "핵심 논문",
                        "연구 동향",
                        "향후 방향",
                    ],
                    "output_format": "섹션별로 구분하여 체계적으로 분석 결과를 제시해주세요.",
                },
                "en": {
                    "specific_instruction": "Provide comprehensive analysis from multiple perspectives.",
                    "focus_areas": [
                        "Overall overview",
                        "Key researchers",
                        "Core papers",
                        "Research trends",
                        "Future directions",
                    ],
                    "output_format": "Present analysis results systematically by sections.",
                },
            },
            QueryType.FACTUAL_LOOKUP: {
                "ko": {
                    "specific_instruction": "요청된 정보를 정확하고 간단명료하게 제공해주세요.",
                    "focus_areas": ["정확한 정보", "구체적 데이터", "명확한 답변"],
                    "output_format": "질문에 대한 직접적인 답변을 제공해주세요.",
                },
                "en": {
                    "specific_instruction": "Provide the requested information accurately and concisely.",
                    "focus_areas": [
                        "Accurate information",
                        "Specific data",
                        "Clear answers",
                    ],
                    "output_format": "Provide direct answers to the question.",
                },
            },
        }

    def _load_complexity_templates(self) -> None:
        """복잡도별 템플릿 로드"""
        self.complexity_templates = {
            QueryComplexity.SIMPLE: {
                "ko": {
                    "approach": "간단명료한 답변",
                    "detail_level": "핵심 정보만 포함",
                    "structure": "단답형 또는 짧은 설명",
                },
                "en": {
                    "approach": "Simple and clear answer",
                    "detail_level": "Include only key information",
                    "structure": "Short answer or brief explanation",
                },
            },
            QueryComplexity.MEDIUM: {
                "ko": {
                    "approach": "적절한 수준의 상세 분석",
                    "detail_level": "주요 내용과 배경 정보 포함",
                    "structure": "구조화된 설명 (2-3개 섹션)",
                },
                "en": {
                    "approach": "Moderately detailed analysis",
                    "detail_level": "Include main content and background",
                    "structure": "Structured explanation (2-3 sections)",
                },
            },
            QueryComplexity.COMPLEX: {
                "ko": {
                    "approach": "심층적이고 종합적인 분석",
                    "detail_level": "다양한 관점과 세부 정보 포함",
                    "structure": "체계적인 다중 섹션 구성",
                },
                "en": {
                    "approach": "In-depth and comprehensive analysis",
                    "detail_level": "Include multiple perspectives and details",
                    "structure": "Systematic multi-section organization",
                },
            },
            QueryComplexity.EXPLORATORY: {
                "ko": {
                    "approach": "탐색적이고 창의적인 분석",
                    "detail_level": "숨겨진 패턴과 새로운 인사이트 발굴",
                    "structure": "발견적 접근법으로 다각도 분석",
                },
                "en": {
                    "approach": "Exploratory and creative analysis",
                    "detail_level": "Discover hidden patterns and new insights",
                    "structure": "Multi-angle analysis with discovery approach",
                },
            },
        }

    def get_base_prompt(
        self, query_analysis: Optional[QueryAnalysisResult] = None
    ) -> str:
        """기본 프롬프트 생성"""

        # 언어 결정
        language = self._determine_language(query_analysis)
        base = self.base_instructions[language]

        # 기본 프롬프트 구성
        prompt_parts = [
            base["system_role"],
            "",
            base["task_description"],
            "",
            base["context_explanation"],
            "{context}",
            "",
        ]

        # 답변 가이드라인 추가
        if self.config.include_metadata:
            prompt_parts.append(
                "답변 가이드라인:"
                if language in ["ko", "mixed"]
                else "Answer Guidelines:"
            )
            for guideline in base["answer_guidelines"]:
                prompt_parts.append(f"- {guideline}")
            prompt_parts.append("")

        # 인용 지시문 추가
        if self.config.citation_style != "minimal":
            prompt_parts.append(base["citation_instruction"])
            prompt_parts.append("")

        prompt_parts.extend(
            [
                (
                    "질문: {question}"
                    if language in ["ko", "mixed"]
                    else "Question: {question}"
                ),
                "",
                "답변:" if language in ["ko", "mixed"] else "Answer:",
            ]
        )

        return "\n".join(prompt_parts)

    def get_query_specific_prompt(
        self, query_analysis: QueryAnalysisResult, include_base: bool = True
    ) -> str:
        """쿼리 타입별 특화 프롬프트 생성"""

        language = self._determine_language(query_analysis)
        query_type = query_analysis.query_type
        complexity = query_analysis.complexity

        # 기본 프롬프트
        if include_base:
            prompt_parts = [self.get_base_prompt(query_analysis)]
        else:
            prompt_parts = []

        # 쿼리 타입별 특화 지시문
        if query_type in self.query_type_templates:
            type_template = self.query_type_templates[query_type].get(
                language, self.query_type_templates[query_type].get("en", {})
            )

            if type_template:
                if language in ["ko", "mixed"]:
                    prompt_parts.append("\n특별 지시사항:")
                else:
                    prompt_parts.append("\nSpecific Instructions:")

                prompt_parts.append(
                    f"- {type_template.get('specific_instruction', '')}"
                )

                if "focus_areas" in type_template:
                    focus_label = (
                        "중점 분석 영역:"
                        if language in ["ko", "mixed"]
                        else "Focus Areas:"
                    )
                    prompt_parts.append(
                        f"- {focus_label} {', '.join(type_template['focus_areas'])}"
                    )

                if "output_format" in type_template:
                    prompt_parts.append(f"- {type_template['output_format']}")

        # 복잡도별 조정
        if complexity in self.complexity_templates:
            complexity_template = self.complexity_templates[complexity].get(
                language, self.complexity_templates[complexity].get("en", {})
            )

            if complexity_template:
                if language in ["ko", "mixed"]:
                    prompt_parts.append("\n답변 형식:")
                else:
                    prompt_parts.append("\nResponse Format:")

                for key, value in complexity_template.items():
                    prompt_parts.append(f"- {key}: {value}")

        # 신뢰도 정보 추가
        if self.config.include_confidence and hasattr(
            query_analysis, "confidence_score"
        ):
            confidence_text = (
                f"\n참고: 이 분석의 신뢰도는 {query_analysis.confidence_score:.2f}입니다."
                if language in ["ko", "mixed"]
                else f"\nNote: The confidence score for this analysis is {query_analysis.confidence_score:.2f}."
            )
            prompt_parts.append(confidence_text)

        return "\n".join(prompt_parts)

    def create_langchain_prompt(
        self,
        query_analysis: Optional[QueryAnalysisResult] = None,
        prompt_type: str = "base",  # "base", "query_specific", "chat"
    ) -> BasePromptTemplate:
        """LangChain 호환 프롬프트 템플릿 생성"""

        if not _langchain_available:
            raise ImportError("LangChain is required for creating prompt templates")

        if prompt_type == "chat":
            return self._create_chat_prompt_template(query_analysis)
        elif prompt_type == "query_specific" and query_analysis:
            prompt_text = self.get_query_specific_prompt(query_analysis)
        else:
            prompt_text = self.get_base_prompt(query_analysis)

        # PromptTemplate 생성
        return PromptTemplate(
            template=prompt_text,
            input_variables=["context", "question"],
            template_format="f-string",
        )

    def _create_chat_prompt_template(
        self, query_analysis: Optional[QueryAnalysisResult] = None
    ) -> ChatPromptTemplate:
        """채팅용 프롬프트 템플릿 생성"""

        language = self._determine_language(query_analysis)
        base = self.base_instructions[language]

        # 시스템 메시지
        system_content = f"{base['system_role']}\n\n{base['task_description']}"

        if query_analysis:
            # 쿼리별 특화 지시문 추가
            query_prompt = self.get_query_specific_prompt(
                query_analysis, include_base=False
            )
            if query_prompt.strip():
                system_content += f"\n\n{query_prompt}"

        system_message = SystemMessage(content=system_content)

        # 휴먼 메시지 템플릿
        human_content = f"{base['context_explanation']}\n{{context}}\n\n"
        human_content += (
            "질문: {question}"
            if language in ["ko", "mixed"]
            else "Question: {question}"
        )

        human_message = HumanMessage(content=human_content)

        return ChatPromptTemplate.from_messages([system_message, human_message])

    def _determine_language(
        self, query_analysis: Optional[QueryAnalysisResult] = None
    ) -> str:
        """사용할 언어 결정"""
        if query_analysis and hasattr(query_analysis, "language"):
            detected_lang = query_analysis.language
            if detected_lang in self.base_instructions:
                return detected_lang

        # 설정된 언어 사용
        if self.config.language in self.base_instructions:
            return self.config.language

        return "mixed"  # 기본값

    def update_config(self, **kwargs) -> None:
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"📝 Updated config.{key} = {value}")

    def get_available_templates(self) -> Dict[str, List[str]]:
        """사용 가능한 템플릿 목록 반환"""
        return {
            "query_types": [qt.value for qt in QueryType],
            "complexities": [qc.value for qc in QueryComplexity],
            "languages": list(self.base_instructions.keys()),
            "styles": [style.value for style in PromptStyle],
        }


# 편의 함수들
def create_base_prompt(
    language: str = "mixed", style: str = "academic", **kwargs
) -> PromptTemplate:
    """기본 프롬프트 생성 편의 함수"""
    config = PromptConfig(language=language, style=style, **kwargs)
    template_manager = GraphRAGPromptTemplates(config)
    return template_manager.create_langchain_prompt()


def create_query_prompt(
    query_analysis: QueryAnalysisResult, language: str = "auto", **kwargs
) -> PromptTemplate:
    """쿼리별 특화 프롬프트 생성 편의 함수"""
    if language == "auto":
        language = getattr(query_analysis, "language", "mixed")

    config = PromptConfig(language=language, **kwargs)
    template_manager = GraphRAGPromptTemplates(config)
    return template_manager.create_langchain_prompt(
        query_analysis=query_analysis, prompt_type="query_specific"
    )


def create_chat_prompt(
    query_analysis: Optional[QueryAnalysisResult] = None,
    language: str = "mixed",
    **kwargs,
) -> ChatPromptTemplate:
    """채팅용 프롬프트 생성 편의 함수"""
    config = PromptConfig(language=language, **kwargs)
    template_manager = GraphRAGPromptTemplates(config)
    return template_manager.create_langchain_prompt(
        query_analysis=query_analysis, prompt_type="chat"
    )


def main():
    """GraphRAGPromptTemplates 테스트"""
    print("🧪 Testing GraphRAGPromptTemplates...")

    # 설정 생성
    config = PromptConfig(
        language="mixed",
        style="academic",
        include_metadata=True,
        include_confidence=True,
    )

    # 템플릿 관리자 생성
    template_manager = GraphRAGPromptTemplates(config)

    # 기본 프롬프트 테스트
    print("📝 Base Prompt:")
    base_prompt = template_manager.get_base_prompt()
    print(base_prompt[:300] + "...")

    # 더미 쿼리 분석 결과
    class DummyQueryAnalysis:
        def __init__(self):
            self.language = "mixed"
            self.query_type = QueryType.KEYWORD_ANALYSIS
            self.complexity = QueryComplexity.MEDIUM
            self.confidence_score = 0.85

    dummy_analysis = DummyQueryAnalysis()

    # 쿼리별 특화 프롬프트 테스트
    print(f"\n📊 Query-Specific Prompt ({dummy_analysis.query_type.value}):")
    specific_prompt = template_manager.get_query_specific_prompt(dummy_analysis)
    print(specific_prompt[:400] + "...")

    # LangChain 프롬프트 생성 테스트
    if _langchain_available:
        print(f"\n🔗 LangChain PromptTemplate:")
        lc_prompt = template_manager.create_langchain_prompt(dummy_analysis)
        print(f"Input variables: {lc_prompt.input_variables}")

        print(f"\n💬 ChatPromptTemplate:")
        chat_prompt = template_manager.create_langchain_prompt(
            dummy_analysis, prompt_type="chat"
        )
        print(f"Messages: {len(chat_prompt.messages)}")

    # 사용 가능한 템플릿 목록
    print(f"\n📋 Available Templates:")
    available = template_manager.get_available_templates()
    for category, items in available.items():
        print(f"   {category}: {items}")

    print(f"\n✅ GraphRAGPromptTemplates test completed!")


if __name__ == "__main__":
    main()
