"""
PDF에서 키워드 및 Abstract 추출 모듈
PDF Keyword and Abstract Extraction Module
"""

import re
import json
import warnings
import logging
from pathlib import Path
import pdfplumber
import PyPDF2
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# PDF 관련 경고 메시지 억제
warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pypdf2")

# PyPDF2 로그 레벨 조정
logging.getLogger("pypdf2").setLevel(logging.ERROR)

# NLTK 데이터 다운로드 (한 번만 실행)
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    print("Downloading NLTK data...")
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)


class PDFKeywordExtractor:
    """PDF 파일에서 키워드와 Abstract를 추출하는 클래스"""

    def __init__(self):
        # 기술 관련 키워드 사전 (전기차 배터리 AI 도메인)
        self.tech_keywords = {
            # AI/ML 관련
            "machine learning",
            "deep learning",
            "artificial intelligence",
            "neural network",
            "reinforcement learning",
            "supervised learning",
            "unsupervised learning",
            "convolutional neural network",
            "cnn",
            "lstm",
            "rnn",
            "transformer",
            "federated learning",
            "transfer learning",
            "ensemble learning",
            # 배터리 관련
            "lithium-ion battery",
            "li-ion battery",
            "battery management system",
            "bms",
            "state of charge",
            "soc",
            "state of health",
            "soh",
            "battery capacity",
            "battery degradation",
            "battery aging",
            "battery thermal management",
            "electrode",
            "cathode",
            "anode",
            "electrolyte",
            "separator",
            "lithium iron phosphate",
            "lifepo4",
            "nickel manganese cobalt",
            "nmc",
            # 전기차 관련
            "electric vehicle",
            "ev",
            "electric car",
            "plug-in hybrid",
            "phev",
            "vehicle-to-grid",
            "v2g",
            "charging station",
            "fast charging",
            "wireless charging",
            "battery swapping",
            "energy management",
            # 제조/생산 관련
            "battery production",
            "battery manufacturing",
            "electrode manufacturing",
            "coating process",
            "calendering",
            "drying process",
            "cell assembly",
            "quality control",
            "process optimization",
            "production line",
            # 기타 기술
            "optimization",
            "internet of things",
            "iot",
            "cyber-physical system",
            "smart grid",
            "energy storage",
            "renewable energy",
            "sustainability",
        }

        # 불용어 설정
        self.stop_words = set(stopwords.words("english"))

        # 추가 불용어 (논문에서 흔한 단어들)
        self.additional_stopwords = {
            "paper",
            "study",
            "research",
            "method",
            "approach",
            "analysis",
            "result",
            "conclusion",
            "abstract",
            "introduction",
            "literature",
            "article",
            "journal",
            "conference",
            "proceedings",
            "ieee",
            "acm",
            "university",
            "department",
            "technology",
            "science",
            "engineering",
            "international",
            "national",
            "system",
            "systems",
            "based",
            "using",
        }
        self.stop_words.update(self.additional_stopwords)

    def clean_text(self, text):
        """텍스트 정제 함수"""
        if not text:
            return ""
        # 중괄호 제거
        text = re.sub(r"[{}]", "", str(text))
        # 연속된 공백 정리
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_text_from_pdf(self, pdf_path):
        """PDF에서 텍스트 추출 (경고 메시지 억제)"""
        text = ""

        try:
            # pdfplumber 사용 (더 정확한 텍스트 추출)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pdfplumber.open(pdf_path) as pdf:
                    # 처음 3페이지만 추출 (제목, 초록, 키워드가 보통 여기에 있음)
                    for page_num in range(min(3, len(pdf.pages))):
                        try:
                            page = pdf.pages[page_num]
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                        except Exception as page_error:
                            # 개별 페이지 오류는 무시하고 계속 진행
                            continue

        except Exception as e:
            # pdfplumber 실패시 PyPDF2 사용
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(pdf_path, "rb") as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        # 처음 3페이지만 추출
                        for page_num in range(min(3, len(pdf_reader.pages))):
                            try:
                                page = pdf_reader.pages[page_num]
                                extracted_text = page.extract_text()
                                if extracted_text:
                                    text += extracted_text + "\n"
                            except Exception as page_error:
                                # 개별 페이지 오류는 무시하고 계속 진행
                                continue
            except Exception as e2:
                # 조용히 실패 (tqdm에 방해되지 않도록)
                return ""

        return text

    def find_keywords_section(self, text):
        """텍스트에서 Keywords 섹션 찾기"""
        keywords = []

        # Keywords 섹션 패턴들
        keyword_patterns = [
            r"Keywords?[:\-\s]+(.*?)(?:\n\s*\n|\n\s*[A-Z]|\n\s*\d+)",
            r"Index Terms?[:\-\s]+(.*?)(?:\n\s*\n|\n\s*[A-Z]|\n\s*\d+)",
            r"Key words?[:\-\s]+(.*?)(?:\n\s*\n|\n\s*[A-Z]|\n\s*\d+)",
        ]

        for pattern in keyword_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # 키워드 텍스트 정제
                keyword_text = re.sub(r"\s+", " ", match).strip()
                # 쉼표, 세미콜론으로 분리
                kw_list = re.split(r"[,;]", keyword_text)
                keywords.extend([kw.strip().lower() for kw in kw_list if kw.strip()])

        return list(set(keywords))

    def extract_abstract(self, text):
        """초록 섹션 추출 (강화된 버전)"""
        if not text:
            return ""

        # 더 다양한 Abstract 패턴들
        abstract_patterns = [
            # 기본 패턴들
            r"Abstract[:\-\s]*\n\s*(.*?)(?:\n\s*\n|\n\s*(?:Keywords?|Index\s+Terms?|Introduction|1\.?\s*Introduction))",
            r"ABSTRACT[:\-\s]*\n\s*(.*?)(?:\n\s*\n|\n\s*(?:KEYWORDS?|INDEX\s+TERMS?|INTRODUCTION|1\.?\s*INTRODUCTION))",
            # 콜론 뒤 바로 텍스트
            r"Abstract[:\-\s]*[:\-]\s*(.*?)(?:\n\s*\n|\n\s*(?:Keywords?|Introduction|1\.?\s*Introduction))",
            r"ABSTRACT[:\-\s]*[:\-]\s*(.*?)(?:\n\s*\n|\n\s*(?:KEYWORDS?|INTRODUCTION|1\.?\s*INTRODUCTION))",
            # 더 유연한 끝점 탐지
            r"Abstract[:\-\s]+(.{100,2000}?)(?:\n\s*(?:Keywords?|Index\s+Terms?|Introduction|1\.?\s*Introduction|\d+\.?\s*Introduction))",
            r"ABSTRACT[:\-\s]+(.{100,2000}?)(?:\n\s*(?:KEYWORDS?|INDEX\s+TERMS?|INTRODUCTION|1\.?\s*INTRODUCTION|\d+\.?\s*INTRODUCTION))",
            # 섹션 번호가 있는 경우
            r"Abstract[:\-\s]*\n\s*(.*?)(?:\n\s*(?:1\.?\s*Introduction|2\.?\s*|Keywords?))",
            # 더 관대한 패턴 (길이 제한)
            r"Abstract[:\-\s]+(.*?)(?:\n\s*[A-Z][a-z]+\s*:|\n\s*\d+\.|\n\s*\n\s*[A-Z])",
        ]

        for i, pattern in enumerate(abstract_patterns):
            try:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    abstract = match.group(1)

                    # 텍스트 정제
                    abstract = re.sub(r"\s+", " ", abstract).strip()

                    # 너무 짧거나 긴 것 필터링
                    if 50 <= len(abstract) <= 3000:
                        # 의미없는 시작 부분 제거
                        abstract = re.sub(r"^[:\-\s]+", "", abstract)
                        return abstract

            except Exception as e:
                continue

        # 패턴 매칭이 실패한 경우, 다른 방법 시도
        return self.extract_abstract_fallback(text)

    def extract_abstract_fallback(self, text):
        """대안 Abstract 추출 방법"""
        lines = text.split("\n")

        # "Abstract" 키워드가 포함된 줄 찾기
        abstract_start_idx = -1
        for i, line in enumerate(lines):
            if re.search(r"\babstract\b", line, re.IGNORECASE):
                abstract_start_idx = i
                break

        if abstract_start_idx == -1:
            return ""

        # Abstract 시작점부터 텍스트 수집
        abstract_lines = []
        for i in range(abstract_start_idx, min(abstract_start_idx + 20, len(lines))):
            line = lines[i].strip()

            # 종료 조건들
            if re.search(
                r"\b(keywords?|introduction|1\.?\s*introduction)\b", line, re.IGNORECASE
            ):
                break

            # Abstract 라인 자체는 스킵
            if re.search(r"^\s*abstract\s*[:\-]?\s*$", line, re.IGNORECASE):
                continue

            if line and len(line) > 10:
                abstract_lines.append(line)

        if abstract_lines:
            abstract = " ".join(abstract_lines)
            abstract = re.sub(r"\s+", " ", abstract).strip()

            if 50 <= len(abstract) <= 3000:
                return abstract

        return ""

    def extract_title_from_text(self, text):
        """PDF 텍스트에서 제목 추출"""
        lines = text.split("\n")

        # 첫 번째 페이지의 처음 몇 줄에서 제목 찾기
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            # 제목은 보통 길고, 대문자가 많으며, 특정 패턴을 피함
            if (
                len(line) > 20
                and not re.match(r"^\d+", line)  # 숫자로 시작하지 않음
                and not line.lower().startswith(
                    ("abstract", "introduction", "keywords")
                )
                and len(line.split()) > 3
            ):
                return line

        return ""

    def extract_technical_keywords(self, text):
        """기술 키워드 추출 (사전 기반)"""
        text_lower = text.lower()
        found_keywords = []

        for keyword in self.tech_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)

        return found_keywords

    def extract_frequent_terms(self, text, min_freq=2, max_keywords=20):
        """빈도 기반 키워드 추출"""
        # 텍스트 전처리
        text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = word_tokenize(text)

        # 불용어 제거 및 필터링
        filtered_tokens = [
            token
            for token in tokens
            if (
                token not in self.stop_words
                and len(token) > 2
                and token.isalpha()
                and not token.isdigit()
            )
        ]

        # n-gram 추출 (1-gram, 2-gram)
        unigrams = filtered_tokens
        bigrams = [
            f"{filtered_tokens[i]} {filtered_tokens[i+1]}"
            for i in range(len(filtered_tokens) - 1)
        ]

        # 빈도 계산
        term_freq = Counter(unigrams + bigrams)

        # 최소 빈도 이상의 키워드만 선택
        frequent_terms = [term for term, freq in term_freq.items() if freq >= min_freq]

        # 빈도순으로 정렬하여 상위 키워드 반환
        top_terms = term_freq.most_common(max_keywords)
        return [term for term, freq in top_terms if freq >= min_freq]

    def extract_keywords_from_pdf(self, pdf_path):
        """PDF에서 종합적으로 키워드 추출"""
        # PDF 텍스트 추출
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return []

        all_keywords = []

        # 1. Keywords 섹션에서 추출
        section_keywords = self.find_keywords_section(text)
        all_keywords.extend(section_keywords)

        # 2. 기술 키워드 사전 기반 추출
        tech_keywords = self.extract_technical_keywords(text)
        all_keywords.extend(tech_keywords)

        # 3. 초록에서 빈도 기반 키워드 추출
        abstract = self.extract_abstract(text)
        if abstract:
            abstract_keywords = self.extract_frequent_terms(
                abstract, min_freq=1, max_keywords=10
            )
            all_keywords.extend(abstract_keywords)

        # 4. 전체 텍스트에서 빈도 기반 키워드 추출
        frequent_keywords = self.extract_frequent_terms(
            text, min_freq=2, max_keywords=15
        )
        all_keywords.extend(frequent_keywords)

        # 중복 제거 및 정제
        unique_keywords = []
        seen = set()

        for kw in all_keywords:
            kw_clean = kw.lower().strip()
            if kw_clean and len(kw_clean) > 2 and kw_clean not in seen:
                unique_keywords.append(kw_clean)
                seen.add(kw_clean)

        return unique_keywords

    def process_all_pdfs(self, pdf_dir, papers_metadata):
        """모든 PDF 파일에서 키워드 추출하여 메타데이터 업데이트"""
        print("🔍 Extracting keywords from PDF files...")

        updated_papers = []
        successful_extractions = 0
        failed_extractions = 0

        for paper in tqdm(papers_metadata, desc="Processing PDFs", ncols=80):
            paper_copy = paper.copy()

            if paper["has_pdf"] and paper["pdf_file"]:
                pdf_path = Path(paper["pdf_file"])

                if pdf_path.exists():
                    try:
                        # PDF에서 키워드 추출
                        pdf_keywords = self.extract_keywords_from_pdf(pdf_path)

                        if pdf_keywords:  # 키워드가 성공적으로 추출된 경우
                            successful_extractions += 1

                        # 기존 키워드와 병합
                        existing_keywords = paper.get("keywords", [])
                        combined_keywords = list(set(existing_keywords + pdf_keywords))

                        paper_copy["keywords"] = combined_keywords
                        paper_copy["pdf_keywords"] = pdf_keywords
                        paper_copy["keyword_source"] = "pdf_extracted"

                    except Exception as e:
                        # PDF 처리 실패시 조용히 처리
                        failed_extractions += 1
                        paper_copy["pdf_keywords"] = []
                        paper_copy["keyword_source"] = "pdf_failed"
                else:
                    paper_copy["pdf_keywords"] = []
                    paper_copy["keyword_source"] = "pdf_not_found"
            else:
                paper_copy["pdf_keywords"] = []
                paper_copy["keyword_source"] = "no_pdf"

            updated_papers.append(paper_copy)

        print(f"✅ PDF processing completed:")
        print(f"   📄 Total papers: {len(updated_papers)}")
        print(f"   ✅ Successful extractions: {successful_extractions}")
        if failed_extractions > 0:
            print(f"   ⚠️  Failed extractions: {failed_extractions}")

        return updated_papers

    def save_updated_metadata(self, papers_metadata, output_dir):
        """업데이트된 메타데이터 저장"""
        output_dir = Path(output_dir)

        # JSON 저장
        json_file = output_dir / "papers_metadata_with_pdf_keywords.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(papers_metadata, f, ensure_ascii=False, indent=2)

        print(f"💾 Updated metadata saved to {json_file}")

        # 키워드 통계 생성
        all_keywords = []
        for paper in papers_metadata:
            all_keywords.extend(paper.get("keywords", []))

        keyword_stats = Counter(all_keywords)

        # 키워드 통계 저장
        stats_file = output_dir / "keyword_statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_unique_keywords": len(keyword_stats),
                    "top_50_keywords": keyword_stats.most_common(50),
                    "keyword_frequencies": dict(keyword_stats),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"📊 Keyword statistics saved to {stats_file}")

        return json_file

    def debug_abstract_extraction(self, pdf_path, save_debug=False):
        """Abstract 추출 디버깅용 함수"""
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "No text extracted from PDF"}

        # 처음 1500자 정도만 확인 (Abstract가 보통 첫 페이지에 있음)
        first_part = text[:1500]

        # Abstract 관련 키워드 찾기
        abstract_mentions = []
        for match in re.finditer(r"abstract", text, re.IGNORECASE):
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 200)
            context = text[start:end].replace("\n", " ").strip()
            abstract_mentions.append({"position": match.start(), "context": context})

        # 실제 추출 결과
        extracted_abstract = self.extract_abstract(text)

        debug_info = {
            "pdf_path": str(pdf_path),
            "text_length": len(text),
            "first_1500_chars": first_part,
            "abstract_mentions": abstract_mentions,
            "extracted_abstract": extracted_abstract,
            "extraction_success": bool(extracted_abstract),
        }

        if save_debug:
            debug_file = Path(pdf_path).parent / f"debug_{Path(pdf_path).stem}.json"
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(debug_info, f, ensure_ascii=False, indent=2)
            print(f"Debug info saved to: {debug_file}")

        return debug_info


def main():
    """메인 실행 함수"""
    from src import PROCESSED_DIR

    # 기존 메타데이터 로드
    metadata_file = PROCESSED_DIR / "papers_metadata.json"

    if not metadata_file.exists():
        print("❌ Papers metadata not found. Run bibtex_parser.py first.")
        return

    with open(metadata_file, "r", encoding="utf-8") as f:
        papers_metadata = json.load(f)

    print(f"📄 Loaded {len(papers_metadata)} papers metadata")

    # PDF 키워드 추출기 초기화
    extractor = PDFKeywordExtractor()

    # 모든 PDF에서 키워드 추출
    updated_papers = extractor.process_all_pdfs(None, papers_metadata)

    # 업데이트된 메타데이터 저장
    extractor.save_updated_metadata(updated_papers, PROCESSED_DIR)

    # 통계 출력
    papers_with_keywords = sum(
        1 for p in updated_papers if len(p.get("keywords", [])) > 0
    )
    total_keywords = sum(len(p.get("keywords", [])) for p in updated_papers)

    print(f"\n📊 Updated Statistics:")
    print(f"   Papers with keywords: {papers_with_keywords}/{len(updated_papers)}")
    print(f"   Total keywords extracted: {total_keywords}")
    print(f"   Average keywords per paper: {total_keywords/len(updated_papers):.1f}")


if __name__ == "__main__":
    main()
