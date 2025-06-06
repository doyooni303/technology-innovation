"""
PDF에서 References 추출 및 인용 관계 분석 모듈
Reference Extraction and Citation Analysis Module
"""

import re
import json
import warnings
from pathlib import Path
import pdfplumber
import PyPDF2
from collections import defaultdict
from tqdm import tqdm
from difflib import SequenceMatcher

# PDF 관련 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")


class ReferenceExtractor:
    """PDF에서 References를 추출하고 인용 관계를 분석하는 클래스"""

    def __init__(self):
        self.reference_patterns = [
            r"References?\s*\n",
            r"REFERENCES?\s*\n",
            r"Bibliography\s*\n",
            r"BIBLIOGRAPHY\s*\n",
            r"Literature Cited\s*\n",
            r"Works Cited\s*\n",
        ]

        # 년도 패턴 (2000-2025)
        self.year_pattern = r"\b(20[0-2][0-9])\b"

        # 저자명 패턴 (간단한 형태)
        self.author_pattern = r"^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*[A-Z][a-z]+)"

    def extract_full_text_from_pdf(self, pdf_path):
        """PDF 전체 텍스트 추출"""
        text = ""

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                        except:
                            continue

        except Exception:
            # pdfplumber 실패시 PyPDF2 사용
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(pdf_path, "rb") as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        for page in pdf_reader.pages:
                            try:
                                extracted_text = page.extract_text()
                                if extracted_text:
                                    text += extracted_text + "\n"
                            except:
                                continue
            except Exception:
                return ""

        return text

    def find_references_section(self, text):
        """텍스트에서 References 섹션 찾기"""
        references_start = -1

        # References 시작점 찾기
        for pattern in self.reference_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                references_start = match.end()
                break

        if references_start == -1:
            return ""

        # References 끝점 찾기 (Appendix, 다음 섹션 등)
        end_patterns = [
            r"\n\s*Appendix",
            r"\n\s*APPENDIX",
            r"\n\s*Author.*Biography",
            r"\n\s*AUTHOR.*BIOGRAPHY",
            r"\n\s*Acknowledgments?",
            r"\n\s*ACKNOWLEDGMENTS?",
            r"\n\s*Supplementary",
            r"\n\s*Index",
            r"\n\s*\d+\.\s*Conclusion",  # 다음 섹션 번호
        ]

        references_text = text[references_start:]
        references_end = len(references_text)

        for pattern in end_patterns:
            match = re.search(pattern, references_text, re.IGNORECASE)
            if match:
                references_end = min(references_end, match.start())

        return references_text[:references_end]

    def parse_references(self, references_text):
        """References 텍스트를 개별 참고문헌으로 파싱"""
        if not references_text.strip():
            return []

        references = []

        # 참고문헌 항목 분리 패턴들
        split_patterns = [
            r"\n\s*\[\d+\]",  # [1], [2] 형태
            r"\n\s*\d+\.\s",  # 1. 2. 형태
            r"\n\s*\(\d+\)",  # (1), (2) 형태
            r"\n(?=[A-Z][a-z]+.*?20[0-2][0-9])",  # 저자명으로 시작하는 줄
        ]

        # 가장 적합한 패턴 찾기
        best_pattern = None
        max_splits = 0

        for pattern in split_patterns:
            splits = re.split(pattern, references_text)
            if len(splits) > max_splits:
                max_splits = len(splits)
                best_pattern = pattern

        if best_pattern and max_splits > 1:
            ref_items = re.split(best_pattern, references_text)
        else:
            # 패턴이 없으면 빈 줄로 분리
            ref_items = re.split(r"\n\s*\n", references_text)

        # 각 참고문헌 정제
        for item in ref_items:
            item = item.strip()
            if len(item) > 20:  # 너무 짧은 것은 제외
                # 여러 줄을 한 줄로 합치기
                item = re.sub(r"\s+", " ", item)
                references.append(item)

        return references

    def extract_reference_info(self, reference_text):
        """개별 참고문헌에서 정보 추출"""
        info = {
            "raw_text": reference_text,
            "authors": [],
            "title": "",
            "year": "",
            "journal": "",
            "confidence": 0.0,
        }

        # 년도 추출
        year_matches = re.findall(self.year_pattern, reference_text)
        if year_matches:
            info["year"] = year_matches[0]  # 첫 번째 년도
            info["confidence"] += 0.3

        # 제목 추출 (큰따옴표 안의 텍스트)
        title_matches = re.findall(r'"([^"]+)"', reference_text)
        if title_matches:
            info["title"] = title_matches[0].strip()
            info["confidence"] += 0.4
        else:
            # 큰따옴표가 없으면 다른 패턴 시도
            # 저자명 다음부터 년도/저널명 전까지
            title_pattern = (
                r"[A-Z][a-z]+.*?\.\s*([^.]+?)\s*(?:"
                + self.year_pattern
                + r"|In\s|Proceedings|Journal)"
            )
            title_match = re.search(title_pattern, reference_text)
            if title_match:
                potential_title = title_match.group(1).strip()
                if len(potential_title) > 10:
                    info["title"] = potential_title
                    info["confidence"] += 0.2

        # 저자 추출 (간단한 방법)
        # 첫 번째 대문자로 시작하는 단어들
        author_pattern = r"^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*[A-Z][a-z]+)"
        author_match = re.match(author_pattern, reference_text)
        if author_match:
            info["authors"] = [author_match.group(1).strip()]
            info["confidence"] += 0.3

        return info

    def match_reference_to_papers(self, reference_info, paper_titles):
        """참고문헌을 논문 데이터셋의 논문들과 매칭"""
        if not reference_info["title"]:
            return None, 0.0

        ref_title = reference_info["title"].lower().strip()
        if len(ref_title) < 10:  # 너무 짧은 제목은 제외
            return None, 0.0

        best_match = None
        best_similarity = 0.0

        for paper_id, paper_title in paper_titles.items():
            paper_title_clean = paper_title.lower().strip()

            # 유사도 계산
            similarity = SequenceMatcher(None, ref_title, paper_title_clean).ratio()

            # 정확한 매치 우선
            if ref_title in paper_title_clean or paper_title_clean in ref_title:
                similarity += 0.3

            # 년도가 일치하면 보너스
            if reference_info["year"] and reference_info["year"] in paper_title:
                similarity += 0.1

            if similarity > best_similarity and similarity > 0.7:  # 임계값
                best_similarity = similarity
                best_match = paper_id

        return best_match, best_similarity

    def extract_citations_for_paper(self, paper_id, pdf_path, all_paper_titles):
        """개별 논문의 인용 관계 추출"""
        text = self.extract_full_text_from_pdf(pdf_path)
        if not text:
            return []

        # References 섹션 추출
        references_text = self.find_references_section(text)
        if not references_text:
            return []

        # 참고문헌 파싱
        references = self.parse_references(references_text)

        citations = []
        for ref_text in references:
            ref_info = self.extract_reference_info(ref_text)

            # 논문 데이터셋과 매칭
            matched_paper, similarity = self.match_reference_to_papers(
                ref_info, all_paper_titles
            )

            if matched_paper and similarity > 0.7:
                citations.append(
                    {
                        "cited_paper_id": matched_paper,
                        "similarity": similarity,
                        "reference_text": (
                            ref_text[:200] + "..." if len(ref_text) > 200 else ref_text
                        ),
                        "extracted_title": ref_info["title"],
                        "extracted_year": ref_info["year"],
                    }
                )

        return citations

    def build_citation_network(self, papers_metadata):
        """전체 논문 데이터셋의 인용 네트워크 구축"""
        print("🔗 Building citation network from PDF references...")

        # 논문 제목 딕셔너리 생성 (매칭용)
        paper_titles = {}
        pdf_papers = {}

        for i, paper in enumerate(papers_metadata):
            paper_id = f"paper_{i}"
            paper_titles[paper_id] = paper["title"]

            if paper["has_pdf"] and paper["pdf_file"]:
                pdf_papers[paper_id] = paper["pdf_file"]

        print(f"📄 Processing {len(pdf_papers)} papers with PDFs...")

        citation_network = {}
        total_citations = 0

        # 각 논문의 인용 관계 추출
        for paper_id, pdf_path in tqdm(pdf_papers.items(), desc="Extracting citations"):
            try:
                citations = self.extract_citations_for_paper(
                    paper_id, pdf_path, paper_titles
                )
                citation_network[paper_id] = citations
                total_citations += len(citations)

            except Exception as e:
                citation_network[paper_id] = []
                continue

        # 통계 정보
        papers_with_citations = sum(1 for cites in citation_network.values() if cites)

        print(f"✅ Citation network extraction completed:")
        print(f"   📄 Papers processed: {len(citation_network)}")
        print(f"   🔗 Papers with citations: {papers_with_citations}")
        print(f"   📊 Total citations found: {total_citations}")
        print(
            f"   📈 Average citations per paper: {total_citations/len(citation_network):.1f}"
        )

        return citation_network

    def save_citation_network(self, citation_network, output_dir):
        """인용 네트워크를 파일로 저장"""
        output_dir = Path(output_dir)

        # 상세 정보 저장
        detailed_file = output_dir / "citation_network_detailed.json"
        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(citation_network, f, ensure_ascii=False, indent=2)

        # 간단한 형태로 변환 (그래프 구축용)
        simple_network = {}
        for citing_paper, citations in citation_network.items():
            simple_network[citing_paper] = [
                {
                    "cited_paper": cite["cited_paper_id"],
                    "similarity": cite["similarity"],
                }
                for cite in citations
            ]

        simple_file = output_dir / "citation_network_simple.json"
        with open(simple_file, "w", encoding="utf-8") as f:
            json.dump(simple_network, f, ensure_ascii=False, indent=2)

        print(f"💾 Citation network saved:")
        print(f"   📄 Detailed: {detailed_file}")
        print(f"   📊 Simple: {simple_file}")

        return simple_file


def main():
    """메인 실행 함수"""
    from src import PROCESSED_DIR

    # 통합 메타데이터 로드
    metadata_file = PROCESSED_DIR / "integrated_papers_metadata.json"

    if not metadata_file.exists():
        print("❌ Integrated papers metadata not found. Run main.py first.")
        return

    with open(metadata_file, "r", encoding="utf-8") as f:
        papers_metadata = json.load(f)

    print(f"📄 Loaded {len(papers_metadata)} papers metadata")

    # Reference 추출기 초기화
    extractor = ReferenceExtractor()

    # 인용 네트워크 구축
    citation_network = extractor.build_citation_network(papers_metadata)

    # 결과 저장
    output_file = extractor.save_citation_network(citation_network, PROCESSED_DIR)

    return citation_network, output_file


if __name__ == "__main__":
    main()
