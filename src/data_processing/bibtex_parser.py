"""
Bibtex 파일 파싱 및 메타데이터 추출 모듈
"""

import os
import pandas as pd
from pathlib import Path
from pybtex.database import parse_file
from pybtex.database.input import bibtex
from tqdm import tqdm
import json
import re


class BibtexParser:
    """Bibtex 파일들을 파싱하여 메타데이터를 추출하는 클래스"""

    def __init__(self, bibs_dir, pdfs_dir):
        """
        Args:
            bibs_dir (str/Path): bibtex 파일들이 있는 디렉토리
            pdfs_dir (str/Path): PDF 파일들이 있는 디렉토리
        """
        self.bibs_dir = Path(bibs_dir)
        self.pdfs_dir = Path(pdfs_dir)
        self.papers_metadata = []

    def clean_text(self, text):
        """텍스트 정제 함수"""
        if not text:
            return ""
        # 중괄호 제거
        text = re.sub(r"[{}]", "", str(text))
        # 연속된 공백 정리
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_keywords(self, entry):
        """키워드 추출 함수"""
        keywords = []

        # keywords 필드에서 추출
        if "keywords" in entry.fields:
            kw_text = self.clean_text(entry.fields["keywords"])
            # 쉼표, 세미콜론으로 분리
            keywords.extend(
                [kw.strip() for kw in re.split(r"[,;]", kw_text) if kw.strip()]
            )

        # 제목에서 주요 키워드 추출 (간단한 방법)
        title = self.clean_text(entry.fields.get("title", ""))
        title_keywords = self.extract_title_keywords(title)
        keywords.extend(title_keywords)

        # 중복 제거 및 정제
        keywords = list(set([kw.lower().strip() for kw in keywords if len(kw) > 2]))
        return keywords

    def extract_title_keywords(self, title):
        """제목에서 기술 키워드 추출"""
        tech_keywords = [
            "machine learning",
            "deep learning",
            "artificial intelligence",
            "AI",
            "reinforcement learning",
            "neural network",
            "battery",
            "lithium-ion",
            "electric vehicle",
            "EV",
            "charging",
            "energy management",
            "optimization",
            "IoT",
            "blockchain",
            "federated learning",
            "CNN",
            "LSTM",
            "RNN",
        ]

        found_keywords = []
        title_lower = title.lower()

        for keyword in tech_keywords:
            if keyword.lower() in title_lower:
                found_keywords.append(keyword)

        return found_keywords

    def extract_authors_and_institutions(self, entry):
        """저자와 소속기관 정보 추출"""
        authors = []
        institutions = set()

        if "author" in entry.persons:
            for person in entry.persons["author"]:
                # 저자명 추출
                first_names = " ".join(person.first_names)
                last_names = " ".join(person.last_names)
                full_name = f"{first_names} {last_names}".strip()
                authors.append(full_name)

        # 소속기관은 보통 별도 필드에 있지만, 간단히 처리
        if "institution" in entry.fields:
            inst_text = self.clean_text(entry.fields["institution"])
            institutions.add(inst_text)

        # author 필드에서 소속기관 정보 추출 시도
        if "author" in entry.fields:
            author_text = self.clean_text(entry.fields["author"])
            # 간단한 패턴 매칭으로 대학/기관명 추출
            inst_patterns = [
                r"University of ([^,\n]+)",
                r"([^,\n]*University[^,\n]*)",
                r"([^,\n]*Institute[^,\n]*)",
                r"([^,\n]*College[^,\n]*)",
            ]

            for pattern in inst_patterns:
                matches = re.findall(pattern, author_text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        institutions.update([m.strip() for m in match if m.strip()])
                    else:
                        institutions.add(match.strip())

        return authors, list(institutions)

    def parse_single_bibtex(self, bibtex_file):
        """개별 bibtex 파일 파싱"""
        try:
            # bibtex 파일 읽기
            with open(bibtex_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # 임시 파일로 저장 후 파싱 (pybtex가 파일 경로를 요구함)
            temp_file = bibtex_file.with_suffix(".bib")
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(content)

            # pybtex로 파싱
            bib_data = parse_file(str(temp_file))

            # 임시 파일 삭제
            temp_file.unlink()

            papers = []
            for key, entry in bib_data.entries.items():
                # 기본 메타데이터 추출
                title = self.clean_text(entry.fields.get("title", ""))
                year = entry.fields.get("year", "")
                journal = self.clean_text(entry.fields.get("journal", ""))
                publisher = self.clean_text(entry.fields.get("publisher", ""))

                # 저자 및 소속기관 추출
                authors, institutions = self.extract_authors_and_institutions(entry)

                # 키워드 추출
                keywords = self.extract_keywords(entry)

                # PDF 파일 매칭 확인
                pdf_file = self.find_matching_pdf(bibtex_file.stem)

                paper_data = {
                    "bibtex_key": key,
                    "title": title,
                    "authors": authors,
                    "institutions": institutions,
                    "year": year,
                    "journal": journal,
                    "publisher": publisher,
                    "keywords": keywords,
                    "bibtex_file": str(bibtex_file),
                    "pdf_file": str(pdf_file) if pdf_file else None,
                    "has_pdf": pdf_file is not None,
                }

                papers.append(paper_data)

            return papers

        except Exception as e:
            print(f"❌ Error parsing {bibtex_file}: {e}")
            return []

    def find_matching_pdf(self, bibtex_stem):
        """bibtex 파일에 대응하는 PDF 파일 찾기"""
        # 동일한 파일명으로 PDF 찾기
        pdf_file = self.pdfs_dir / f"{bibtex_stem}.pdf"

        if pdf_file.exists():
            return pdf_file

        # 대소문자 무시하고 찾기
        for pdf_file in self.pdfs_dir.glob("*.pdf"):
            if pdf_file.stem.lower() == bibtex_stem.lower():
                return pdf_file

        return None

    def parse_all_bibtex_files(self):
        """모든 bibtex 파일들을 파싱하여 메타데이터 추출"""
        print("🔍 Scanning bibtex files...")
        bibtex_files = list(self.bibs_dir.glob("*.txt"))

        print(f"📄 Found {len(bibtex_files)} bibtex files")

        all_papers = []
        failed_files = []

        for bibtex_file in tqdm(bibtex_files, desc="Parsing bibtex files"):
            papers = self.parse_single_bibtex(bibtex_file)
            if papers:
                all_papers.extend(papers)
            else:
                failed_files.append(bibtex_file)

        self.papers_metadata = all_papers

        print(f"✅ Successfully parsed {len(all_papers)} papers")
        if failed_files:
            print(f"❌ Failed to parse {len(failed_files)} files:")
            for f in failed_files[:5]:  # 처음 5개만 출력
                print(f"   - {f.name}")

        return all_papers

    def save_metadata(self, output_file):
        """메타데이터를 JSON/CSV 파일로 저장"""
        output_path = Path(output_file)

        if output_path.suffix.lower() == ".json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.papers_metadata, f, ensure_ascii=False, indent=2)
        elif output_path.suffix.lower() == ".csv":
            df = pd.DataFrame(self.papers_metadata)
            df.to_csv(output_path, index=False, encoding="utf-8")

        print(f"💾 Metadata saved to {output_path}")

    def get_statistics(self):
        """파싱된 데이터의 통계 정보 반환"""
        if not self.papers_metadata:
            return {}

        df = pd.DataFrame(self.papers_metadata)

        stats = {
            "total_papers": len(self.papers_metadata),
            "papers_with_pdf": df["has_pdf"].sum(),
            "papers_without_pdf": (~df["has_pdf"]).sum(),
            "papers_with_keywords": (df["keywords"].apply(len) > 0).sum(),
            "avg_authors_per_paper": df["authors"].apply(len).mean(),
            "total_unique_authors": len(
                set([author for authors in df["authors"] for author in authors])
            ),
            "total_unique_institutions": len(
                set(
                    [
                        inst
                        for institutions in df["institutions"]
                        for inst in institutions
                    ]
                )
            ),
            "total_unique_keywords": len(
                set([kw for keywords in df["keywords"] for kw in keywords])
            ),
        }

        return stats


def main():
    """메인 실행 함수"""
    # 프로젝트 경로 설정
    from src import BIBS_DIR, PDFS_DIR, PROCESSED_DIR

    # BibtexParser 인스턴스 생성
    parser = BibtexParser(BIBS_DIR, PDFS_DIR)

    # 모든 bibtex 파일 파싱
    papers = parser.parse_all_bibtex_files()

    # 메타데이터 저장
    parser.save_metadata(PROCESSED_DIR / "papers_metadata.json")
    parser.save_metadata(PROCESSED_DIR / "papers_metadata.csv")

    # 통계 정보 출력
    stats = parser.get_statistics()
    print("\n📊 Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    main()
