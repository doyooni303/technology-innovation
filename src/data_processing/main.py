"""
통합 데이터 처리 파이프라인
Integrated Data Processing Pipeline
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from . import BibtexParser, PDFKeywordExtractor


def merge_keywords(bibtex_keywords, pdf_keywords):
    """Bibtex와 PDF에서 추출한 키워드를 통합"""
    # 모든 키워드를 소문자로 정규화
    all_keywords = []

    # Bibtex 키워드 추가
    for kw in bibtex_keywords:
        if kw and isinstance(kw, str):
            all_keywords.append(kw.lower().strip())

    # PDF 키워드 추가
    for kw in pdf_keywords:
        if kw and isinstance(kw, str):
            all_keywords.append(kw.lower().strip())

    # 중복 제거 및 빈 문자열 제거
    unique_keywords = list(set([kw for kw in all_keywords if kw and len(kw) > 2]))

    return sorted(unique_keywords)


def create_integrated_metadata(papers_bibtex, pdf_keywords_data, pdf_abstracts_data):
    """Bibtex, PDF 키워드, PDF Abstract를 통합한 완전한 메타데이터 생성"""
    integrated_papers = []

    print("🔄 Integrating bibtex, PDF keywords, and PDF abstracts...")

    for paper in papers_bibtex:
        integrated_paper = paper.copy()

        # 원본 키워드 보존
        bibtex_keywords = paper.get("keywords", [])
        integrated_paper["bibtex_keywords"] = bibtex_keywords

        # PDF 키워드 찾기 (제목 기반 매칭)
        pdf_keywords = []
        pdf_abstract = ""

        for pdf_data in pdf_keywords_data:
            if (
                pdf_data.get("title", "").lower().strip()
                == paper.get("title", "").lower().strip()
            ):
                pdf_keywords = pdf_data.get("pdf_keywords", [])
                break

        # PDF Abstract 찾기 (제목 기반 매칭)
        for abstract_data in pdf_abstracts_data:
            if (
                abstract_data.get("title", "").lower().strip()
                == paper.get("title", "").lower().strip()
            ):
                pdf_abstract = abstract_data.get("abstract", "")
                break

        integrated_paper["pdf_keywords"] = pdf_keywords
        integrated_paper["abstract"] = pdf_abstract  # ⭐ Abstract 추가

        # 통합 키워드 생성
        merged_keywords = merge_keywords(bibtex_keywords, pdf_keywords)
        integrated_paper["keywords"] = merged_keywords

        # 키워드 소스 정보
        keyword_sources = []
        if bibtex_keywords:
            keyword_sources.append("bibtex")
        if pdf_keywords:
            keyword_sources.append("pdf")
        if not keyword_sources:
            keyword_sources.append("none")

        integrated_paper["keyword_sources"] = keyword_sources
        integrated_paper["total_keyword_count"] = len(merged_keywords)
        integrated_paper["bibtex_keyword_count"] = len(bibtex_keywords)
        integrated_paper["pdf_keyword_count"] = len(pdf_keywords)
        integrated_paper["has_abstract"] = bool(pdf_abstract)  # Abstract 유무

        integrated_papers.append(integrated_paper)

    return integrated_papers


def run_complete_data_processing():
    """전체 데이터 처리 파이프라인 실행"""
    print("🚀 Starting integrated data processing pipeline...")

    # 프로젝트 경로 설정
    from src import BIBS_DIR, PDFS_DIR, PROCESSED_DIR

    # Step 1: Bibtex 파싱
    print("\n" + "=" * 60)
    print("STEP 1: Parsing Bibtex Files")
    print("=" * 60)

    parser = BibtexParser(BIBS_DIR, PDFS_DIR)
    papers_bibtex = parser.parse_all_bibtex_files()

    # Bibtex 기반 중간 결과 저장
    parser.save_metadata(PROCESSED_DIR / "step1_bibtex_metadata.json")

    # Bibtex 기반 통계
    stats_bibtex = parser.get_statistics()
    print("\n📊 Bibtex-only Statistics:")
    for key, value in stats_bibtex.items():
        print(f"   {key}: {value}")

    # Step 2: PDF에서 키워드 및 Abstract 추출
    print("\n" + "=" * 60)
    print("STEP 2: Extracting Keywords and Abstracts from PDFs")
    print("=" * 60)

    extractor = PDFKeywordExtractor()

    # PDF 키워드 개별 추출 (bibtex 데이터와 별도)
    pdf_papers_with_keywords = []
    pdf_papers_with_abstracts = []

    for paper in papers_bibtex:
        if paper["has_pdf"] and paper["pdf_file"]:
            pdf_path = Path(paper["pdf_file"])
            if pdf_path.exists():
                try:
                    # 키워드 추출
                    pdf_keywords = extractor.extract_keywords_from_pdf(pdf_path)
                    pdf_papers_with_keywords.append(
                        {
                            "title": paper["title"],
                            "pdf_file": paper["pdf_file"],
                            "pdf_keywords": pdf_keywords,
                        }
                    )

                    # Abstract 추출 ⭐
                    pdf_text = extractor.extract_text_from_pdf(pdf_path)
                    abstract = extractor.extract_abstract(pdf_text) if pdf_text else ""
                    pdf_papers_with_abstracts.append(
                        {
                            "title": paper["title"],
                            "pdf_file": paper["pdf_file"],
                            "abstract": abstract,
                        }
                    )

                except Exception as e:
                    # 실패시 빈 데이터
                    pdf_papers_with_keywords.append(
                        {
                            "title": paper["title"],
                            "pdf_file": paper["pdf_file"],
                            "pdf_keywords": [],
                        }
                    )
                    pdf_papers_with_abstracts.append(
                        {
                            "title": paper["title"],
                            "pdf_file": paper["pdf_file"],
                            "abstract": "",
                        }
                    )

    print(f"📄 Extracted keywords from {len(pdf_papers_with_keywords)} PDF files")
    print(f"📝 Extracted abstracts from {len(pdf_papers_with_abstracts)} PDF files")

    # Abstract 품질 확인
    abstracts_with_content = sum(
        1 for item in pdf_papers_with_abstracts if len(item["abstract"]) > 50
    )
    print(f"📊 Papers with meaningful abstracts: {abstracts_with_content}")

    # Step 3: 데이터 통합
    print("\n" + "=" * 60)
    print("STEP 3: Integrating Bibtex + PDF Keywords + PDF Abstracts")
    print("=" * 60)

    integrated_papers = create_integrated_metadata(
        papers_bibtex, pdf_papers_with_keywords, pdf_papers_with_abstracts
    )

    # Step 4: 통합 결과 저장
    print("\n" + "=" * 60)
    print("STEP 4: Saving Integrated Results (Keywords + Abstracts)")
    print("=" * 60)

    # 최종 통합 메타데이터 저장
    final_file = PROCESSED_DIR / "integrated_papers_metadata.json"
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(integrated_papers, f, ensure_ascii=False, indent=2)

    # CSV 버전도 저장
    import pandas as pd

    # DataFrame 생성을 위해 리스트 컬럼들을 문자열로 변환
    df_data = []
    for paper in integrated_papers:
        row = paper.copy()
        row["keywords"] = "; ".join(paper.get("keywords", []))
        row["bibtex_keywords"] = "; ".join(paper.get("bibtex_keywords", []))
        row["pdf_keywords"] = "; ".join(paper.get("pdf_keywords", []))
        row["authors"] = "; ".join(paper.get("authors", []))
        row["institutions"] = "; ".join(paper.get("institutions", []))
        row["keyword_sources"] = "; ".join(paper.get("keyword_sources", []))
        # Abstract는 이미 문자열이므로 그대로 유지
        df_data.append(row)

    df = pd.DataFrame(df_data)
    csv_file = PROCESSED_DIR / "integrated_papers_metadata.csv"
    df.to_csv(csv_file, index=False, encoding="utf-8")

    print(f"💾 Integrated metadata saved to:")
    print(f"   📄 JSON: {final_file}")
    print(f"   📊 CSV: {csv_file}")

    # Step 5: 종합 통계 및 키워드 분석
    print("\n" + "=" * 60)
    print("FINAL INTEGRATED STATISTICS")
    print("=" * 60)

    # 키워드 소스별 통계
    papers_with_bibtex_only = sum(
        1
        for p in integrated_papers
        if "bibtex" in p["keyword_sources"] and "pdf" not in p["keyword_sources"]
    )
    papers_with_pdf_only = sum(
        1
        for p in integrated_papers
        if "pdf" in p["keyword_sources"] and "bibtex" not in p["keyword_sources"]
    )
    papers_with_both = sum(
        1
        for p in integrated_papers
        if "bibtex" in p["keyword_sources"] and "pdf" in p["keyword_sources"]
    )
    papers_with_none = sum(
        1 for p in integrated_papers if "none" in p["keyword_sources"]
    )

    total_integrated_keywords = sum(
        len(p.get("keywords", [])) for p in integrated_papers
    )
    total_bibtex_keywords = sum(
        p.get("bibtex_keyword_count", 0) for p in integrated_papers
    )
    total_pdf_keywords = sum(p.get("pdf_keyword_count", 0) for p in integrated_papers)

    # 모든 유니크 키워드 수집
    all_integrated_keywords = set()
    for paper in integrated_papers:
        all_integrated_keywords.update(paper.get("keywords", []))

    # Abstract 통계 계산
    abstracts_with_content = [
        p for p in integrated_papers if len(p.get("abstract", "")) > 50
    ]
    abstract_lengths = [
        len(p.get("abstract", "")) for p in integrated_papers if p.get("abstract")
    ]
    avg_abstract_length = np.mean(abstract_lengths) if abstract_lengths else 0

    print(f"📄 Total papers processed: {len(integrated_papers)}")
    print(
        f"🔗 Papers with PDF files: {sum(1 for p in integrated_papers if p['has_pdf'])}"
    )
    print(
        f"📝 Papers with abstracts: {sum(1 for p in integrated_papers if p.get('has_abstract', False))}"
    )
    print("")
    print("📊 Keyword Source Distribution:")
    print(f"   📝 Bibtex keywords only: {papers_with_bibtex_only}")
    print(f"   📋 PDF keywords only: {papers_with_pdf_only}")
    print(f"   🔄 Both sources: {papers_with_both}")
    print(f"   ❌ No keywords: {papers_with_none}")
    print("")
    print("🔢 Content Counts:")
    print(f"   📝 Total bibtex keywords: {total_bibtex_keywords}")
    print(f"   📋 Total PDF keywords: {total_pdf_keywords}")
    print(f"   🔄 Total integrated keywords: {total_integrated_keywords}")
    print(f"   🔍 Unique integrated keywords: {len(all_integrated_keywords)}")
    print(
        f"   📈 Average keywords per paper: {total_integrated_keywords/len(integrated_papers):.1f}"
    )

    # Abstract 통계 추가
    abstracts_with_content = [
        p for p in integrated_papers if len(p.get("abstract", "")) > 50
    ]
    avg_abstract_length = np.mean(
        [len(p.get("abstract", "")) for p in integrated_papers if p.get("abstract")]
    )
    print(f"   📄 Average abstract length: {avg_abstract_length:.0f} characters")

    # 키워드가 가장 많은 논문 top 5
    papers_by_keyword_count = sorted(
        integrated_papers, key=lambda x: len(x.get("keywords", [])), reverse=True
    )

    print(f"\n🏆 Top 5 papers by integrated keyword count:")
    for i, paper in enumerate(papers_by_keyword_count[:5]):
        keyword_count = len(paper.get("keywords", []))
        bibtex_count = paper.get("bibtex_keyword_count", 0)
        pdf_count = paper.get("pdf_keyword_count", 0)
        title = (
            paper["title"][:50] + "..." if len(paper["title"]) > 50 else paper["title"]
        )
        print(f"   {i+1}. {title}")
        print(
            f"      Total: {keyword_count} (Bibtex: {bibtex_count}, PDF: {pdf_count})"
        )

    # 가장 빈번한 키워드 top 15
    all_keywords_list = []
    for paper in integrated_papers:
        all_keywords_list.extend(paper.get("keywords", []))

    keyword_freq = Counter(all_keywords_list)
    print(f"\n🔥 Top 15 most frequent integrated keywords:")
    for i, (keyword, count) in enumerate(keyword_freq.most_common(15)):
        print(f"   {i+1:2d}. {keyword}: {count} papers")

    # 키워드 통계 저장
    keyword_stats = {
        "total_papers": len(integrated_papers),
        "keyword_source_distribution": {
            "bibtex_only": papers_with_bibtex_only,
            "pdf_only": papers_with_pdf_only,
            "both_sources": papers_with_both,
            "no_keywords": papers_with_none,
        },
        "keyword_counts": {
            "total_bibtex_keywords": total_bibtex_keywords,
            "total_pdf_keywords": total_pdf_keywords,
            "total_integrated_keywords": total_integrated_keywords,
            "unique_integrated_keywords": len(all_integrated_keywords),
        },
        "abstract_statistics": {
            "papers_with_abstracts": sum(
                1 for p in integrated_papers if p.get("has_abstract", False)
            ),
            "papers_with_meaningful_abstracts": len(abstracts_with_content),
            "average_abstract_length": avg_abstract_length,
        },
        "top_keywords": keyword_freq.most_common(50),
        "keyword_frequencies": dict(keyword_freq),
    }

    stats_file = PROCESSED_DIR / "integrated_keyword_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(keyword_stats, f, ensure_ascii=False, indent=2)

    print(f"\n📊 Keyword statistics saved to: {stats_file}")
    print(f"\n✅ Integrated data processing (with abstracts) completed successfully!")
    print(f"📁 Main output: {final_file}")

    return integrated_papers, final_file


def main():
    """메인 실행 함수"""
    try:
        papers, output_file = run_complete_data_processing()
        print(
            f"\n🎉 Success! Integrated {len(papers)} papers with keywords and abstracts."
        )
        print(f"📂 Output file: {output_file}")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
