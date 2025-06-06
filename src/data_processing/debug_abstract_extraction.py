"""
Abstract 추출 디버깅 및 테스트 스크립트
"""

import json
from pathlib import Path
from .pdf_keyword_extractor import PDFKeywordExtractor


def test_abstract_extraction(num_samples=5):
    """몇 개 샘플 PDF에서 Abstract 추출 테스트"""
    from src import PDFS_DIR, PROCESSED_DIR

    # PDF 파일들 가져오기
    pdf_files = list(PDFS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found")
        return

    # 처음 몇 개 파일만 테스트
    sample_files = pdf_files[:num_samples]

    extractor = PDFKeywordExtractor()

    print(f"🔍 Testing abstract extraction on {len(sample_files)} samples...")

    results = []
    for i, pdf_file in enumerate(sample_files):
        print(f"\n📄 Testing {i+1}/{len(sample_files)}: {pdf_file.name}")

        # 디버그 정보 추출
        debug_info = extractor.debug_abstract_extraction(pdf_file, save_debug=True)

        print(f"   📏 Text length: {debug_info['text_length']} characters")
        print(f"   🔍 Abstract mentions found: {len(debug_info['abstract_mentions'])}")
        print(f"   ✅ Extraction success: {debug_info['extraction_success']}")

        if debug_info["extracted_abstract"]:
            abstract_preview = debug_info["extracted_abstract"][:100] + "..."
            print(f"   📝 Abstract preview: {abstract_preview}")
        else:
            print("   ❌ No abstract extracted")

        # Abstract 언급 컨텍스트 출력
        if debug_info["abstract_mentions"]:
            print("   📍 Abstract mentions context:")
            for j, mention in enumerate(
                debug_info["abstract_mentions"][:2]
            ):  # 처음 2개만
                context_preview = mention["context"][:150] + "..."
                print(f"      {j+1}. {context_preview}")

        results.append(
            {
                "file": pdf_file.name,
                "success": debug_info["extraction_success"],
                "abstract_length": (
                    len(debug_info["extracted_abstract"])
                    if debug_info["extracted_abstract"]
                    else 0
                ),
                "mentions_found": len(debug_info["abstract_mentions"]),
            }
        )

    # 전체 결과 요약
    print(f"\n📊 Test Results Summary:")
    successful = sum(1 for r in results if r["success"])
    print(f"   ✅ Successful extractions: {successful}/{len(results)}")
    print(f"   📈 Success rate: {successful/len(results)*100:.1f}%")

    if successful > 0:
        avg_length = (
            sum(r["abstract_length"] for r in results if r["success"]) / successful
        )
        print(f"   📏 Average abstract length: {avg_length:.0f} characters")

    return results


def inspect_specific_pdf(pdf_name):
    """특정 PDF 파일 상세 분석"""
    from src import PDFS_DIR

    pdf_path = PDFS_DIR / pdf_name
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_name}")
        return

    extractor = PDFKeywordExtractor()

    print(f"🔍 Detailed inspection of: {pdf_name}")

    # 상세 디버그 정보
    debug_info = extractor.debug_abstract_extraction(pdf_path, save_debug=True)

    print(f"\n📄 PDF Information:")
    print(f"   📏 Total text length: {debug_info['text_length']} characters")
    print(f"   🔍 Abstract mentions: {len(debug_info['abstract_mentions'])}")

    print(f"\n📝 First 1500 characters of PDF:")
    print("=" * 50)
    print(debug_info["first_1500_chars"])
    print("=" * 50)

    if debug_info["abstract_mentions"]:
        print(f"\n📍 All Abstract mention contexts:")
        for i, mention in enumerate(debug_info["abstract_mentions"]):
            print(f"\n{i+1}. Position {mention['position']}:")
            print(f"   Context: {mention['context']}")

    if debug_info["extracted_abstract"]:
        print(f"\n✅ Extracted Abstract:")
        print(f"   Length: {len(debug_info['extracted_abstract'])} characters")
        print(f"   Content: {debug_info['extracted_abstract']}")
    else:
        print(f"\n❌ No abstract extracted")

    return debug_info


def test_abstract_patterns():
    """다양한 Abstract 패턴 테스트"""

    # 테스트용 샘플 텍스트들
    sample_texts = [
        # 패턴 1: 기본 형태
        """
        Title: Machine Learning for Battery Optimization
        
        Abstract
        This paper presents a novel approach to battery optimization using machine learning techniques. We propose a new algorithm that significantly improves battery performance.
        
        Keywords: machine learning, battery, optimization
        
        1. Introduction
        """,
        # 패턴 2: 콜론 있는 형태
        """
        Abstract: This study investigates the application of artificial intelligence in electric vehicle battery management systems. Our results show promising improvements in efficiency.
        
        Index Terms: AI, electric vehicle, battery management
        """,
        # 패턴 3: 대문자 형태
        """
        ABSTRACT
        
        Recent advances in deep learning have opened new possibilities for state-of-charge estimation in lithium-ion batteries. This work explores these applications.
        
        KEYWORDS: deep learning, battery, state-of-charge
        """,
        # 패턴 4: 복잡한 형태
        """
        Paper Title
        
        Abstract— Electric vehicles require accurate battery management for optimal performance. This paper introduces a reinforcement learning approach for battery control systems.
        
        I. INTRODUCTION
        """,
    ]

    extractor = PDFKeywordExtractor()

    print("🧪 Testing Abstract extraction patterns...")

    for i, text in enumerate(sample_texts):
        print(f"\n📝 Test {i+1}:")
        print("Input text:")
        print(text.strip())

        extracted = extractor.extract_abstract(text)
        print(f"\n✅ Extracted: {extracted}")
        print(f"📏 Length: {len(extracted)} characters")
        print("-" * 60)


def main():
    """메인 테스트 함수"""
    import sys

    if len(sys.argv) > 1:
        # 특정 PDF 파일 분석
        pdf_name = sys.argv[1]
        inspect_specific_pdf(pdf_name)
    else:
        # 패턴 테스트
        print("🧪 Testing Abstract extraction patterns...")
        test_abstract_patterns()

        print("\n" + "=" * 60)

        # 샘플 PDF 테스트
        print("📄 Testing on sample PDFs...")
        test_abstract_extraction(num_samples=3)

        print("\n💡 To test a specific PDF:")
        print(
            "   python -m src.data_processing.debug_abstract_extraction <filename.pdf>"
        )


if __name__ == "__main__":
    main()
