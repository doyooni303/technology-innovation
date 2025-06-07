# check_source_graphs_abstract.py - 소스 그래프들에서 abstract 확인

import json
from pathlib import Path
import pandas as pd


def check_all_source_graphs():
    """모든 소스 그래프에서 abstract 정보 확인"""

    print("🔍 모든 소스 그래프에서 Abstract 확인")
    print("=" * 60)

    graphs_dir = Path("data/processed/graphs")

    if not graphs_dir.exists():
        print(f"❌ 그래프 디렉토리가 없습니다: {graphs_dir}")
        return

    # 개별 그래프 파일들 찾기
    graph_files = list(graphs_dir.glob("*.json"))
    print(f"📂 발견된 그래프 파일: {len(graph_files)}개")

    abstract_summary = {}

    for graph_file in graph_files:
        graph_name = graph_file.stem
        print(f"\n📄 {graph_name}")
        print("-" * 40)

        try:
            with open(graph_file, "r", encoding="utf-8") as f:
                graph_data = json.load(f)

            nodes = graph_data.get("nodes", [])
            print(f"   총 노드: {len(nodes)}개")

            # 논문 노드만 필터링
            paper_nodes = [node for node in nodes if node.get("node_type") == "paper"]

            print(f"   논문 노드: {len(paper_nodes)}개")

            if not paper_nodes:
                print("   ⚠️ 논문 노드가 없습니다")
                continue

            # Abstract 통계
            abstract_fields_found = set()
            papers_with_abstract = 0
            papers_with_content = 0

            # 처음 5개 논문 상세 분석
            for i, paper in enumerate(paper_nodes[:5]):
                paper_id = paper.get("id", f"paper_{i}")
                title = paper.get("title", "No title")[:50]

                print(f"\n   📋 {paper_id}: {title}...")

                # 모든 필드 확인
                for key, value in paper.items():
                    if key.lower() in ["abstract", "description", "summary", "content"]:
                        abstract_fields_found.add(key)
                        if value and str(value).strip():
                            print(f"     ✅ {key}: {len(str(value))} 문자")
                            if key == "abstract":
                                papers_with_abstract += 1
                            papers_with_content += 1
                        else:
                            print(f"     ❌ {key}: 비어있음")

            # 전체 통계
            total_with_abstract = sum(
                1 for paper in paper_nodes if paper.get("abstract", "").strip()
            )

            total_with_any_content = sum(
                1
                for paper in paper_nodes
                if any(
                    paper.get(field, "").strip()
                    for field in ["abstract", "description", "summary", "content"]
                )
            )

            abstract_summary[graph_name] = {
                "total_papers": len(paper_nodes),
                "papers_with_abstract": total_with_abstract,
                "papers_with_any_content": total_with_any_content,
                "abstract_fields_found": list(abstract_fields_found),
                "abstract_percentage": (
                    total_with_abstract / len(paper_nodes) * 100 if paper_nodes else 0
                ),
            }

            print(f"\n   📊 전체 통계:")
            print(
                f"     Abstract 필드가 있는 논문: {total_with_abstract}/{len(paper_nodes)} ({total_with_abstract/len(paper_nodes)*100:.1f}%)"
            )
            print(
                f"     텍스트 내용이 있는 논문: {total_with_any_content}/{len(paper_nodes)}"
            )
            print(f"     발견된 텍스트 필드: {abstract_fields_found}")

        except Exception as e:
            print(f"   ❌ 오류: {e}")
            abstract_summary[graph_name] = {"error": str(e)}

    # 전체 요약
    print(f"\n📋 전체 요약")
    print("=" * 60)

    for graph_name, stats in abstract_summary.items():
        if "error" in stats:
            print(f"❌ {graph_name}: 오류")
        else:
            print(f"📄 {graph_name}:")
            print(f"   논문: {stats['total_papers']}개")
            print(
                f"   Abstract: {stats['papers_with_abstract']}개 ({stats['abstract_percentage']:.1f}%)"
            )
            print(f"   텍스트 필드: {stats['abstract_fields_found']}")

    return abstract_summary


def check_raw_metadata():
    """원본 메타데이터에서 abstract 확인"""

    print(f"\n🔍 원본 메타데이터에서 Abstract 확인")
    print("=" * 60)

    # 원본 메타데이터 파일들 확인
    metadata_files = [
        "data/processed/raw_extractions/papers_metadata.json",
        "data/processed/raw_extractions/integrated_papers_metadata.json",
        "data/processed/raw_extractions/papers_metadata.csv",
    ]

    for metadata_file in metadata_files:
        file_path = Path(metadata_file)

        if not file_path.exists():
            print(f"❌ {metadata_file}: 파일 없음")
            continue

        print(f"\n📄 {file_path.name}")
        print("-" * 40)

        try:
            if file_path.suffix == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    papers = data
                elif isinstance(data, dict):
                    papers = list(data.values())
                else:
                    print("   ⚠️ 알 수 없는 데이터 형태")
                    continue

            elif file_path.suffix == ".csv":
                df = pd.read_csv(file_path)
                papers = df.to_dict("records")

            print(f"   총 논문: {len(papers)}개")

            # Abstract 관련 필드 확인
            abstract_fields = set()
            papers_with_abstract = 0

            for paper in papers[:5]:  # 처음 5개만 상세 확인
                print(f"\n   📋 샘플: {paper.get('title', 'No title')[:50]}...")

                for key, value in paper.items():
                    if "abstract" in key.lower() or key.lower() in [
                        "description",
                        "summary",
                        "content",
                    ]:
                        abstract_fields.add(key)
                        if value and str(value).strip():
                            print(f"     ✅ {key}: {len(str(value))} 문자")
                        else:
                            print(f"     ❌ {key}: 비어있음")

            # 전체 통계
            for paper in papers:
                if any(
                    paper.get(field, "").strip()
                    for field in ["abstract", "description", "summary", "content"]
                    if field in paper
                ):
                    papers_with_abstract += 1

            print(f"\n   📊 통계:")
            print(f"     Abstract 관련 필드: {abstract_fields}")
            print(
                f"     텍스트 내용이 있는 논문: {papers_with_abstract}/{len(papers)} ({papers_with_abstract/len(papers)*100:.1f}%)"
            )

        except Exception as e:
            print(f"   ❌ 오류: {e}")


def recommend_next_steps():
    """다음 단계 추천"""

    print(f"\n💡 다음 단계 추천")
    print("=" * 60)

    print(f"1. 📊 원본 데이터 확인:")
    print(f"   - PDF에서 abstract 추출이 제대로 되었는지 확인")
    print(f"   - bibtex 파일에 abstract 정보가 있는지 확인")

    print(f"\n2. 🛠️ 그래프 구축 수정:")
    print(f"   - unified_graph_builder.py에 위의 수정사항 적용")
    print(f"   - 개별 그래프 빌더들도 abstract 포함하도록 수정")

    print(f"\n3. 🔄 재구축:")
    print(f"   - 그래프 재구축: python -m src.graphrag.unified_graph_builder")
    print(
        f"   - 임베딩 재생성: python -m src.graphrag.graphrag_pipeline build_embeddings --force-rebuild"
    )

    print(f"\n4. ✅ 검증:")
    print(f"   - python check_graph_abstract.py")
    print(f"   - python retrieval_debug_with_logging.py")


if __name__ == "__main__":
    # 1. 소스 그래프 확인
    summary = check_all_source_graphs()

    # 2. 원본 메타데이터 확인
    check_raw_metadata()

    # 3. 다음 단계 추천
    recommend_next_steps()
