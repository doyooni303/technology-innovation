# check_graph_abstract.py - 그래프에서 abstract 데이터 확인

import json
from pathlib import Path


def check_abstract_in_graph():
    """통합 그래프에서 abstract 정보 확인"""

    print("🔍 통합 그래프에서 Abstract 정보 확인")
    print("=" * 60)

    graph_path = Path("data/processed/graphs/unified/unified_knowledge_graph.json")

    if not graph_path.exists():
        print(f"❌ 그래프 파일이 없습니다: {graph_path}")
        return

    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        print(f"✅ 그래프 로드 성공")
        print(f"   총 노드: {len(graph_data.get('nodes', []))}")
        print(f"   총 엣지: {len(graph_data.get('edges', []))}")

        # 논문 노드들만 필터링
        paper_nodes = [
            node
            for node in graph_data.get("nodes", [])
            if node.get("node_type") == "paper"
        ]

        print(f"\n📄 논문 노드 분석:")
        print(f"   총 논문 수: {len(paper_nodes)}")

        # Abstract 통계
        has_abstract_count = 0
        has_abstract_field_count = 0
        abstract_samples = []

        for node in paper_nodes[:10]:  # 처음 10개만 확인
            node_id = node.get("id", "unknown")
            title = node.get("title", "No title")
            has_abstract_field = node.get("has_abstract", False)
            abstract = node.get("abstract", "")

            print(f"\n📋 노드: {node_id}")
            print(f"   제목: {title[:50]}...")
            print(f"   has_abstract 필드: {has_abstract_field}")
            print(f"   abstract 필드 존재: {'abstract' in node}")

            if "abstract" in node:
                has_abstract_field_count += 1
                if abstract and abstract.strip():
                    has_abstract_count += 1
                    abstract_preview = abstract.replace("\n", " ").strip()[:100]
                    print(f"   abstract 내용: {abstract_preview}...")
                    abstract_samples.append((node_id, title, abstract))
                else:
                    print(f"   abstract 내용: (비어있음)")
            else:
                print(f"   abstract 필드: (없음)")

        print(f"\n📊 Abstract 통계:")
        print(f"   abstract 필드가 있는 노드: {has_abstract_field_count}/10")
        print(f"   실제 abstract 내용이 있는 노드: {has_abstract_count}/10")

        # 전체 통계
        total_with_abstract = sum(
            1 for node in paper_nodes if node.get("abstract", "").strip()
        )
        total_with_abstract_field = sum(1 for node in paper_nodes if "abstract" in node)

        print(f"\n🔢 전체 논문 Abstract 통계:")
        print(f"   전체 논문 수: {len(paper_nodes)}")
        print(f"   abstract 필드가 있는 논문: {total_with_abstract_field}")
        print(f"   실제 abstract 내용이 있는 논문: {total_with_abstract}")
        print(f"   비율: {total_with_abstract/len(paper_nodes)*100:.1f}%")

        return abstract_samples

    except Exception as e:
        print(f"❌ 그래프 분석 실패: {e}")
        import traceback

        traceback.print_exc()
        return []


def check_specific_node():
    """특정 노드의 상세 정보 확인"""

    print(f"\n🎯 특정 노드 상세 분석")
    print("-" * 40)

    graph_path = Path("data/processed/graphs/unified/unified_knowledge_graph.json")

    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        # "Jorge Flores-Triana" 논문 찾기
        target_papers = [
            node
            for node in graph_data.get("nodes", [])
            if node.get("node_type") == "paper"
            and "Jorge Flores-Triana" in str(node.get("authors", []))
        ]

        print(f"📋 Jorge Flores-Triana 관련 논문: {len(target_papers)}개")

        for paper in target_papers:
            print(f"\n📄 논문 상세:")
            print(f"   ID: {paper.get('id')}")
            print(f"   제목: {paper.get('title')}")
            print(f"   저자: {paper.get('authors')}")
            print(f"   연도: {paper.get('year')}")
            print(f"   has_abstract: {paper.get('has_abstract')}")

            # 모든 필드 출력
            print(f"   전체 필드:")
            for key, value in paper.items():
                if key not in ["id", "title", "authors", "year", "has_abstract"]:
                    if isinstance(value, str) and len(value) > 100:
                        print(f"     {key}: {value[:100]}...")
                    else:
                        print(f"     {key}: {value}")

    except Exception as e:
        print(f"❌ 특정 노드 분석 실패: {e}")


if __name__ == "__main__":
    # 1. 전체 abstract 통계 확인
    abstract_samples = check_abstract_in_graph()

    # 2. 특정 노드 상세 분석
    check_specific_node()

    print(f"\n💡 다음 단계:")
    if abstract_samples:
        print(f"   ✅ Abstract 데이터가 존재합니다")
        print(f"   🛠️ Context Serializer 코드를 수정하여 abstract 포함하세요")
    else:
        print(f"   ❌ Abstract 데이터가 없습니다")
        print(f"   🔄 그래프 구축 단계에서 abstract 추출이 필요합니다")
