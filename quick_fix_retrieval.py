# quick_fix_retrieval.py - 즉시 적용 가능한 retrieval 문제 해결

import logging
from pathlib import Path


def test_different_thresholds_and_models():
    """다양한 임계값과 모델로 테스트"""

    print("🔧 Retrieval 문제 해결 테스트 시작")
    print("=" * 80)

    from src.graphrag.langchain.custom_retriever import create_graphrag_retriever

    # 1. 임계값을 단계적으로 낮춰서 테스트
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",  # 가장 일반적
        "sentence-transformers/all-mpnet-base-v2",  # 고성능
        "sentence-transformers/paraphrase-MiniLM-L6-v2",  # 빠름
        "auto",  # 기본값
    ]

    question = "What machine learning techniques are used for battery SoC prediction?"

    best_result = None
    best_score = 0

    for model in models:
        print(f"\n🤖 테스트 모델: {model}")
        print("-" * 60)

        for threshold in thresholds:
            try:
                print(f"   임계값 {threshold}: ", end="")

                retriever = create_graphrag_retriever(
                    unified_graph_path="data/processed/graphs/unified/unified_knowledge_graph.json",
                    vector_store_path="data/processed/vector_store",
                    embedding_model=model,
                    max_docs=10,
                    min_relevance_score=threshold,
                    enable_caching=False,
                )

                docs = retriever.get_relevant_documents(question)

                if docs:
                    confidence = docs[0].metadata.get("confidence_score", 0.0)
                    total_nodes = docs[0].metadata.get("total_nodes", 0)

                    print(
                        f"✅ {len(docs)}개 문서, 신뢰도: {confidence:.3f}, 노드: {total_nodes}"
                    )

                    # 최고 결과 추적
                    if confidence > best_score:
                        best_score = confidence
                        best_result = {
                            "model": model,
                            "threshold": threshold,
                            "docs": docs,
                            "confidence": confidence,
                        }
                else:
                    print("❌ 검색 결과 없음")

            except Exception as e:
                print(f"❌ 에러: {str(e)[:50]}...")

    # 최고 결과 출력
    if best_result:
        print(f"\n🏆 최고 결과:")
        print(f"   모델: {best_result['model']}")
        print(f"   임계값: {best_result['threshold']}")
        print(f"   신뢰도: {best_result['confidence']:.3f}")
        print(f"   문서 수: {len(best_result['docs'])}")

        # 최고 결과의 내용 미리보기
        best_doc = best_result["docs"][0]
        print(f"\n📄 최고 결과 내용 미리보기:")
        print(f"{best_doc.page_content[:300]}...")

        return best_result
    else:
        print("\n❌ 모든 테스트에서 검색 결과 없음")
        return None


def test_simplified_queries():
    """단순화된 질문으로 테스트"""

    print(f"\n🔍 단순화된 질문으로 테스트")
    print("-" * 60)

    simple_queries = [
        "battery",
        "machine learning",
        "SoC prediction",
        "neural network",
        "deep learning",
        "artificial intelligence",
        "energy",
        "lithium",
    ]

    from src.graphrag.langchain.custom_retriever import create_graphrag_retriever

    # 가장 관대한 설정으로 테스트
    retriever = create_graphrag_retriever(
        unified_graph_path="data/processed/graphs/unified/unified_knowledge_graph.json",
        vector_store_path="data/processed/vector_store",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        max_docs=5,
        min_relevance_score=0.1,  # 매우 낮은 임계값
        enable_caching=False,
    )

    results = []

    for query in simple_queries:
        try:
            docs = retriever.get_relevant_documents(query)
            if docs:
                confidence = docs[0].metadata.get("confidence_score", 0.0)
                total_nodes = docs[0].metadata.get("total_nodes", 0)
                print(
                    f"   '{query}': ✅ {len(docs)}개, 신뢰도: {confidence:.3f}, 노드: {total_nodes}"
                )
                results.append((query, docs, confidence))
            else:
                print(f"   '{query}': ❌ 결과 없음")
        except Exception as e:
            print(f"   '{query}': ❌ 에러: {str(e)[:30]}...")

    return results


def inspect_vector_store_directly():
    """벡터 스토어 직접 검사"""

    print(f"\n🔍 벡터 스토어 직접 검사")
    print("-" * 60)

    try:
        # FAISS 인덱스 직접 로드
        import faiss
        import numpy as np
        from pathlib import Path

        index_path = Path("data/processed/vector_store/faiss/faiss/faiss_index.bin")

        if index_path.exists():
            print(f"✅ FAISS 인덱스 파일 존재: {index_path}")

            # 인덱스 로드
            index = faiss.read_index(str(index_path))
            print(f"   벡터 수: {index.ntotal}")
            print(f"   차원: {index.d}")

            # 임베딩 메타데이터 확인
            embeddings_dir = Path("data/processed/vector_store/embeddings")
            if embeddings_dir.exists():
                embedding_files = list(embeddings_dir.glob("*.json"))
                print(f"   임베딩 메타데이터 파일 수: {len(embedding_files)}")

                # 첫 번째 파일 내용 확인
                if embedding_files:
                    import json

                    with open(embedding_files[0], "r", encoding="utf-8") as f:
                        sample = json.load(f)
                    print(f"   샘플 메타데이터: {list(sample.keys())[:5]}")

            return True
        else:
            print(f"❌ FAISS 인덱스 파일이 없습니다: {index_path}")
            return False

    except Exception as e:
        print(f"❌ 벡터 스토어 검사 실패: {e}")
        return False


def create_working_retriever():
    """작동하는 retriever 설정 생성"""

    print(f"\n🛠️ 작동하는 Retriever 설정 생성")
    print("-" * 60)

    from src.graphrag.langchain.custom_retriever import create_graphrag_retriever

    # 최적화된 설정
    optimal_config = {
        "unified_graph_path": "data/processed/graphs/unified/unified_knowledge_graph.json",
        "vector_store_path": "data/processed/vector_store",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "max_docs": 15,
        "min_relevance_score": 0.1,  # 매우 낮은 임계값
        "enable_caching": True,
        "enable_query_analysis": True,
    }

    try:
        retriever = create_graphrag_retriever(**optimal_config)

        # 테스트 질문들로 검증
        test_queries = [
            "battery SoC prediction machine learning",
            "neural network energy storage",
            "deep learning lithium battery",
        ]

        working_queries = []

        for query in test_queries:
            docs = retriever.get_relevant_documents(query)
            if docs and docs[0].metadata.get("total_nodes", 0) > 0:
                working_queries.append(query)
                print(f"✅ 작동: '{query}' → {len(docs)}개 문서")
            else:
                print(f"❌ 실패: '{query}'")

        if working_queries:
            print(f"\n🎉 작동하는 설정 발견!")
            print(f"   모델: {optimal_config['embedding_model']}")
            print(f"   임계값: {optimal_config['min_relevance_score']}")
            print(f"   작동하는 질문: {len(working_queries)}개")

            # 설정을 파일로 저장
            import json

            with open("working_retriever_config.json", "w") as f:
                json.dump(optimal_config, f, indent=2)
            print(f"   설정 저장: working_retriever_config.json")

            return retriever, optimal_config
        else:
            print(f"❌ 모든 테스트 질문에서 실패")
            return None, None

    except Exception as e:
        print(f"❌ Retriever 생성 실패: {e}")
        return None, None


# 메인 실행
if __name__ == "__main__":
    print("🚀 Retrieval 문제 해결 시작")
    print("=" * 80)

    # 1. 벡터 스토어 상태 확인
    vector_store_ok = inspect_vector_store_directly()

    if vector_store_ok:
        # 2. 다양한 임계값과 모델 테스트
        best_result = test_different_thresholds_and_models()

        # 3. 단순 질문 테스트
        simple_results = test_simplified_queries()

        # 4. 작동하는 설정 생성
        working_retriever, config = create_working_retriever()

        print(f"\n✅ 문제 해결 완료!")

        if working_retriever and config:
            print(f"추천 설정:")
            print(f"  - 모델: {config['embedding_model']}")
            print(f"  - 임계값: {config['min_relevance_score']}")
            print(f"  - 최대 문서: {config['max_docs']}")

            print(f"\n📝 사용법:")
            print(f"retriever = create_graphrag_retriever(**config)")
    else:
        print(f"❌ 벡터 스토어에 문제가 있습니다. 먼저 임베딩을 다시 생성해야 합니다.")
