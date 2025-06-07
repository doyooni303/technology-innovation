# retrieval_debug.py - Retrieval 과정 상세 디버깅

import logging
from pathlib import Path

# GraphRAG 컴포넌트들 직접 생성
from src.graphrag.langchain.custom_retriever import create_graphrag_retriever

# 1. 상세 로깅 활성화
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("graphrag")
logger.setLevel(logging.DEBUG)


def debug_retrieval_process(question: str):
    """Retrieval 과정을 단계별로 디버깅"""

    print(f"🔍 디버깅 질문: {question}")
    print("=" * 80)

    try:
        # Retriever 생성
        retriever = create_graphrag_retriever(
            unified_graph_path="data/processed/graphs/unified/unified_knowledge_graph.json",
            vector_store_path="data/processed/vector_store",
            embedding_model="auto",
            max_docs=10,
            min_relevance_score=0.3,  # 낮춰서 더 많은 결과 확인
            enable_caching=False,  # 캐시 비활성화
        )

        print("✅ Retriever 생성 완료")

        # 1단계: 원시 검색 결과 확인
        print("\n📋 1단계: 원시 검색 결과")
        documents = retriever.get_relevant_documents(question)
        print(f"검색된 문서 수: {len(documents)}")

        for i, doc in enumerate(documents):
            print(f"\n--- 문서 {i+1} ---")
            print(f"내용 (앞 200자): {doc.page_content[:200]}...")
            print(f"메타데이터: {doc.metadata}")

            # 관련성 점수 확인
            if "confidence_score" in doc.metadata:
                print(f"신뢰도 점수: {doc.metadata['confidence_score']}")

        # 2단계: Context Serialization 결과 확인
        print(f"\n📝 2단계: 직렬화된 컨텍스트")
        if documents:
            main_doc = documents[0]  # 메인 문서
            print(f"직렬화된 텍스트:\n{main_doc.page_content}")
        else:
            print("❌ 검색된 문서가 없습니다!")

        # 3단계: 프롬프트 생성 과정 확인
        print(f"\n🎯 3단계: 프롬프트 변환 과정")

        # 컨텍스트 결합
        context = "\n\n".join([doc.page_content for doc in documents])
        print(f"결합된 컨텍스트 길이: {len(context)} 문자")
        print(f"결합된 컨텍스트 (앞 300자):\n{context[:300]}...")

        # 프롬프트 템플릿 적용
        from src.graphrag.langchain.prompt_templates import GraphRAGPromptTemplates

        prompt_builder = GraphRAGPromptTemplates()
        prompt_template = prompt_builder.create_langchain_prompt()

        # 프롬프트 포맷팅
        formatted_prompt = prompt_template.format(context=context, question=question)

        print(f"\n📄 최종 프롬프트:")
        print("=" * 50)
        print(formatted_prompt)
        print("=" * 50)

        return {
            "documents": documents,
            "context": context,
            "formatted_prompt": formatted_prompt,
        }

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback

        traceback.print_exc()
        return None


def debug_vector_search_threshold(question: str):
    """벡터 검색 임계값 테스트"""

    print(f"\n🎯 벡터 검색 임계값 테스트: {question}")

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    for threshold in thresholds:
        # try:
        retriever = create_graphrag_retriever(
            unified_graph_path="data/processed/graphs/unified/unified_knowledge_graph.json",
            vector_store_path="data/processed/vector_store",
            embedding_model="auto",
            max_docs=5,
            min_relevance_score=threshold,
            enable_caching=False,
        )

        docs = retriever.get_relevant_documents(question)
        print(f"임계값 {threshold}: {len(docs)}개 문서")

        if docs:
            best_score = docs[0].metadata.get("confidence_score", 0.0)
            print(f"   최고 점수: {best_score:.3f}")

    # except Exception as e:
    #     print(f"임계값 {threshold}: 에러 - {e}")


def debug_embedding_similarity(question: str):
    from src.graphrag.embeddings.embedding_models import create_embedding_model

    model = create_embedding_model("auto")  # 올바른 함수

    # 질문 임베딩
    question_embedding = model.encode([question])
    print(f"질문 임베딩 차원: {question_embedding.shape}")

    # 벡터 스토어에서 상위 결과 직접 검색
    from src.graphrag.embeddings.vector_store_manager import VectorStoreManager

    vector_manager = VectorStoreManager(
        "faiss", "data/processed/vector_store/faiss/faiss"
    )

    # 상위 20개 결과 가져오기
    results = vector_manager.search(question_embedding[0], k=20)

    print(f"벡터 검색 결과: {len(results)}개")

    for i, result in enumerate(results[:5]):
        print(
            f"결과 {i+1}: node_id={result.node_id}, score={result.similarity_score:.3f}"
        )

    # except Exception as e:
    #     print(f"❌ 임베딩 테스트 에러: {e}")
    #     import traceback

    #     traceback.print_exc()


# 사용 예시
if __name__ == "__main__":
    question = "What machine learning techniques are used for battery SoC prediction?"

    print("🚀 GraphRAG Retrieval 디버깅 시작")
    print("=" * 80)

    # 1. 전체 retrieval 과정 디버깅
    result = debug_retrieval_process(question)

    # 2. 임계값 테스트
    debug_vector_search_threshold(question)

    # 3. 임베딩 유사도 테스트
    debug_embedding_similarity(question)

    print("\n✅ 디버깅 완료!")
