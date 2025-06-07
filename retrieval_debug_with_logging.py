# retrieval_debug_with_logging.py - 출력을 파일로 저장하는 디버깅

import logging
from pathlib import Path
import sys
from datetime import datetime


class TeeOutput:
    """콘솔과 파일에 동시 출력하는 클래스"""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()  # 즉시 파일에 쓰기

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()


def setup_logging_to_file():
    """출력을 파일로 저장하도록 설정"""

    # 로그 디렉토리 생성
    log_dir = Path("debug_logs")
    log_dir.mkdir(exist_ok=True)

    # 타임스탬프가 포함된 파일명
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"retrieval_debug_{timestamp}.txt"

    print(f"📝 출력이 다음 파일에 저장됩니다: {log_filename}")

    # stdout을 파일과 콘솔에 동시 출력하도록 변경
    sys.stdout = TeeOutput(log_filename)

    return log_filename


def debug_retrieval_process(question: str):
    """Retrieval 과정을 단계별로 디버깅 (축약된 출력)"""

    print(f"🔍 디버깅 질문: {question}")
    print("=" * 80)

    try:
        from src.graphrag.langchain.custom_retriever import create_graphrag_retriever

        # Retriever 생성
        retriever = create_graphrag_retriever(
            unified_graph_path="data/processed/graphs/unified/unified_knowledge_graph.json",
            vector_store_path="data/processed/vector_store",
            embedding_model="auto",
            max_docs=10,
            min_relevance_score=0.1,
            enable_caching=False,
        )

        print("✅ Retriever 생성 완료")

        # 검색 실행
        print("\n📋 검색 실행 중...")
        documents = retriever.get_relevant_documents(question)
        print(f"검색된 문서 수: {len(documents)}")

        if documents:
            main_doc = documents[0]
            total_nodes = main_doc.metadata.get("total_nodes", 0)
            confidence = main_doc.metadata.get("confidence_score", 0.0)

            print(f"✅ 검색 성공!")
            print(f"   노드 수: {total_nodes}")
            print(f"   신뢰도: {confidence:.3f}")

            # 컨텍스트 미리보기만 (전체는 파일에만)
            content_preview = main_doc.page_content[:100]
            print(f"   컨텍스트 미리보기: {content_preview}...")

            # 전체 컨텍스트는 파일에만 저장
            print(f"\n📄 전체 컨텍스트 (파일에만 저장):")
            print("=" * 60)
            print(main_doc.page_content)
            print("=" * 60)

            # 프롬프트 생성
            print(f"\n🎯 프롬프트 생성...")
            from src.graphrag.langchain.prompt_templates import GraphRAGPromptTemplates

            prompt_builder = GraphRAGPromptTemplates()
            prompt_template = prompt_builder.create_langchain_prompt()

            context = "\n\n".join([doc.page_content for doc in documents])
            formatted_prompt = prompt_template.format(
                context=context, question=question
            )

            print(f"프롬프트 길이: {len(formatted_prompt)} 문자")
            print(f"프롬프트 미리보기: {formatted_prompt[:150]}...")

            # 전체 프롬프트는 파일에만 저장
            print(f"\n📄 전체 프롬프트 (파일에만 저장):")
            print("=" * 60)
            print(formatted_prompt)
            print("=" * 60)

            return True
        else:
            print("❌ 검색 결과 없음")
            return False

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_different_models_quick():
    """빠른 모델 테스트"""

    print(f"\n🤖 빠른 모델 테스트")
    print("-" * 40)

    models = [
        "paraphrase-multilingual-mpnet-base-v2",  # 768차원
        "all-MiniLM-L6-v2",  # 384차원
        "auto",  # 자동 선택
    ]

    question = "battery machine learning"

    for model in models:
        print(f"\n🧪 테스트 모델: {model}")

        try:
            from src.graphrag.langchain.custom_retriever import (
                create_graphrag_retriever,
            )

            retriever = create_graphrag_retriever(
                unified_graph_path="data/processed/graphs/unified/unified_knowledge_graph.json",
                vector_store_path="data/processed/vector_store",
                embedding_model=model,
                max_docs=5,
                min_relevance_score=0.1,
                enable_caching=False,
            )

            docs = retriever.get_relevant_documents(question)

            if docs:
                total_nodes = docs[0].metadata.get("total_nodes", 0)
                confidence = docs[0].metadata.get("confidence_score", 0.0)
                print(f"   ✅ 성공: {total_nodes}개 노드, 신뢰도: {confidence:.3f}")
            else:
                print(f"   ❌ 실패: 검색 결과 없음")

        except Exception as e:
            print(f"   ❌ 에러: {str(e)[:50]}...")


def test_config_manager_usage():
    """Config Manager 사용 테스트"""

    print(f"\n🔧 Config Manager 사용 테스트")
    print("-" * 40)

    try:
        from src.graphrag.config_manager import GraphRAGConfigManager

        config_manager = GraphRAGConfigManager("graphrag_config.yaml")
        embedding_config = config_manager.get_embeddings_config()

        print(f"✅ Config Manager 로드 성공")
        print(f"   YAML 모델: {embedding_config['model_name']}")

        # Config Manager를 사용한 Retriever 테스트
        from src.graphrag.langchain.custom_retriever import create_graphrag_retriever

        retriever = create_graphrag_retriever(
            unified_graph_path="data/processed/graphs/unified/unified_knowledge_graph.json",
            vector_store_path="data/processed/vector_store",
            embedding_model="auto",
            config_manager=config_manager,  # 핵심: config_manager 전달
            max_docs=5,
            min_relevance_score=0.1,
            enable_caching=False,
        )

        docs = retriever.get_relevant_documents("battery prediction")

        if docs and docs[0].metadata.get("total_nodes", 0) > 0:
            print(f"✅ Config Manager 기반 검색 성공!")
            total_nodes = docs[0].metadata.get("total_nodes", 0)
            print(f"   노드: {total_nodes}개")
            return True
        else:
            print(f"❌ Config Manager 기반 검색 실패")
            return False

    except Exception as e:
        print(f"❌ Config Manager 테스트 실패: {e}")
        return False


def main():
    """메인 실행 함수"""

    # 파일 로깅 설정
    log_filename = setup_logging_to_file()

    print("🚀 GraphRAG Retrieval 디버깅 시작 (파일 저장 버전)")
    print("=" * 80)
    print(f"📝 상세 출력은 {log_filename}에 저장됩니다")
    print("💡 콘솔에는 요약만 출력됩니다")
    print("=" * 80)

    try:
        # 1. Config Manager 테스트
        config_success = test_config_manager_usage()

        # 2. 빠른 모델 테스트
        test_different_models_quick()

        # 3. 전체 retrieval 프로세스 테스트 (성공한 경우만)
        if config_success:
            question = (
                "What machine learning techniques are used for battery SoC prediction?"
            )
            print(f"\n🔍 전체 프로세스 테스트: {question}")
            success = debug_retrieval_process(question)

            if success:
                print(f"\n🎉 디버깅 성공! 이제 정상적인 답변을 받을 수 있습니다.")
            else:
                print(f"\n⚠️ 디버깅에서 문제 발견됨")
        else:
            print(f"\n⚠️ Config Manager 문제로 전체 테스트 건너뜀")

        print(f"\n📝 상세 로그는 다음 파일에서 확인하세요:")
        print(f"   {log_filename}")

    finally:
        # stdout 복원
        if hasattr(sys.stdout, "close"):
            sys.stdout.close()
            sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
