# qa_chain_debug_patch.py - QA Chain에 디버깅 기능 추가

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def patch_qa_chain_for_debugging():
    """QA Chain에 디버깅 기능 추가하는 패치"""

    from src.graphrag.langchain.qa_chain_builder import GraphRAGQAChain

    # 원본 메서드 백업
    original_process_basic_qa = GraphRAGQAChain._process_basic_qa
    original_process_graph_enhanced_qa = GraphRAGQAChain._process_graph_enhanced_qa

    def debug_process_basic_qa(self, question: str, run_manager=None):
        """디버깅이 추가된 기본 QA 처리"""

        print(f"\n🔍 [DEBUG] Basic QA 처리 시작")
        print(f"질문: {question}")
        print("-" * 60)

        try:
            # 1. 문서 검색
            print("📋 1단계: 문서 검색")
            docs = self.retriever.get_relevant_documents(question)
            print(f"검색된 문서 수: {len(docs)}")

            for i, doc in enumerate(docs[:3]):  # 상위 3개만 출력
                print(f"\n문서 {i+1}:")
                print(f"  내용: {doc.page_content[:150]}...")
                print(f"  메타데이터: {doc.metadata}")

            # 2. 컨텍스트 결합
            print(f"\n📝 2단계: 컨텍스트 결합")
            context = "\n\n".join([doc.page_content for doc in docs[:5]])
            print(f"결합된 컨텍스트 길이: {len(context)} 문자")

            if len(context) > 0:
                print(f"컨텍스트 미리보기:\n{context[:200]}...")
            else:
                print("⚠️ 컨텍스트가 비어있습니다!")

            # 3. 프롬프트 생성
            print(f"\n🎯 3단계: 프롬프트 생성")
            prompt_inputs = {
                "context": context,
                "input": question,
            }

            formatted_prompt = self.prompt_template.format(**prompt_inputs)

            print(f"프롬프트 길이: {len(formatted_prompt)} 문자")
            print(f"\n📄 최종 프롬프트:")
            print("=" * 60)
            print(formatted_prompt)
            print("=" * 60)

            # 4. LLM 호출
            print(f"\n🤖 4단계: LLM 호출")
            answer = self.llm.invoke(formatted_prompt)
            answer_str = answer if isinstance(answer, str) else str(answer)

            print(f"LLM 응답 길이: {len(answer_str)} 문자")
            print(f"LLM 응답: {answer_str[:200]}...")

            # 원본 메서드 결과와 동일한 형태로 반환
            from src.graphrag.langchain.qa_chain_builder import process_source_documents

            processed_context = process_source_documents(docs)

            result = {
                "answer": answer_str,
                "source_documents": processed_context,
            }

            print(f"\n✅ Basic QA 처리 완료")
            return result

        except Exception as e:
            print(f"\n❌ Basic QA 처리 실패: {e}")
            import traceback

            traceback.print_exc()

            return {
                "answer": f"처리 중 오류가 발생했습니다: {str(e)}",
                "source_documents": [],
            }

    def debug_process_graph_enhanced_qa(
        self, question: str, query_analysis, run_manager
    ):
        """디버깅이 추가된 GraphRAG QA 처리"""

        print(f"\n🔍 [DEBUG] Graph Enhanced QA 처리 시작")
        print(f"질문: {question}")
        if query_analysis:
            print(
                f"쿼리 분석: {query_analysis.query_type.value} ({query_analysis.complexity.value})"
            )
        print("-" * 60)

        try:
            if self.retrieval_chain:
                print("📋 LangChain Retrieval Chain 사용")
                result = self.retrieval_chain.invoke({"input": question})

                # context 처리 디버깅
                context_docs = result.get("context", [])
                print(f"Retrieval Chain 결과 문서 수: {len(context_docs)}")

                for i, doc in enumerate(context_docs[:2]):
                    print(f"문서 {i+1}: {doc.page_content[:100]}...")

                from src.graphrag.langchain.qa_chain_builder import (
                    process_source_documents,
                )

                processed_context = process_source_documents(context_docs)

            else:
                print("📋 직접 Retriever 사용")
                docs = self.retriever.get_relevant_documents(question)
                print(f"검색된 문서 수: {len(docs)}")

                context = "\n\n".join([doc.page_content for doc in docs])
                print(f"결합된 컨텍스트 길이: {len(context)} 문자")

                # 프롬프트 변수 디버깅
                if hasattr(self.prompt_template, "input_variables"):
                    print(f"프롬프트 입력 변수: {self.prompt_template.input_variables}")

                    if "question" in self.prompt_template.input_variables:
                        prompt_inputs = {"context": context, "question": question}
                    else:
                        prompt_inputs = {"context": context, "input": question}
                else:
                    prompt_inputs = {"context": context, "question": question}

                print(f"프롬프트 입력: {list(prompt_inputs.keys())}")

                formatted_prompt = self.prompt_template.format(**prompt_inputs)
                print(f"\n📄 최종 프롬프트 (앞 300자):")
                print(formatted_prompt[:300] + "...")

                answer = self.llm.invoke(formatted_prompt)

                from src.graphrag.langchain.qa_chain_builder import (
                    process_source_documents,
                )

                processed_context = process_source_documents(docs)
                result = {"answer": str(answer), "context": processed_context}

            print(f"\n✅ Graph Enhanced QA 처리 완료")
            return {
                "answer": result.get("answer", ""),
                "source_documents": result.get("context", processed_context),
            }

        except Exception as e:
            print(f"\n❌ Graph Enhanced QA 처리 실패: {e}")
            import traceback

            traceback.print_exc()

            return {
                "answer": f"처리 중 오류가 발생했습니다: {str(e)}",
                "source_documents": [],
            }

    # 메서드 패치 적용
    GraphRAGQAChain._process_basic_qa = debug_process_basic_qa
    GraphRAGQAChain._process_graph_enhanced_qa = debug_process_graph_enhanced_qa

    print("✅ QA Chain 디버깅 패치 적용 완료")


def create_debug_qa_chain():
    """디버깅이 활성화된 QA Chain 생성"""

    # 패치 적용
    patch_qa_chain_for_debugging()

    # QA Chain Builder 생성
    from src.graphrag.langchain.qa_chain_builder import QAChainBuilder

    builder = QAChainBuilder(
        unified_graph_path="data/processed/graphs/unified/unified_knowledge_graph.json",
        vector_store_path="data/processed/vector_store",
    )

    # 디버깅용 설정
    qa_chain = builder.create_basic_chain(
        max_docs=10,
        min_relevance_score=0.2,  # 낮은 임계값
        enable_memory=False,
        return_source_documents=True,
    )

    return qa_chain


# 사용 예시
if __name__ == "__main__":
    print("🚀 QA Chain 디버깅 패치 테스트")

    # 디버깅 QA Chain 생성
    qa_chain = create_debug_qa_chain()

    # 테스트 질문
    question = "What machine learning techniques are used for battery SoC prediction?"

    print(f"\n📋 테스트 질문: {question}")
    print("=" * 80)

    # QA Chain 실행 (디버깅 출력과 함께)
    result = qa_chain.invoke({"question": question})

    print(f"\n📊 최종 결과:")
    print(f"답변: {result.get('answer', 'N/A')}")
    print(f"소스 문서 수: {len(result.get('source_documents', []))}")
