# faiss_diagnostic.py - FAISS 검색 실패 문제 진단 및 해결

import logging
import traceback
from pathlib import Path
import numpy as np


def diagnose_faiss_issue():
    """FAISS 문제 상세 진단"""

    print("🔍 FAISS 문제 진단 시작")
    print("=" * 80)

    # 1. FAISS 모듈 및 GPU 상태 확인
    print("1️⃣ FAISS 모듈 상태 확인")
    print("-" * 40)

    try:
        import faiss

        print(f"✅ FAISS 버전: {faiss.__version__}")

        # GPU 확인
        ngpus = faiss.get_num_gpus()
        print(f"🎮 사용 가능한 GPU: {ngpus}개")

        if ngpus > 0:
            for i in range(ngpus):
                try:
                    gpu_info = faiss.GpuResourcesInfo(i)
                    print(f"   GPU {i}: 사용 가능")
                except:
                    print(f"   GPU {i}: 사용 불가")
    except Exception as e:
        print(f"❌ FAISS 모듈 문제: {e}")
        return False

    # 2. 기존 FAISS 인덱스 파일 확인
    print(f"\n2️⃣ FAISS 인덱스 파일 확인")
    print("-" * 40)

    index_path = Path("data/processed/vector_store/faiss/faiss/faiss_index.bin")
    metadata_path = Path("data/processed/vector_store/faiss/faiss/faiss_metadata.pkl")

    if not index_path.exists():
        print(f"❌ FAISS 인덱스 파일이 없습니다: {index_path}")
        return False

    if not metadata_path.exists():
        print(f"❌ FAISS 메타데이터 파일이 없습니다: {metadata_path}")
        return False

    print(
        f"✅ 인덱스 파일 존재: {index_path} ({index_path.stat().st_size / 1024 / 1024:.1f} MB)"
    )
    print(f"✅ 메타데이터 파일 존재: {metadata_path}")

    # 3. FAISS 인덱스 직접 로드 테스트
    print(f"\n3️⃣ FAISS 인덱스 직접 로드 테스트")
    print("-" * 40)

    try:
        # CPU에서 인덱스 로드
        print("📂 CPU에서 인덱스 로딩...")
        index = faiss.read_index(str(index_path))
        print(f"✅ 인덱스 로드 성공")
        print(f"   벡터 수: {index.ntotal}")
        print(f"   차원: {index.d}")
        print(f"   인덱스 타입: {type(index)}")

        # 메타데이터 로드
        print("\n📂 메타데이터 로딩...")
        import pickle

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        print(f"✅ 메타데이터 로드 성공")
        print(f"   노드 수: {len(metadata.get('node_id_to_idx', {}))}")
        print(f"   총 벡터: {metadata.get('total_vectors', 0)}")
        print(f"   차원: {metadata.get('dimension', 0)}")

        return index, metadata

    except Exception as e:
        print(f"❌ 인덱스 로드 실패: {e}")
        traceback.print_exc()
        return False


def test_faiss_search_directly(index, metadata):
    """FAISS 검색을 직접 테스트"""

    print(f"\n4️⃣ FAISS 검색 직접 테스트")
    print("-" * 40)

    try:
        # 테스트용 랜덤 쿼리 벡터 생성
        dimension = index.d
        print(f"📏 차원: {dimension}")

        # 실제 임베딩 하나를 가져와서 테스트
        if index.ntotal > 0:
            print("🔍 기존 벡터로 검색 테스트...")
            # 첫 번째 벡터를 쿼리로 사용
            test_vector = index.reconstruct(0).reshape(1, -1)

            print(f"   테스트 벡터 형태: {test_vector.shape}")
            print(f"   테스트 벡터 타입: {test_vector.dtype}")

            # 검색 실행
            scores, indices = index.search(test_vector, 5)

            print(f"✅ 검색 성공!")
            print(f"   반환된 점수: {scores[0]}")
            print(f"   반환된 인덱스: {indices[0]}")

            # 노드 ID 매핑 확인
            idx_to_node_id = metadata.get("idx_to_node_id", {})
            for i, idx in enumerate(indices[0]):
                node_id = idx_to_node_id.get(idx, f"unknown_{idx}")
                print(
                    f"   결과 {i+1}: index={idx}, node_id={node_id}, score={scores[0][i]:.4f}"
                )

            return True
        else:
            print("❌ 인덱스에 벡터가 없습니다")
            return False

    except Exception as e:
        print(f"❌ FAISS 검색 실패: {e}")
        traceback.print_exc()

        # 더 상세한 에러 정보
        print(f"\n🔍 상세 에러 분석:")
        print(f"   인덱스 타입: {type(index)}")
        print(f"   인덱스 ntotal: {index.ntotal}")
        print(f"   인덱스 is_trained: {index.is_trained}")

        return False


# faiss_diag.py 파일 수정
def test_embedding_model():
    """임베딩 모델 테스트"""
    try:
        # ✅ 수정된 import
        from src.graphrag.embeddings import create_embedding_model

        # get_embedding_model 대신 create_embedding_model 사용
        model = create_embedding_model(
            model_name="paraphrase-multilingual-mpnet-base-v2", device="auto"
        )

        # 테스트 임베딩
        test_text = "This is a test sentence for embedding."
        embedding = model.encode([test_text])

        print(f"✅ 임베딩 모델 테스트 성공")
        print(f"   모델: {model}")
        print(f"   임베딩 차원: {embedding.shape}")

        return True

    except Exception as e:
        print(f"❌ 임베딩 모델 테스트 실패: {e}")
        return False


def test_end_to_end_search():
    """전체 검색 파이프라인 테스트"""

    print(f"\n6️⃣ 전체 검색 파이프라인 테스트")
    print("-" * 40)

    try:
        from src.graphrag.embeddings.vector_store_manager import VectorStoreManager

        # VectorStoreManager 생성 (CPU만 사용)
        print("🔧 VectorStoreManager 생성 (CPU 모드)...")
        vector_manager = VectorStoreManager(
            store_type="faiss",
            persist_directory="data/processed/vector_store/faiss/faiss",
            use_gpu=False,  # GPU 비활성화
        )

        # 기존 인덱스 로드
        print("📂 기존 인덱스 로드...")
        vector_manager.load()

        print(f"✅ VectorStoreManager 초기화 완료")
        print(f"   총 벡터 수: {vector_manager.total_vectors}")

        # 임베딩 모델로 쿼리 벡터 생성
        print(f"\n🎯 쿼리 벡터 생성...")
        from src.graphrag.embeddings.embedding_models import get_embedding_model

        model = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = model.encode(["battery machine learning"])[0]

        print(f"   쿼리 임베딩 형태: {query_embedding.shape}")

        # 검색 실행
        print(f"\n🔍 검색 실행...")
        results = vector_manager.search(query_embedding, k=5)

        print(f"✅ 검색 성공!")
        print(f"   결과 수: {len(results)}")

        for i, result in enumerate(results):
            print(
                f"   결과 {i+1}: node_id={result.node_id}, score={result.similarity_score:.4f}"
            )
            print(f"     텍스트: {result.text[:100]}...")

        return True

    except Exception as e:
        print(f"❌ 전체 파이프라인 테스트 실패: {e}")
        traceback.print_exc()
        return False


def fix_faiss_issues():
    """FAISS 문제 수정 시도"""

    print(f"\n🛠️ FAISS 문제 수정 시도")
    print("-" * 40)

    try:
        # 1. GPU 사용 비활성화하여 재시도
        print("1️⃣ GPU 비활성화하여 FAISS 테스트...")

        from src.graphrag.langchain.custom_retriever import create_graphrag_retriever

        retriever = create_graphrag_retriever(
            unified_graph_path="data/processed/graphs/unified/unified_knowledge_graph.json",
            vector_store_path="data/processed/vector_store",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            max_docs=5,
            min_relevance_score=0.1,
            enable_caching=False,
            # GPU 비활성화
            vector_store_config={"use_gpu": False, "store_type": "faiss"},
        )

        # 테스트 검색
        docs = retriever.get_relevant_documents("battery")

        if docs and docs[0].metadata.get("total_nodes", 0) > 0:
            print("✅ GPU 비활성화로 문제 해결!")
            print(f"   검색된 노드 수: {docs[0].metadata.get('total_nodes', 0)}")
            return "gpu_disabled"
        else:
            print("❌ GPU 비활성화로도 해결되지 않음")

        # 2. 임베딩 재생성 시도
        print(f"\n2️⃣ 임베딩 재생성 시도...")
        # 이 부분은 너무 오래 걸리므로 제안만 함
        print("💡 임베딩을 다시 생성해야 할 수 있습니다:")
        print(
            "   python -m src.graphrag.graphrag_pipeline build_embeddings --force-rebuild"
        )

        return "embeddings_rebuild_needed"

    except Exception as e:
        print(f"❌ 수정 시도 실패: {e}")
        traceback.print_exc()
        return "failed"


# 메인 실행
if __name__ == "__main__":
    print("🚀 FAISS 문제 진단 시작")
    print("=" * 80)

    # 1. 기본 진단
    result = diagnose_faiss_issue()

    if result:
        index, metadata = result

        # 2. 직접 검색 테스트
        search_ok = test_faiss_search_directly(index, metadata)

        # 3. 임베딩 모델 테스트
        embeddings = test_embedding_model()

        # 4. 전체 파이프라인 테스트
        if search_ok and embeddings is not None:
            pipeline_ok = test_end_to_end_search()

            if not pipeline_ok:
                # 5. 문제 수정 시도
                fix_result = fix_faiss_issues()
                print(f"\n📋 최종 진단 결과: {fix_result}")
        else:
            # FAISS 레벨에서 문제 발생
            fix_result = fix_faiss_issues()
            print(f"\n📋 최종 진단 결과: {fix_result}")
    else:
        print(f"\n❌ FAISS 기본 설정에 문제가 있습니다.")
        print(f"💡 해결 방법:")
        print(f"   1. FAISS 재설치: pip uninstall faiss-cpu && pip install faiss-cpu")
        print(f"   2. NumPy 버전 확인: pip install 'numpy<2.0'")
        print(f"   3. 임베딩 재생성 필요")

    print(f"\n✅ 진단 완료!")
