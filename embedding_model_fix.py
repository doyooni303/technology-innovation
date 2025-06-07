# embedding_model_fix.py - 임베딩 모델 불일치 문제 해결

import json
import pickle
import numpy as np
from pathlib import Path


def diagnose_embedding_model_mismatch():
    """임베딩 모델 불일치 문제 진단"""

    print("🔍 임베딩 모델 불일치 문제 진단")
    print("=" * 60)

    # 1. 저장된 임베딩 메타데이터 확인
    metadata_path = Path("data/processed/vector_store/faiss/faiss/faiss_metadata.pkl")

    if metadata_path.exists():
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        print("📊 저장된 임베딩 정보:")
        print(f"   차원: {metadata.get('dimension', 'Unknown')}")
        print(f"   총 벡터: {metadata.get('total_vectors', 'Unknown')}")
        print(f"   노드 수: {len(metadata.get('node_id_to_idx', {}))}")

        # 첫 번째 노드의 임베딩 확인
        if metadata.get("node_metadatas"):
            first_node_id = list(metadata["node_metadatas"].keys())[0]
            print(f"   첫 번째 노드: {first_node_id}")
            print(f"   메타데이터: {metadata['node_metadatas'][first_node_id]}")

    # 2. 임베딩 원본 파일들 확인
    embeddings_dir = Path("data/processed/vector_store/embeddings")

    if embeddings_dir.exists():
        print(f"\n📂 원본 임베딩 파일들:")

        embedding_files = list(embeddings_dir.glob("*.npy"))
        json_files = list(embeddings_dir.glob("*.json"))

        print(f"   NumPy 파일: {len(embedding_files)}개")
        print(f"   JSON 파일: {len(json_files)}개")

        # 첫 번째 임베딩 파일 샘플 확인
        if embedding_files:
            sample_file = embedding_files[0]
            embeddings = np.load(sample_file)
            print(f"   샘플 임베딩 형태: {embeddings.shape}")
            print(f"   샘플 파일: {sample_file.name}")

            return embeddings.shape[1]  # 차원 반환

    return None


def test_current_embedding_model():
    """현재 임베딩 모델의 차원 확인"""

    print(f"\n🤖 현재 임베딩 모델 테스트")
    print("-" * 40)

    try:
        # SentenceTransformers 직접 테스트
        print("📥 SentenceTransformers 모델 로드...")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        test_text = "battery machine learning"
        embedding = model.encode([test_text])

        print(f"✅ 현재 모델 테스트 성공")
        print(f"   모델: sentence-transformers/all-MiniLM-L6-v2")
        print(f"   차원: {embedding.shape[1]}")
        print(f"   테스트 임베딩 형태: {embedding.shape}")

        return embedding.shape[1]

    except Exception as e:
        print(f"❌ 현재 모델 테스트 실패: {e}")
        return None


def fix_embedding_model_mismatch():
    """임베딩 모델 불일치 해결"""

    print(f"\n🛠️ 임베딩 모델 불일치 해결")
    print("-" * 40)

    # 저장된 차원과 현재 모델 차원 확인
    stored_dim = diagnose_embedding_model_mismatch()
    current_dim = test_current_embedding_model()

    if stored_dim and current_dim:
        print(f"\n📊 차원 비교:")
        print(f"   저장된 임베딩 차원: {stored_dim}")
        print(f"   현재 모델 차원: {current_dim}")

        if stored_dim != current_dim:
            print(f"\n❌ 차원 불일치 발견!")
            print(f"   해결 방법들:")

            # 방법 1: 올바른 모델로 변경
            if stored_dim == 768:
                print(f"\n🎯 방법 1: 저장된 임베딩과 호환되는 모델 사용")
                print(f"   추천 모델: paraphrase-multilingual-mpnet-base-v2 (768차원)")
                return "use_768_model"

            elif stored_dim == 384:
                print(f"\n🎯 방법 1: 현재 모델이 올바름")
                print(f"   저장된 임베딩에 문제가 있을 수 있음")
                return "current_model_correct"

            # 방법 2: 임베딩 재생성
            print(f"\n🎯 방법 2: 임베딩 재생성 (확실한 해결)")
            print(f"   명령어: python rebuild_embeddings.py")

            return "rebuild_needed"
        else:
            print(f"\n✅ 차원이 일치합니다!")
            print(f"   다른 문제가 있을 수 있습니다.")
            return "dimensions_match"

    return "unknown"


def create_compatible_retriever():
    """호환 가능한 retriever 생성"""

    print(f"\n🔧 호환 가능한 Retriever 생성")
    print("-" * 40)

    # 768차원 모델로 테스트
    try:
        from src.graphrag.langchain.custom_retriever import create_graphrag_retriever

        print("🧪 768차원 모델로 테스트...")
        retriever = create_graphrag_retriever(
            unified_graph_path="data/processed/graphs/unified/unified_knowledge_graph.json",
            vector_store_path="data/processed/vector_store",
            embedding_model="paraphrase-multilingual-mpnet-base-v2",  # 768차원
            max_docs=5,
            min_relevance_score=0.1,
            enable_caching=False,
        )

        # 테스트 검색
        docs = retriever.get_relevant_documents("battery machine learning")

        if docs and docs[0].metadata.get("total_nodes", 0) > 0:
            print("✅ 768차원 모델로 성공!")
            print(f"   검색된 노드 수: {docs[0].metadata.get('total_nodes')}")
            return "768_model_works"
        else:
            print("❌ 768차원 모델로도 실패")

    except Exception as e:
        print(f"❌ 768차원 모델 테스트 실패: {e}")

    # 384차원 모델로 재시도
    try:
        print("\n🧪 384차원 모델로 테스트...")
        retriever = create_graphrag_retriever(
            unified_graph_path="data/processed/graphs/unified/unified_knowledge_graph.json",
            vector_store_path="data/processed/vector_store",
            embedding_model="all-MiniLM-L6-v2",  # 384차원
            max_docs=5,
            min_relevance_score=0.1,
            enable_caching=False,
        )

        # 테스트 검색
        docs = retriever.get_relevant_documents("battery machine learning")

        if docs and docs[0].metadata.get("total_nodes", 0) > 0:
            print("✅ 384차원 모델로 성공!")
            print(f"   검색된 노드 수: {docs[0].metadata.get('total_nodes')}")
            return "384_model_works"
        else:
            print("❌ 384차원 모델로도 실패")

    except Exception as e:
        print(f"❌ 384차원 모델 테스트 실패: {e}")

    return "all_failed"


def quick_rebuild_embeddings():
    """빠른 임베딩 재생성"""

    print(f"\n🚀 빠른 임베딩 재생성")
    print("-" * 40)

    try:
        # 기존 벡터 스토어 백업
        import shutil

        vector_store_path = Path("data/processed/vector_store")
        backup_path = Path("data/processed/vector_store_backup")

        if vector_store_path.exists():
            print("📦 기존 벡터 스토어 백업...")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.copytree(vector_store_path, backup_path)
            print(f"   백업 완료: {backup_path}")

        # FAISS 인덱스만 삭제 (원본 임베딩은 유지)
        faiss_path = vector_store_path / "faiss"
        if faiss_path.exists():
            print("🗑️ 기존 FAISS 인덱스 삭제...")
            shutil.rmtree(faiss_path)

        print("💡 이제 다음을 실행하세요:")
        print("   1. 설정 파일에서 임베딩 모델 확인")
        print("   2. python -m src.graphrag.graphrag_pipeline build_embeddings")

        return True

    except Exception as e:
        print(f"❌ 재생성 준비 실패: {e}")
        return False


# 메인 실행
if __name__ == "__main__":
    print("🚀 임베딩 모델 불일치 문제 해결 시작")
    print("=" * 60)

    # 1. 문제 진단
    issue_type = fix_embedding_model_mismatch()

    # 2. 호환 모델 테스트
    if issue_type in ["use_768_model", "rebuild_needed"]:
        working_model = create_compatible_retriever()

        if working_model in ["768_model_works", "384_model_works"]:
            print(f"\n✅ 해결책 발견!")

            if working_model == "768_model_works":
                print(f"📝 graphrag_config.yaml에서 다음으로 변경:")
                print(f"   model_name: paraphrase-multilingual-mpnet-base-v2")
            elif working_model == "384_model_works":
                print(f"📝 graphrag_config.yaml에서 다음으로 변경:")
                print(f"   model_name: all-MiniLM-L6-v2")

        else:
            print(f"\n🔄 임베딩 재생성이 필요합니다")
            quick_rebuild_embeddings()

    print(f"\n✅ 진단 완료!")
