"""
의미적 유사도 기반 그래프 구축 모듈
Semantic Similarity Graph Construction Module using Longformer
"""

import json
import re
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import LongformerTokenizer, LongformerModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd


class SemanticSimilarityExtractor:
    """Longformer를 사용하여 논문 간 의미적 유사도를 계산하고 그래프를 구축하는 클래스"""

    def __init__(self, model_name="allenai/longformer-base-4096", device=None):
        """
        Args:
            model_name (str): 사용할 Longformer 모델명
            device (str): 사용할 디바이스 ('cuda', 'cpu', None for auto)
        """
        self.model_name = model_name
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"🤖 Loading Longformer model: {model_name}")
        print(f"💻 Using device: {self.device}")

        # 모델과 토크나이저 로드
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = LongformerModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # 최대 토큰 길이
        self.max_length = 4096

        print(f"✅ Model loaded successfully")

    def clean_text(self, text):
        """텍스트 전처리: 특수문자 제거 및 정리"""
        if not text:
            return ""

        # 특수문자 제거 (알파벳, 숫자, 기본 구두점만 유지)
        text = re.sub(r"[^\w\s\.\,\;\:\!\?\-\(\)]", " ", text)

        # 연속된 공백 정리
        text = re.sub(r"\s+", " ", text)

        # 앞뒤 공백 제거
        text = text.strip()

        return text

    def prepare_text_input(self, title, abstract):
        """제목과 초록을 결합하여 입력 텍스트 생성"""
        title_clean = self.clean_text(title) if title else ""
        abstract_clean = self.clean_text(abstract) if abstract else ""

        if abstract_clean:
            input_text = f"Title: {title_clean} Abstract: {abstract_clean}"
        else:
            input_text = f"Title: {title_clean}"

        return input_text

    def analyze_text_lengths(self, papers_metadata):
        """텍스트 길이 분포 분석"""
        print("📊 Analyzing text length distribution...")

        lengths = []
        texts_data = []

        for paper in papers_metadata:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")  # 초록이 메타데이터에 있는지 확인

            # 초록이 메타데이터에 없으면 빈 문자열
            if not abstract:
                abstract = ""

            input_text = self.prepare_text_input(title, abstract)

            # 토큰 수 계산
            tokens = self.tokenizer.encode(input_text, add_special_tokens=True)
            token_length = len(tokens)

            lengths.append(token_length)
            texts_data.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "input_text": input_text,
                    "token_length": token_length,
                }
            )

        lengths = np.array(lengths)

        print(f"📏 Text length statistics:")
        print(f"   Mean: {lengths.mean():.1f} tokens")
        print(f"   Median: {np.median(lengths):.1f} tokens")
        print(f"   95th percentile: {np.percentile(lengths, 95):.1f} tokens")
        print(f"   Max: {lengths.max()} tokens")
        print(f"   Papers > 4096 tokens: {sum(lengths > 4096)}")

        # 길이 분포 히스토그램 저장
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=50, alpha=0.7, edgecolor="black")
        plt.axvline(x=4096, color="red", linestyle="--", label="Longformer Max (4096)")
        plt.axvline(
            x=np.percentile(lengths, 95),
            color="orange",
            linestyle="--",
            label="95th percentile",
        )
        plt.xlabel("Token Length")
        plt.ylabel("Number of Papers")
        plt.title("Distribution of Text Lengths (Tokens)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        return texts_data, lengths

    def extract_embeddings(self, papers_metadata, batch_size=8):
        """논문들의 임베딩 벡터 추출"""
        print("🔍 Extracting semantic embeddings...")

        # 텍스트 길이 분석
        texts_data, lengths = self.analyze_text_lengths(papers_metadata)

        # 효과적인 최대 길이 설정 (95th percentile 또는 4096 중 작은 값)
        effective_max_length = min(int(np.percentile(lengths, 95)), self.max_length)
        print(f"📏 Using effective max length: {effective_max_length} tokens")

        embeddings = []
        paper_ids = []

        # GPU 메모리가 충분하므로 배치 크기 조정
        if self.device == "cuda":
            # A100 80GB면 더 큰 배치도 가능하지만 안전하게 설정
            batch_size = min(batch_size, 16)

        print(f"🚀 Processing with batch size: {batch_size}")

        # 배치 단위로 처리
        for i in tqdm(
            range(0, len(texts_data), batch_size), desc="Extracting embeddings"
        ):
            batch_texts = []
            batch_ids = []

            # 배치 데이터 준비
            for j in range(i, min(i + batch_size, len(texts_data))):
                text_data = texts_data[j]
                input_text = text_data["input_text"]

                batch_texts.append(input_text)
                batch_ids.append(f"paper_{j}")

            # 토크나이징
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=effective_max_length,
                return_tensors="pt",
            )

            # GPU로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 임베딩 추출
            with torch.no_grad():
                outputs = self.model(**inputs)

                # [CLS] 토큰의 임베딩 사용 (문서 전체 표현)
                cls_embeddings = outputs.last_hidden_state[
                    :, 0, :
                ]  # [batch_size, hidden_size]

                # CPU로 이동하여 저장
                batch_embeddings = cls_embeddings.cpu().numpy()
                embeddings.extend(batch_embeddings)
                paper_ids.extend(batch_ids)

        embeddings = np.array(embeddings)
        print(f"✅ Extracted embeddings: {embeddings.shape}")

        return embeddings, paper_ids, texts_data

    def compute_similarity_matrix(self, embeddings):
        """임베딩 간 코사인 유사도 행렬 계산"""
        print("📊 Computing cosine similarity matrix...")

        # 코사인 유사도 계산
        similarity_matrix = cosine_similarity(embeddings)

        print(f"✅ Similarity matrix computed: {similarity_matrix.shape}")
        print(
            f"📈 Similarity range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]"
        )

        return similarity_matrix

    def build_similarity_graph(self, similarity_matrix, paper_ids, top_k_percent=20):
        """유사도 행렬을 기반으로 방향 그래프 구축"""
        print(
            f"🔗 Building similarity graph (top {top_k_percent}% connections per paper)..."
        )

        n_papers = len(paper_ids)
        top_k = max(1, int(n_papers * top_k_percent / 100))  # 각 논문별 연결할 상위 k개

        similarity_graph = {}
        total_edges = 0

        for i, paper_id in enumerate(paper_ids):
            # 자기 자신 제외하고 유사도 계산
            similarities = similarity_matrix[i].copy()
            similarities[i] = -1  # 자기 자신은 제외

            # 상위 k개 선택
            top_indices = np.argsort(similarities)[-top_k:][::-1]  # 내림차순

            connections = []
            for j in top_indices:
                if similarities[j] > 0:  # 양수 유사도만
                    connections.append(
                        {
                            "target_paper": paper_ids[j],
                            "similarity": float(similarities[j]),
                        }
                    )
                    total_edges += 1

            similarity_graph[paper_id] = connections

        print(f"✅ Similarity graph built:")
        print(f"   📄 Papers: {n_papers}")
        print(f"   🔗 Total edges: {total_edges}")
        print(f"   📈 Average edges per paper: {total_edges/n_papers:.1f}")
        print(f"   🎯 Target edges per paper: {top_k}")

        return similarity_graph

    def analyze_similarity_distribution(self, similarity_matrix):
        """유사도 분포 분석"""
        print("📊 Analyzing similarity distribution...")

        # 대각선 제외 (자기 자신과의 유사도 제외)
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        similarities = similarity_matrix[mask]

        print(f"📈 Similarity statistics:")
        print(f"   Mean: {similarities.mean():.3f}")
        print(f"   Median: {np.median(similarities):.3f}")
        print(f"   Std: {similarities.std():.3f}")
        print(f"   Min: {similarities.min():.3f}")
        print(f"   Max: {similarities.max():.3f}")
        print(f"   80th percentile: {np.percentile(similarities, 80):.3f}")
        print(f"   90th percentile: {np.percentile(similarities, 90):.3f}")
        print(f"   95th percentile: {np.percentile(similarities, 95):.3f}")

        return similarities

    def save_similarity_graph(
        self, similarity_graph, similarity_matrix, paper_ids, texts_data, output_dir
    ):
        """유사도 그래프와 관련 데이터 저장"""
        output_dir = Path(output_dir)

        # 1. 그래프 구조 저장 (간단한 형태)
        graph_file = output_dir / "semantic_similarity_graph.json"
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(similarity_graph, f, ensure_ascii=False, indent=2)

        # 2. 유사도 행렬 저장 (numpy format)
        matrix_file = output_dir / "similarity_matrix.npy"
        np.save(matrix_file, similarity_matrix)

        # 3. 논문 ID 매핑 저장
        mapping_file = output_dir / "paper_id_mapping.json"
        paper_mapping = {
            "paper_ids": paper_ids,
            "id_to_index": {paper_id: i for i, paper_id in enumerate(paper_ids)},
        }
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(paper_mapping, f, ensure_ascii=False, indent=2)

        # 4. 텍스트 데이터 저장 (검증용)
        texts_file = output_dir / "processed_texts.json"
        with open(texts_file, "w", encoding="utf-8") as f:
            json.dump(texts_data, f, ensure_ascii=False, indent=2)

        # 5. 그래프 통계 저장
        stats = self.get_graph_statistics(similarity_graph, similarity_matrix)
        stats_file = output_dir / "similarity_graph_statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"💾 Similarity graph saved:")
        print(f"   🔗 Graph: {graph_file}")
        print(f"   📊 Matrix: {matrix_file}")
        print(f"   🗂️  Mapping: {mapping_file}")
        print(f"   📄 Texts: {texts_file}")
        print(f"   📈 Statistics: {stats_file}")

        return graph_file

    def get_graph_statistics(self, similarity_graph, similarity_matrix):
        """그래프 통계 정보 생성"""
        total_edges = sum(len(connections) for connections in similarity_graph.values())
        n_papers = len(similarity_graph)

        # 유사도 분포
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        similarities = similarity_matrix[mask]

        # 연결 수 분포
        edge_counts = [len(connections) for connections in similarity_graph.values()]

        stats = {
            "graph_info": {
                "total_papers": n_papers,
                "total_edges": total_edges,
                "average_edges_per_paper": total_edges / n_papers,
                "graph_density": total_edges / (n_papers * (n_papers - 1)),
            },
            "similarity_distribution": {
                "mean": float(similarities.mean()),
                "median": float(np.median(similarities)),
                "std": float(similarities.std()),
                "min": float(similarities.min()),
                "max": float(similarities.max()),
                "percentiles": {
                    "80th": float(np.percentile(similarities, 80)),
                    "90th": float(np.percentile(similarities, 90)),
                    "95th": float(np.percentile(similarities, 95)),
                },
            },
            "edge_distribution": {
                "mean_edges": float(np.mean(edge_counts)),
                "median_edges": float(np.median(edge_counts)),
                "min_edges": int(np.min(edge_counts)),
                "max_edges": int(np.max(edge_counts)),
            },
        }

        return stats

    def process_papers(self, papers_metadata, top_k_percent=20, batch_size=8):
        """전체 처리 파이프라인"""
        print("🚀 Starting semantic similarity analysis...")

        # 1. 임베딩 추출
        embeddings, paper_ids, texts_data = self.extract_embeddings(
            papers_metadata, batch_size=batch_size
        )

        # 2. 유사도 행렬 계산
        similarity_matrix = self.compute_similarity_matrix(embeddings)

        # 3. 유사도 분포 분석
        self.analyze_similarity_distribution(similarity_matrix)

        # 4. 그래프 구축
        similarity_graph = self.build_similarity_graph(
            similarity_matrix, paper_ids, top_k_percent=top_k_percent
        )

        return similarity_graph, similarity_matrix, paper_ids, texts_data


def main():
    """메인 실행 함수"""
    from src import PROCESSED_DIR

    # 통합 메타데이터 로드
    metadata_file = PROCESSED_DIR / "integrated_papers_metadata.json"

    if not metadata_file.exists():
        print("❌ Integrated papers metadata not found. Run main.py first.")
        return

    with open(metadata_file, "r", encoding="utf-8") as f:
        papers_metadata = json.load(f)

    print(f"📄 Loaded {len(papers_metadata)} papers metadata")

    # Semantic Similarity 추출기 초기화
    extractor = SemanticSimilarityExtractor()

    # 전체 처리
    similarity_graph, similarity_matrix, paper_ids, texts_data = (
        extractor.process_papers(
            papers_metadata,
            top_k_percent=20,  # 각 논문별 상위 20% 연결
            batch_size=8,  # A100 80GB에 맞게 조정 가능
        )
    )

    # 결과 저장
    output_file = extractor.save_similarity_graph(
        similarity_graph, similarity_matrix, paper_ids, texts_data, PROCESSED_DIR
    )

    print(f"✅ Semantic similarity analysis completed!")
    print(f"📁 Main output: {output_file}")

    return similarity_graph, output_file


if __name__ == "__main__":
    main()
