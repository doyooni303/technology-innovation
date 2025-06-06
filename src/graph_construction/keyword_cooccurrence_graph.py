"""
키워드 동시 출현 그래프 구축 모듈
Keyword Co-occurrence Graph Construction Module
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
import re


class KeywordCooccurrenceGraphBuilder:
    """키워드 동시 출현 그래프를 구축하는 클래스"""

    def __init__(
        self, min_keyword_freq=2, min_cooccurrence=2, max_keywords_per_paper=50
    ):
        """
        Args:
            min_keyword_freq (int): 포함할 키워드의 최소 출현 빈도
            min_cooccurrence (int): 그래프에 포함할 최소 동시 출현 횟수
            max_keywords_per_paper (int): 논문당 최대 키워드 수 (노이즈 제거)
        """
        self.min_keyword_freq = min_keyword_freq
        self.min_cooccurrence = min_cooccurrence
        self.max_keywords_per_paper = max_keywords_per_paper

        # 결과 저장용
        self.keyword_freq = Counter()
        self.cooccurrence_matrix = defaultdict(lambda: defaultdict(int))
        self.keyword_to_papers = defaultdict(set)
        self.paper_to_keywords = {}

    def clean_keyword(self, keyword):
        """키워드 정제 함수"""
        if not keyword or not isinstance(keyword, str):
            return None

        # 소문자 변환
        keyword = keyword.lower().strip()

        # 너무 짧거나 긴 키워드 제거
        if len(keyword) < 2 or len(keyword) > 50:
            return None

        # 숫자만 있는 키워드 제거
        if keyword.isdigit():
            return None

        # 특수 문자만 있는 키워드 제거
        if re.match(r"^[^a-zA-Z0-9]+$", keyword):
            return None

        # 일반적인 불용어 제거
        stopwords = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "this",
            "that",
            "these",
            "those",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "paper",
            "study",
            "research",
            "method",
            "approach",
            "analysis",
            "result",
            "conclusion",
        }

        if keyword in stopwords:
            return None

        return keyword

    def extract_keywords_from_papers(self, papers_metadata):
        """논문들에서 키워드 추출 및 전처리"""
        print("🔍 Extracting and cleaning keywords from papers...")

        valid_papers = 0
        total_raw_keywords = 0
        total_clean_keywords = 0

        for i, paper in enumerate(tqdm(papers_metadata, desc="Processing papers")):
            paper_id = f"paper_{i}"
            title = paper.get("title", "")
            keywords = paper.get("keywords", [])

            if not keywords:
                continue

            # 키워드 정제
            clean_keywords = []
            total_raw_keywords += len(keywords)

            for keyword in keywords:
                clean_kw = self.clean_keyword(keyword)
                if clean_kw:
                    clean_keywords.append(clean_kw)

            # 키워드가 너무 많으면 제한 (품질 확보)
            if len(clean_keywords) > self.max_keywords_per_paper:
                # 빈도순으로 정렬하여 상위 키워드만 선택
                clean_keywords = clean_keywords[: self.max_keywords_per_paper]

            if clean_keywords:
                self.paper_to_keywords[paper_id] = {
                    "title": title,
                    "keywords": clean_keywords,
                    "original_keyword_count": len(keywords),
                    "clean_keyword_count": len(clean_keywords),
                }

                # 키워드 빈도 계산
                for keyword in clean_keywords:
                    self.keyword_freq[keyword] += 1
                    self.keyword_to_papers[keyword].add(paper_id)

                valid_papers += 1
                total_clean_keywords += len(clean_keywords)

        print(f"✅ Keyword extraction completed:")
        print(f"   📄 Papers with keywords: {valid_papers}/{len(papers_metadata)}")
        print(f"   🔤 Raw keywords: {total_raw_keywords}")
        print(f"   ✨ Clean keywords: {total_clean_keywords}")
        print(f"   📊 Unique keywords: {len(self.keyword_freq)}")
        print(f"   📈 Avg keywords per paper: {total_clean_keywords/valid_papers:.1f}")

    def filter_keywords_by_frequency(self):
        """빈도 기준으로 키워드 필터링"""
        print(f"🔍 Filtering keywords (min frequency: {self.min_keyword_freq})...")

        # 최소 빈도 이상의 키워드만 선택
        frequent_keywords = {
            kw
            for kw, freq in self.keyword_freq.items()
            if freq >= self.min_keyword_freq
        }

        print(f"   📊 Keywords before filtering: {len(self.keyword_freq)}")
        print(f"   ✅ Keywords after filtering: {len(frequent_keywords)}")

        # 논문 데이터에서도 필터링된 키워드만 유지
        filtered_papers = {}
        for paper_id, paper_data in self.paper_to_keywords.items():
            filtered_keywords = [
                kw for kw in paper_data["keywords"] if kw in frequent_keywords
            ]

            if filtered_keywords:  # 키워드가 남아있는 논문만 유지
                filtered_papers[paper_id] = paper_data.copy()
                filtered_papers[paper_id]["keywords"] = filtered_keywords
                filtered_papers[paper_id]["filtered_keyword_count"] = len(
                    filtered_keywords
                )

        self.paper_to_keywords = filtered_papers

        # 키워드 빈도도 업데이트
        self.keyword_freq = {
            kw: freq
            for kw, freq in self.keyword_freq.items()
            if kw in frequent_keywords
        }

        print(f"   📄 Papers remaining: {len(self.paper_to_keywords)}")

        return frequent_keywords

    def compute_cooccurrence_matrix(self):
        """키워드 동시 출현 행렬 계산"""
        print("📊 Computing keyword co-occurrence matrix...")

        total_pairs = 0

        for paper_id, paper_data in tqdm(
            self.paper_to_keywords.items(), desc="Computing co-occurrences"
        ):
            keywords = paper_data["keywords"]

            # 논문 내 모든 키워드 쌍의 동시 출현 계산
            for kw1, kw2 in combinations(keywords, 2):
                # 사전순으로 정렬하여 중복 방지
                if kw1 > kw2:
                    kw1, kw2 = kw2, kw1

                self.cooccurrence_matrix[kw1][kw2] += 1
                total_pairs += 1

        print(f"✅ Co-occurrence computation completed:")
        print(f"   🔗 Total keyword pairs processed: {total_pairs}")
        print(
            f"   📊 Unique co-occurring pairs: {sum(len(v) for v in self.cooccurrence_matrix.values())}"
        )

    def build_cooccurrence_graph(self):
        """동시 출현 그래프 구축"""
        print(
            f"🔗 Building co-occurrence graph (min co-occurrence: {self.min_cooccurrence})..."
        )

        # NetworkX 그래프 생성
        G = nx.Graph()

        # 키워드 노드 추가 (빈도 정보 포함)
        for keyword, freq in self.keyword_freq.items():
            G.add_node(
                keyword,
                node_type="keyword",
                frequency=freq,
                papers=list(self.keyword_to_papers[keyword]),
            )

        # 동시 출현 엣지 추가
        edges_added = 0
        cooccurrence_weights = []

        for kw1, cooccurrences in self.cooccurrence_matrix.items():
            for kw2, count in cooccurrences.items():
                if count >= self.min_cooccurrence:
                    # 정규화된 가중치 계산 (Jaccard similarity)
                    kw1_freq = self.keyword_freq[kw1]
                    kw2_freq = self.keyword_freq[kw2]
                    jaccard_sim = count / (kw1_freq + kw2_freq - count)

                    G.add_edge(
                        kw1,
                        kw2,
                        weight=count,
                        jaccard_similarity=jaccard_sim,
                        normalized_weight=count / max(kw1_freq, kw2_freq),
                    )

                    edges_added += 1
                    cooccurrence_weights.append(count)

        print(f"✅ Graph construction completed:")
        print(f"   🔗 Nodes (keywords): {G.number_of_nodes()}")
        print(f"   📊 Edges (co-occurrences): {G.number_of_edges()}")
        print(f"   📈 Graph density: {nx.density(G):.4f}")

        if cooccurrence_weights:
            print(f"   🎯 Co-occurrence weight stats:")
            print(f"      Min: {min(cooccurrence_weights)}")
            print(f"      Max: {max(cooccurrence_weights)}")
            print(f"      Mean: {np.mean(cooccurrence_weights):.1f}")
            print(f"      Median: {np.median(cooccurrence_weights):.1f}")

        return G

    def analyze_graph_properties(self, G):
        """그래프 속성 분석"""
        print("📈 Analyzing graph properties...")

        # 기본 통계
        stats = {
            "basic_stats": {
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "density": float(nx.density(G)),  # ✅ float 변환
                "is_connected": nx.is_connected(G),
            }
        }

        # 연결 성분 분석
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            stats["connectivity"] = {
                "num_components": len(components),
                "largest_component_size": len(max(components, key=len)),
                "component_sizes": [len(comp) for comp in components],
            }

        # 중심성 분석 (상위 키워드들만)
        if G.number_of_nodes() > 0:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(
                G, k=min(1000, G.number_of_nodes())
            )
            eigenvector_centrality = nx.eigenvector_centrality(
                G, max_iter=1000, tol=1e-3
            )

            # ✅ numpy 타입을 float로 변환
            stats["centrality"] = {
                "top_degree": [
                    (k, float(v))
                    for k, v in sorted(
                        degree_centrality.items(), key=lambda x: x[1], reverse=True
                    )[:10]
                ],
                "top_betweenness": [
                    (k, float(v))
                    for k, v in sorted(
                        betweenness_centrality.items(), key=lambda x: x[1], reverse=True
                    )[:10]
                ],
                "top_eigenvector": [
                    (k, float(v))
                    for k, v in sorted(
                        eigenvector_centrality.items(), key=lambda x: x[1], reverse=True
                    )[:10]
                ],
            }

        # 클러스터링 계수
        if G.number_of_nodes() > 0:
            clustering_coeffs = nx.clustering(G)
            stats["clustering"] = {
                "average_clustering": float(nx.average_clustering(G)),  # ✅ float 변환
                "global_clustering": float(nx.transitivity(G)),  # ✅ float 변환
                "top_clustered_keywords": [
                    (k, float(v))
                    for k, v in sorted(
                        clustering_coeffs.items(), key=lambda x: x[1], reverse=True
                    )[:10]
                ],  # ✅ float 변환
            }

        return stats

    def save_graph_and_analysis(self, G, stats, output_dir):
        """그래프와 분석 결과 저장"""
        output_dir = Path(output_dir)

        # 1. NetworkX 그래프를 JSON으로 저장 (GraphRAG용)
        graph_data = {"nodes": [], "edges": []}

        # 노드 정보
        for node in G.nodes():
            node_data = G.nodes[node].copy()
            node_data["id"] = node
            # papers 리스트를 문자열로 변환 (JSON 직렬화를 위해)
            if "papers" in node_data:
                node_data["papers"] = list(node_data["papers"])
            graph_data["nodes"].append(node_data)

        # 엣지 정보
        for edge in G.edges():
            edge_data = G.edges[edge].copy()
            edge_data["source"] = edge[0]
            edge_data["target"] = edge[1]
            graph_data["edges"].append(edge_data)

        # JSON 파일로 저장
        graph_file = output_dir / "keyword_cooccurrence_graph.json"
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        # 2. GraphML 파일로 저장 (Gephi, Cytoscape 등에서 사용)
        # GraphML은 복잡한 데이터 타입을 지원하지 않으므로 별도 처리
        try:
            # GraphML 호환을 위해 그래프 복사 및 속성 변환
            G_graphml = G.copy()

            # 노드 속성을 GraphML 호환 형태로 변환
            for node in G_graphml.nodes():
                # papers 리스트를 쉼표로 구분된 문자열로 변환
                if "papers" in G_graphml.nodes[node]:
                    papers_list = G_graphml.nodes[node]["papers"]
                    if isinstance(papers_list, (list, set)):
                        G_graphml.nodes[node]["papers"] = ";".join(
                            str(p) for p in papers_list
                        )

                # 기타 복잡한 타입들을 문자열로 변환
                for attr_name, attr_value in G_graphml.nodes[node].items():
                    if isinstance(attr_value, (list, set, dict)):
                        if isinstance(attr_value, dict):
                            G_graphml.nodes[node][attr_name] = json.dumps(attr_value)
                        else:
                            G_graphml.nodes[node][attr_name] = ";".join(
                                str(v) for v in attr_value
                            )
                    elif not isinstance(attr_value, (str, int, float, bool)):
                        G_graphml.nodes[node][attr_name] = str(attr_value)

            # 엣지 속성도 확인
            for edge in G_graphml.edges():
                for attr_name, attr_value in G_graphml.edges[edge].items():
                    if isinstance(attr_value, (list, set, dict)):
                        if isinstance(attr_value, dict):
                            G_graphml.edges[edge][attr_name] = json.dumps(attr_value)
                        else:
                            G_graphml.edges[edge][attr_name] = ";".join(
                                str(v) for v in attr_value
                            )
                    elif not isinstance(attr_value, (str, int, float, bool)):
                        G_graphml.edges[edge][attr_name] = str(attr_value)

            graphml_file = output_dir / "keyword_cooccurrence_graph.graphml"
            nx.write_graphml(G_graphml, graphml_file)

        except Exception as e:
            print(f"⚠️  GraphML 저장 중 오류 발생: {e}")
            print(f"📄 JSON 파일은 정상적으로 저장되었습니다: {graph_file}")
            graphml_file = None

        # 3. 통계 분석 결과 저장
        stats_file = output_dir / "keyword_cooccurrence_analysis.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # 4. 키워드 빈도 데이터 저장
        keyword_data = {
            "keyword_frequencies": dict(self.keyword_freq),
            "keyword_to_papers": {
                kw: list(papers) for kw, papers in self.keyword_to_papers.items()
            },
            "total_keywords": len(self.keyword_freq),
            "min_frequency_threshold": self.min_keyword_freq,
        }

        keywords_file = output_dir / "keyword_frequencies.json"
        with open(keywords_file, "w", encoding="utf-8") as f:
            json.dump(keyword_data, f, ensure_ascii=False, indent=2)

        # 5. 동시 출현 행렬 저장 (CSV 형태)
        cooccurrence_data = []
        for kw1, cooccurrences in self.cooccurrence_matrix.items():
            for kw2, count in cooccurrences.items():
                if count >= self.min_cooccurrence:
                    cooccurrence_data.append(
                        {
                            "keyword1": kw1,
                            "keyword2": kw2,
                            "cooccurrence_count": count,
                            "jaccard_similarity": count
                            / (self.keyword_freq[kw1] + self.keyword_freq[kw2] - count),
                        }
                    )

        cooccurrence_df = pd.DataFrame(cooccurrence_data)
        cooccurrence_file = output_dir / "keyword_cooccurrence_matrix.csv"
        cooccurrence_df.to_csv(cooccurrence_file, index=False, encoding="utf-8")

        print(f"💾 Results saved:")
        print(f"   🔗 Graph (JSON): {graph_file}")
        if graphml_file:
            print(f"   🔗 Graph (GraphML): {graphml_file}")
        print(f"   📊 Analysis: {stats_file}")
        print(f"   🔤 Keywords: {keywords_file}")
        print(f"   📈 Co-occurrence Matrix: {cooccurrence_file}")

        return graph_file

    def create_visualization_data(self, G, stats):
        """시각화용 데이터 생성"""

        # 상위 키워드들의 서브그래프 추출 (시각화용)
        top_keywords = [kw for kw, freq in Counter(self.keyword_freq).most_common(50)]
        subgraph = G.subgraph(top_keywords).copy()

        # 위치 계산 (spring layout)
        pos = nx.spring_layout(subgraph, k=1, iterations=50)

        visualization_data = {
            "subgraph_nodes": [],
            "subgraph_edges": [],
            "layout_positions": {
                k: [float(v[0]), float(v[1])] for k, v in pos.items()
            },  # ✅ numpy float 변환
        }

        # 노드 데이터 (크기는 빈도, 색상은 중심성)
        degree_centrality = nx.degree_centrality(subgraph)

        for node in subgraph.nodes():
            node_info = {
                "id": node,
                "frequency": self.keyword_freq[node],
                "degree_centrality": float(
                    degree_centrality.get(node, 0)
                ),  # ✅ float 변환
                "degree": subgraph.degree(node),
                "x": float(pos[node][0]),  # ✅ numpy float를 Python float로 변환
                "y": float(pos[node][1]),  # ✅ numpy float를 Python float로 변환
            }
            visualization_data["subgraph_nodes"].append(node_info)

        # 엣지 데이터
        for edge in subgraph.edges():
            edge_info = {
                "source": edge[0],
                "target": edge[1],
                "weight": subgraph.edges[edge]["weight"],
                "jaccard_similarity": float(
                    subgraph.edges[edge]["jaccard_similarity"]
                ),  # ✅ float 변환
            }
            visualization_data["subgraph_edges"].append(edge_info)

        return visualization_data

    def process_papers(self, papers_metadata):
        """전체 처리 파이프라인"""
        print("🚀 Starting keyword co-occurrence graph construction...")

        # 1. 키워드 추출 및 전처리
        self.extract_keywords_from_papers(papers_metadata)

        # 2. 빈도 기준 필터링
        frequent_keywords = self.filter_keywords_by_frequency()

        # 3. 동시 출현 행렬 계산
        self.compute_cooccurrence_matrix()

        # 4. 그래프 구축
        G = self.build_cooccurrence_graph()

        # 5. 그래프 분석
        stats = self.analyze_graph_properties(G)

        # 6. 시각화 데이터 생성
        viz_data = self.create_visualization_data(G, stats)
        stats["visualization_data"] = viz_data

        return G, stats


def main():
    """메인 실행 함수"""
    from src import RAW_EXTRACTIONS_DIR, GRAPHS_DIR

    # 통합 메타데이터 로드
    metadata_file = RAW_EXTRACTIONS_DIR / "integrated_papers_metadata.json"

    if not metadata_file.exists():
        print("❌ Integrated papers metadata not found. Run main.py first.")
        return

    with open(metadata_file, "r", encoding="utf-8") as f:
        papers_metadata = json.load(f)

    print(f"📄 Loaded {len(papers_metadata)} papers metadata")

    # Keyword Co-occurrence Graph Builder 초기화
    builder = KeywordCooccurrenceGraphBuilder(
        min_keyword_freq=2,  # 최소 2번 이상 출현한 키워드만 포함
        min_cooccurrence=2,  # 최소 2번 이상 동시 출현한 쌍만 엣지로 연결
        max_keywords_per_paper=30,  # 논문당 최대 30개 키워드 (품질 확보)
    )

    # 전체 처리
    G, stats = builder.process_papers(papers_metadata)

    # 결과 저장
    output_file = builder.save_graph_and_analysis(G, stats, GRAPHS_DIR)

    print(f"\n✅ Keyword co-occurrence graph construction completed!")
    print(f"📁 Main output: {output_file}")

    # 요약 통계 출력
    print(f"\n📊 Final Summary:")
    print(f"   🔤 Total unique keywords: {len(builder.keyword_freq)}")
    print(f"   🔗 Graph nodes: {G.number_of_nodes()}")
    print(f"   📊 Graph edges: {G.number_of_edges()}")
    print(f"   🎯 Graph density: {nx.density(G):.4f}")

    if stats.get("centrality"):
        top_keywords = stats["centrality"]["top_degree"][:5]
        print(f"   🏆 Top 5 central keywords:")
        for i, (kw, centrality) in enumerate(top_keywords):
            freq = builder.keyword_freq[kw]
            print(f"      {i+1}. {kw} (freq: {freq}, centrality: {centrality:.3f})")

    return G, output_file


if __name__ == "__main__":
    main()
