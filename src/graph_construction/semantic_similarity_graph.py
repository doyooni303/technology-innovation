"""
의미적 유사도 그래프 구축 모듈
Semantic Similarity Graph Construction Module

기존 semantic_similarity_extractor.py에서 생성한 데이터를
NetworkX 그래프로 변환하는 모듈
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm


class SemanticSimilarityGraphBuilder:
    """의미적 유사도 그래프를 구축하는 클래스"""

    def __init__(self, min_similarity=0.7):
        """
        Args:
            min_similarity (float): 그래프에 포함할 최소 유사도 임계값
        """
        self.min_similarity = min_similarity
        self.similarity_data = None
        self.papers_metadata = None

    # ✅ 올바른 코드
    def load_similarity_data(self, similarity_file, metadata_file):
        """기존에 추출된 similarity 데이터와 메타데이터 로드"""
        print("📂 Loading semantic similarity data and metadata...")

        # ✅ Semantic similarity JSON 데이터 로드
        with open(similarity_file, "r", encoding="utf-8") as f:
            self.similarity_data = json.load(f)

        # Papers 메타데이터 로드
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.papers_metadata = json.load(f)

        print(f"   📄 Similarity data loaded: {len(self.similarity_data)} papers")
        print(f"   📚 Metadata loaded: {len(self.papers_metadata)} papers")

        # 데이터 구조 검증
        if self.similarity_data:
            sample_paper = next(iter(self.similarity_data.values()))
            if sample_paper:
                sample_connection = sample_paper[0]
                required_fields = ["target_paper", "similarity"]
                missing_fields = [
                    field for field in required_fields if field not in sample_connection
                ]
                if missing_fields:
                    print(
                        f"⚠️  Warning: Missing fields in similarity data: {missing_fields}"
                    )
                else:
                    print("✅ Similarity data structure validated")

    def analyze_similarity_data(self):
        """Similarity 데이터 분석"""
        print("📊 Analyzing semantic similarity data...")

        total_connections = sum(
            len(connections) for connections in self.similarity_data.values()
        )
        papers_with_connections = sum(
            1 for connections in self.similarity_data.values() if connections
        )

        all_similarities = []
        for connections in self.similarity_data.values():
            for conn in connections:
                all_similarities.append(conn["similarity"])

        print(f"   🔗 Total similarity connections: {total_connections}")
        print(f"   📄 Papers with outgoing connections: {papers_with_connections}")
        print(
            f"   📈 Avg connections per paper: {total_connections/len(self.similarity_data):.1f}"
        )

        if all_similarities:
            print(f"   📊 Similarity distribution:")
            print(f"      Min: {min(all_similarities):.4f}")
            print(f"      Max: {max(all_similarities):.4f}")
            print(f"      Mean: {np.mean(all_similarities):.4f}")
            print(f"      Median: {np.median(all_similarities):.4f}")

    def build_similarity_graph(self):
        """Similarity 데이터로부터 NetworkX 방향 그래프 구축"""
        print(f"🔗 Building directed semantic similarity graph...")

        # ✅ 방향 그래프 생성 (각 논문의 "선호" 관계를 표현)
        G = nx.DiGraph()  # ← DiGraph 사용

        # 논문 ID to 메타데이터 매핑 생성
        paper_metadata_map = {}
        for i, paper in enumerate(self.papers_metadata):
            paper_id = f"paper_{i}"
            paper_metadata_map[paper_id] = paper

        # 모든 논문을 노드로 추가
        for paper_id, metadata in paper_metadata_map.items():

            G.add_node(
                paper_id,
                node_type="paper",
                title=metadata.get("title", ""),
                abstract=metadata.get("abstract", ""),  # ✅ Abstract 추출
                authors=metadata.get("authors", []),
                year=metadata.get("year", ""),
                journal=metadata.get("journal", ""),
                keywords=metadata.get("keywords", []),
                has_pdf=metadata.get("has_pdf", False),
                has_abstract=metadata.get("has_abstract", False),
                abstract_length=len(metadata.get("abstract", "")),  # ✅ Abstract 길이
            )

        # ✅ 방향성 있는 유사도 엣지 추가
        edges_added = 0
        similarity_weights = []

        for source_paper, connections in self.similarity_data.items():
            if not connections:
                continue

            for connection in connections:
                target_paper = connection["target_paper"]
                similarity = connection["similarity"]

                # 자기 자신과의 연결 제외
                if source_paper != target_paper:
                    # 양쪽 논문이 모두 그래프에 있는 경우만 엣지 추가
                    if G.has_node(source_paper) and G.has_node(target_paper):
                        # ✅ 방향성 있는 엣지 (source → target)
                        G.add_edge(
                            source_paper,
                            target_paper,
                            edge_type="semantic_similarity",
                            similarity=float(similarity),
                            weight=float(similarity),
                            rank=len(similarity_weights) + 1,
                        )  # 순위 정보 추가

                        edges_added += 1
                        similarity_weights.append(similarity)

        print(f"✅ Directed semantic similarity graph constructed:")
        print(f"   📄 Nodes (papers): {G.number_of_nodes()}")
        print(f"   🔗 Directed edges (similarities): {G.number_of_edges()}")
        print(f"   📈 Graph density: {nx.density(G):.6f}")
        print(
            f"   🎯 Average out-degree: {G.number_of_edges()/G.number_of_nodes():.1f}"
        )

        return G

    def analyze_graph_properties(self, G):
        """방향 그래프 속성 분석"""
        print("📈 Analyzing directed semantic similarity graph properties...")

        # 기본 통계
        stats = {
            "basic_stats": {
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "density": float(nx.density(G)),
                "is_weakly_connected": nx.is_weakly_connected(G),
                "is_strongly_connected": nx.is_strongly_connected(G),
            }
        }

        # 방향 그래프 연결성 분석
        if not nx.is_strongly_connected(G):
            weak_components = list(nx.weakly_connected_components(G))
            strong_components = list(nx.strongly_connected_components(G))

            stats["connectivity"] = {
                "num_weak_components": len(weak_components),
                "num_strong_components": len(strong_components),
                "largest_weak_component": len(max(weak_components, key=len)),
                "largest_strong_component": len(max(strong_components, key=len)),
            }

        # ✅ 방향 그래프 중심성 분석 (수정된 용어)
        if G.number_of_nodes() > 0:
            in_degree_centrality = nx.in_degree_centrality(G)  # 많이 선택받는 논문
            out_degree_centrality = nx.out_degree_centrality(G)  # 많이 선택하는 논문

            # 가장 "인기 있는" 논문들 (다른 논문들이 유사하다고 선택)
            most_similar_to = sorted(
                in_degree_centrality.items(), key=lambda x: x[1], reverse=True
            )[:10]

            # 가장 "포괄적인" 논문들 (다른 논문들과 유사점을 많이 찾음)
            most_similar = sorted(
                out_degree_centrality.items(), key=lambda x: x[1], reverse=True
            )[:10]

            stats["centrality"] = {
                "most_similar_to_papers": [
                    (k, float(v)) for k, v in most_similar_to
                ],  # ✅ 수정
                "most_similar_papers": [
                    (k, float(v)) for k, v in most_similar
                ],  # ✅ 수정
            }

            # 실제 연결 수로도 정렬
            in_degrees = dict(G.in_degree())
            out_degrees = dict(G.out_degree())

            stats["degree_stats"] = {
                "top_similar_to_by_count": sorted(
                    in_degrees.items(), key=lambda x: x[1], reverse=True  # ✅ 수정
                )[:10],
                "top_similar_by_count": sorted(
                    out_degrees.items(), key=lambda x: x[1], reverse=True  # ✅ 수정
                )[:10],
                "avg_in_degree": float(np.mean(list(in_degrees.values()))),
                "avg_out_degree": float(np.mean(list(out_degrees.values()))),
                "max_in_degree": max(in_degrees.values()),
                "max_out_degree": max(out_degrees.values()),
            }

        return stats

    def analyze_similarity_distribution(self, G):
        """유사도 분포 분석"""
        similarities = []
        for edge in G.edges():
            similarity = G.edges[edge]["similarity"]
            similarities.append(similarity)

        if not similarities:
            return {}

        distribution_stats = {
            "mean": float(np.mean(similarities)),
            "median": float(np.median(similarities)),
            "std": float(np.std(similarities)),
            "min": float(min(similarities)),
            "max": float(max(similarities)),
            "percentiles": {
                "25th": float(np.percentile(similarities, 25)),
                "75th": float(np.percentile(similarities, 75)),
                "90th": float(np.percentile(similarities, 90)),
                "95th": float(np.percentile(similarities, 95)),
            },
        }

        return distribution_stats

    def create_paper_info_map(self, G):
        """논문 정보를 쉽게 조회할 수 있는 맵 생성"""
        paper_info = {}

        for node in G.nodes():
            node_data = G.nodes[node]
            paper_info[node] = {
                "title": node_data.get("title", ""),
                "abstract": node_data.get("abstract", ""),
                "authors": ", ".join(node_data.get("authors", [])),
                "year": node_data.get("year", ""),
                "journal": node_data.get("journal", ""),
                "keywords": ", ".join(node_data.get("keywords", [])),
                "degree": G.degree(node),  # 연결된 논문 수
                "has_pdf": node_data.get("has_pdf", False),
                "has_abstract": node_data.get("has_abstract", False),
                "abstract_length": node_data.get(
                    "abstract_length", 0
                ),  # ✅ Abstract 길이
            }

        return paper_info

    def save_similarity_graph_and_analysis(self, G, stats, output_dir):
        """Similarity 그래프와 분석 결과 저장 (XML 호환성 개선)"""
        output_dir = Path(output_dir)

        # 1. NetworkX 그래프를 JSON으로 저장 (GraphRAG용)
        graph_data = {"nodes": [], "edges": []}

        # 노드 정보
        for node in G.nodes():
            node_data = G.nodes[node].copy()
            node_data["id"] = node
            # 리스트 타입 필드들을 JSON 직렬화 가능하게 변환
            if "authors" in node_data and isinstance(node_data["authors"], list):
                node_data["authors"] = list(node_data["authors"])
            if "keywords" in node_data and isinstance(node_data["keywords"], list):
                node_data["keywords"] = list(node_data["keywords"])
            graph_data["nodes"].append(node_data)

        # 엣지 정보
        for edge in G.edges():
            edge_data = G.edges[edge].copy()
            edge_data["source"] = edge[0]
            edge_data["target"] = edge[1]
            # numpy 타입을 Python 기본 타입으로 변환
            for key, value in edge_data.items():
                if isinstance(value, np.floating):
                    edge_data[key] = float(value)
            graph_data["edges"].append(edge_data)

        # JSON 파일로 저장
        graph_file = output_dir / "semantic_similarity_network_graph.json"
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        # 2. ✅ GraphML 파일로 저장 (XML 호환성 개선)
        try:
            G_graphml = G.copy()

            # ✅ XML 호환 문자열 정제 함수
            def clean_xml_string(text):
                if not isinstance(text, str):
                    text = str(text)

                # NULL 바이트 및 제어 문자 제거
                import re

                text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

                # XML 특수 문자 이스케이프
                text = text.replace("&", "&amp;")
                text = text.replace("<", "&lt;")
                text = text.replace(">", "&gt;")
                text = text.replace('"', "&quot;")
                text = text.replace("'", "&apos;")

                return text

            # 노드 속성을 GraphML 호환 형태로 변환
            for node in G_graphml.nodes():
                # 리스트 타입 속성들을 문자열로 변환
                if "authors" in G_graphml.nodes[node]:
                    authors_list = G_graphml.nodes[node]["authors"]
                    if isinstance(authors_list, (list, set)):
                        cleaned_authors = [
                            clean_xml_string(str(a)) for a in authors_list
                        ]
                        G_graphml.nodes[node]["authors"] = ";".join(cleaned_authors)

                if "keywords" in G_graphml.nodes[node]:
                    keywords_list = G_graphml.nodes[node]["keywords"]
                    if isinstance(keywords_list, (list, set)):
                        cleaned_keywords = [
                            clean_xml_string(str(k)) for k in keywords_list
                        ]
                        G_graphml.nodes[node]["keywords"] = ";".join(cleaned_keywords)

                # ✅ Abstract와 기타 문자열 필드들 정제
                for attr_name, attr_value in G_graphml.nodes[node].items():
                    if isinstance(attr_value, str):
                        # Abstract, title 등 문자열 필드들 정제
                        G_graphml.nodes[node][attr_name] = clean_xml_string(attr_value)
                    elif isinstance(attr_value, (list, set, dict)):
                        if isinstance(attr_value, dict):
                            G_graphml.nodes[node][attr_name] = clean_xml_string(
                                json.dumps(attr_value)
                            )
                        else:
                            cleaned_list = [
                                clean_xml_string(str(v)) for v in attr_value
                            ]
                            G_graphml.nodes[node][attr_name] = ";".join(cleaned_list)
                    elif not isinstance(attr_value, (int, float, bool)):
                        G_graphml.nodes[node][attr_name] = clean_xml_string(
                            str(attr_value)
                        )

            # 엣지 속성도 정제
            for edge in G_graphml.edges():
                for attr_name, attr_value in G_graphml.edges[edge].items():
                    if isinstance(attr_value, str):
                        G_graphml.edges[edge][attr_name] = clean_xml_string(attr_value)
                    elif isinstance(attr_value, (list, set, dict)):
                        if isinstance(attr_value, dict):
                            G_graphml.edges[edge][attr_name] = clean_xml_string(
                                json.dumps(attr_value)
                            )
                        else:
                            cleaned_list = [
                                clean_xml_string(str(v)) for v in attr_value
                            ]
                            G_graphml.edges[edge][attr_name] = ";".join(cleaned_list)
                    elif not isinstance(attr_value, (int, float, bool)):
                        G_graphml.edges[edge][attr_name] = clean_xml_string(
                            str(attr_value)
                        )

            graphml_file = output_dir / "semantic_similarity_network_graph.graphml"
            nx.write_graphml(
                G_graphml, graphml_file, encoding="utf-8", prettyprint=True
            )
            print(f"   🔗 Graph (GraphML): {graphml_file}")

        except Exception as e:
            print(f"⚠️  GraphML 저장 중 오류 발생: {e}")
            print(f"📄 JSON 파일은 정상적으로 저장되었습니다: {graph_file}")
            graphml_file = None

        # 3. 분석 결과 저장
        stats_file = output_dir / "semantic_similarity_network_analysis.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # 4. 논문 정보 테이블 저장
        paper_info = self.create_paper_info_map(G)
        paper_info_file = output_dir / "semantic_similarity_papers_info.json"
        with open(paper_info_file, "w", encoding="utf-8") as f:
            json.dump(paper_info, f, ensure_ascii=False, indent=2)

        # 5. CSV 형태의 엣지 리스트 저장
        edge_list = []
        for edge in G.edges():
            edge_info = G.edges[edge].copy()
            edge_info.update(
                {
                    "source_paper": edge[0],
                    "target_paper": edge[1],
                    "source_title": G.nodes[edge[0]].get("title", ""),
                    "target_title": G.nodes[edge[1]].get("title", ""),
                    "source_year": G.nodes[edge[0]].get("year", ""),
                    "target_year": G.nodes[edge[1]].get("year", ""),
                    # ✅ Abstract 정보 추가
                    "source_has_abstract": G.nodes[edge[0]].get("has_abstract", False),
                    "target_has_abstract": G.nodes[edge[1]].get("has_abstract", False),
                    "source_abstract_length": G.nodes[edge[0]].get(
                        "abstract_length", 0
                    ),
                    "target_abstract_length": G.nodes[edge[1]].get(
                        "abstract_length", 0
                    ),
                }
            )
            # numpy 타입 변환
            for key, value in edge_info.items():
                if isinstance(value, np.floating):
                    edge_info[key] = float(value)
            edge_list.append(edge_info)

        edge_df = pd.DataFrame(edge_list)
        edge_file = output_dir / "semantic_similarity_network_edges.csv"
        edge_df.to_csv(edge_file, index=False, encoding="utf-8")

        print(f"💾 Semantic similarity graph results saved:")
        print(f"   🔗 Graph (JSON): {graph_file}")
        print(f"   📊 Analysis: {stats_file}")
        print(f"   📄 Paper Info: {paper_info_file}")
        print(f"   📈 Edge List: {edge_file}")

        return graph_file

    def process_similarity_data(self, similarity_file, metadata_file):
        """전체 처리 파이프라인"""
        print("🚀 Starting semantic similarity graph construction...")

        # 1. 데이터 로드
        self.load_similarity_data(similarity_file, metadata_file)

        # 2. 데이터 분석
        self.analyze_similarity_data()

        # 3. 그래프 구축
        G = self.build_similarity_graph()

        # 4. 그래프 분석
        stats = self.analyze_graph_properties(G)

        return G, stats


def main():
    """메인 실행 함수"""
    from src import RAW_EXTRACTIONS_DIR, GRAPHS_DIR  # ✅ 변경

    # ✅ 입력 파일들 (raw_extractions에서)
    similarity_file = RAW_EXTRACTIONS_DIR / "semantic_similarity_graph.json"
    metadata_file = RAW_EXTRACTIONS_DIR / "integrated_papers_metadata.json"

    if not similarity_file.exists():
        print(f"❌ Semantic similarity data not found: {similarity_file}")
        print("Please run semantic_similarity_extractor.py first.")
        return

    if not metadata_file.exists():
        print(f"❌ Metadata not found: {metadata_file}")
        print("Please run data processing pipeline first.")
        return

    print(f"📂 Similarity file: {similarity_file}")
    print(f"📂 Metadata file: {metadata_file}")

    # Semantic Similarity Graph Builder 초기화
    builder = SemanticSimilarityGraphBuilder(min_similarity=0.7)

    # 전체 처리
    G, stats = builder.process_similarity_data(similarity_file, metadata_file)

    # ✅ 결과 저장 (graphs 폴더에)
    output_file = builder.save_similarity_graph_and_analysis(G, stats, GRAPHS_DIR)

    print(f"\n✅ Semantic similarity graph construction completed!")
    print(f"📁 Main output: {output_file}")

    return G, output_file


if __name__ == "__main__":
    main()
