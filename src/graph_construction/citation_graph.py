"""
인용 네트워크 그래프 구축 모듈
Citation Network Graph Construction Module

기존 reference_extractor.py에서 추출한 citation 데이터를
NetworkX 그래프로 변환하는 모듈
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt


class CitationGraphBuilder:
    """인용 네트워크 그래프를 구축하는 클래스"""

    def __init__(self, min_citations=1):
        """
        Args:
            min_citations (int): 그래프에 포함할 최소 인용 수
        """
        self.min_citations = min_citations
        self.citation_data = None
        self.papers_metadata = None

    def load_citation_data(self, citation_file, metadata_file):
        """기존에 추출된 citation 데이터와 메타데이터 로드"""
        print("📂 Loading citation data and metadata...")

        # Citation network 데이터 로드
        with open(citation_file, "r", encoding="utf-8") as f:
            self.citation_data = json.load(f)

        # Papers 메타데이터 로드
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.papers_metadata = json.load(f)

        print(f"   📄 Citation data loaded: {len(self.citation_data)} papers")
        print(f"   📚 Metadata loaded: {len(self.papers_metadata)} papers")

    def analyze_citation_data(self):
        """Citation 데이터 분석"""
        print("📊 Analyzing citation data...")

        total_citations = sum(
            len(citations) for citations in self.citation_data.values()
        )
        papers_with_citations = sum(
            1 for citations in self.citation_data.values() if citations
        )

        citation_counts = [len(citations) for citations in self.citation_data.values()]

        print(f"   🔗 Total citation relationships: {total_citations}")
        print(f"   📄 Papers with outgoing citations: {papers_with_citations}")
        print(
            f"   📈 Avg citations per paper: {total_citations/len(self.citation_data):.1f}"
        )

        if citation_counts:
            print(f"   📊 Citation count distribution:")
            print(f"      Min: {min(citation_counts)}")
            print(f"      Max: {max(citation_counts)}")
            print(f"      Median: {np.median(citation_counts):.1f}")

    def build_citation_graph(self):
        """Citation 데이터로부터 NetworkX 그래프 구축"""
        print("🔗 Building citation network graph...")

        # 방향 그래프 생성 (인용은 방향성이 있음)
        G = nx.DiGraph()

        # 논문 ID to 메타데이터 매핑 생성
        paper_metadata_map = {}
        for i, paper in enumerate(self.papers_metadata):
            paper_id = f"paper_{i}"
            paper_metadata_map[paper_id] = paper

        # 모든 논문을 노드로 추가 (메타데이터 포함)
        for paper_id, metadata in paper_metadata_map.items():
            G.add_node(
                paper_id,
                node_type="paper",
                title=metadata.get("title", ""),
                authors=metadata.get("authors", []),
                year=metadata.get("year", ""),
                journal=metadata.get("journal", ""),
                keywords=metadata.get("keywords", []),
                has_pdf=metadata.get("has_pdf", False),
            )

        # Citation 엣지 추가
        edges_added = 0
        citation_weights = []

        for citing_paper, citations in self.citation_data.items():
            if not citations:
                continue

            for citation in citations:
                cited_paper = citation["cited_paper_id"]
                similarity = citation.get("similarity", 0.0)

                # 자기 인용 제외
                if citing_paper != cited_paper:
                    # 양쪽 논문이 모두 그래프에 있는 경우만 엣지 추가
                    if G.has_node(citing_paper) and G.has_node(cited_paper):
                        G.add_edge(
                            citing_paper,
                            cited_paper,
                            edge_type="citation",
                            similarity_score=similarity,
                            reference_text=citation.get("reference_text", ""),
                            extracted_title=citation.get("extracted_title", ""),
                        )

                        edges_added += 1
                        citation_weights.append(similarity)

        print(f"✅ Citation graph constructed:")
        print(f"   📄 Nodes (papers): {G.number_of_nodes()}")
        print(f"   🔗 Edges (citations): {G.number_of_edges()}")
        print(f"   📈 Graph density: {nx.density(G):.6f}")

        if citation_weights:
            print(f"   🎯 Similarity score stats:")
            print(f"      Mean: {np.mean(citation_weights):.3f}")
            print(f"      Min: {min(citation_weights):.3f}")
            print(f"      Max: {max(citation_weights):.3f}")

        return G

    def analyze_graph_properties(self, G):
        """그래프 속성 분석"""
        print("📈 Analyzing citation graph properties...")

        # 기본 통계
        stats = {
            "basic_stats": {
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "density": nx.density(G),
                "is_weakly_connected": nx.is_weakly_connected(G),
                "is_strongly_connected": nx.is_strongly_connected(G),
            }
        }

        # 연결 성분 분석
        weak_components = list(nx.weakly_connected_components(G))
        strong_components = list(nx.strongly_connected_components(G))

        stats["connectivity"] = {
            "num_weak_components": len(weak_components),
            "num_strong_components": len(strong_components),
            "largest_weak_component": len(max(weak_components, key=len)),
            "largest_strong_component": len(max(strong_components, key=len)),
        }

        # 중심성 분석 (계산 비용 고려하여 샘플링)
        if G.number_of_nodes() > 0:
            in_degree_centrality = nx.in_degree_centrality(G)
            out_degree_centrality = nx.out_degree_centrality(G)

            # 피인용 수가 많은 논문들 (영향력 있는 논문)
            most_cited = sorted(
                in_degree_centrality.items(), key=lambda x: x[1], reverse=True
            )[:10]

            # 인용을 많이 하는 논문들 (포괄적 리뷰 논문)
            most_citing = sorted(
                out_degree_centrality.items(), key=lambda x: x[1], reverse=True
            )[:10]

            stats["centrality"] = {
                "most_cited_papers": most_cited,
                "most_citing_papers": most_citing,
            }

            # 실제 인용 수로도 정렬
            in_degrees = dict(G.in_degree())
            out_degrees = dict(G.out_degree())

            stats["degree_stats"] = {
                "top_cited_by_count": sorted(
                    in_degrees.items(), key=lambda x: x[1], reverse=True
                )[:10],
                "top_citing_by_count": sorted(
                    out_degrees.items(), key=lambda x: x[1], reverse=True
                )[:10],
                "avg_in_degree": np.mean(list(in_degrees.values())),
                "avg_out_degree": np.mean(list(out_degrees.values())),
            }

        # 년도별 인용 패턴 분석
        year_citation_pattern = self.analyze_temporal_patterns(G)
        stats["temporal_patterns"] = year_citation_pattern

        return stats

    def analyze_temporal_patterns(self, G):
        """시간적 인용 패턴 분석"""
        print("📅 Analyzing temporal citation patterns...")

        # 년도별 논문 수
        papers_by_year = defaultdict(int)

        # 년도별 인용 패턴
        citations_by_year = defaultdict(list)  # citing_year -> [cited_years]

        for node in G.nodes():
            year = G.nodes[node].get("year", "")
            if year and year.isdigit():
                papers_by_year[int(year)] += 1

        # 인용 관계의 시간적 패턴
        for edge in G.edges():
            citing_paper, cited_paper = edge
            citing_year = G.nodes[citing_paper].get("year", "")
            cited_year = G.nodes[cited_paper].get("year", "")

            if (
                citing_year
                and cited_year
                and citing_year.isdigit()
                and cited_year.isdigit()
            ):
                citing_year, cited_year = int(citing_year), int(cited_year)
                citations_by_year[citing_year].append(cited_year)

        # 평균 citation lag 계산
        citation_lags = []
        for citing_year, cited_years in citations_by_year.items():
            for cited_year in cited_years:
                if citing_year > cited_year:  # 과거 논문을 인용하는 경우
                    citation_lags.append(citing_year - cited_year)

        temporal_stats = {
            "papers_by_year": dict(papers_by_year),
            "citation_count_by_year": {
                year: len(cited_years)
                for year, cited_years in citations_by_year.items()
            },
            "avg_citation_lag": np.mean(citation_lags) if citation_lags else 0,
            "median_citation_lag": np.median(citation_lags) if citation_lags else 0,
            "citation_lag_distribution": citation_lags[:100],  # 처음 100개만 저장
        }

        return temporal_stats

    def create_paper_info_map(self, G):
        """논문 정보를 쉽게 조회할 수 있는 맵 생성"""
        paper_info = {}

        for node in G.nodes():
            node_data = G.nodes[node]
            paper_info[node] = {
                "title": node_data.get("title", ""),
                "authors": ", ".join(node_data.get("authors", [])),
                "year": node_data.get("year", ""),
                "journal": node_data.get("journal", ""),
                "keywords": ", ".join(node_data.get("keywords", [])),
                "in_degree": G.in_degree(node),  # 피인용 수
                "out_degree": G.out_degree(node),  # 인용 수
                "has_pdf": node_data.get("has_pdf", False),
            }

        return paper_info

    def save_citation_graph_and_analysis(self, G, stats, output_dir):
        """Citation 그래프와 분석 결과 저장"""
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
            graph_data["edges"].append(edge_data)

        # JSON 파일로 저장
        graph_file = output_dir / "citation_network_graph.json"
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        # 2. GraphML 파일로 저장
        try:
            # GraphML 호환을 위해 그래프 복사 및 속성 변환
            G_graphml = G.copy()

            # 노드 속성을 GraphML 호환 형태로 변환
            for node in G_graphml.nodes():
                # 리스트 타입 속성들을 문자열로 변환
                if "authors" in G_graphml.nodes[node]:
                    authors_list = G_graphml.nodes[node]["authors"]
                    if isinstance(authors_list, (list, set)):
                        G_graphml.nodes[node]["authors"] = ";".join(
                            str(a) for a in authors_list
                        )

                if "keywords" in G_graphml.nodes[node]:
                    keywords_list = G_graphml.nodes[node]["keywords"]
                    if isinstance(keywords_list, (list, set)):
                        G_graphml.nodes[node]["keywords"] = ";".join(
                            str(k) for k in keywords_list
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

            graphml_file = output_dir / "citation_network_graph.graphml"
            nx.write_graphml(G_graphml, graphml_file)

        except Exception as e:
            print(f"⚠️  GraphML 저장 중 오류 발생: {e}")
            print(f"📄 JSON 파일은 정상적으로 저장되었습니다: {graph_file}")
            graphml_file = None

        # 3. 분석 결과 저장
        stats_file = output_dir / "citation_network_analysis.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # 4. 논문 정보 테이블 저장
        paper_info = self.create_paper_info_map(G)
        paper_info_file = output_dir / "citation_papers_info.json"
        with open(paper_info_file, "w", encoding="utf-8") as f:
            json.dump(paper_info, f, ensure_ascii=False, indent=2)

        # 5. CSV 형태의 엣지 리스트 저장
        edge_list = []
        for edge in G.edges():
            edge_info = G.edges[edge].copy()
            edge_info.update(
                {
                    "citing_paper": edge[0],
                    "cited_paper": edge[1],
                    "citing_title": G.nodes[edge[0]].get("title", ""),
                    "cited_title": G.nodes[edge[1]].get("title", ""),
                    "citing_year": G.nodes[edge[0]].get("year", ""),
                    "cited_year": G.nodes[edge[1]].get("year", ""),
                }
            )
            edge_list.append(edge_info)

        edge_df = pd.DataFrame(edge_list)
        edge_file = output_dir / "citation_network_edges.csv"
        edge_df.to_csv(edge_file, index=False, encoding="utf-8")

        print(f"💾 Citation graph results saved:")
        print(f"   🔗 Graph (JSON): {graph_file}")
        if graphml_file:
            print(f"   🔗 Graph (GraphML): {graphml_file}")
        print(f"   📊 Analysis: {stats_file}")
        print(f"   📄 Paper Info: {paper_info_file}")
        print(f"   📈 Edge List: {edge_file}")

        return graph_file

    def process_citation_data(self, citation_file, metadata_file):
        """전체 처리 파이프라인"""
        print("🚀 Starting citation network graph construction...")

        # 1. 데이터 로드
        self.load_citation_data(citation_file, metadata_file)

        # 2. 데이터 분석
        self.analyze_citation_data()

        # 3. 그래프 구축
        G = self.build_citation_graph()

        # 4. 그래프 분석
        stats = self.analyze_graph_properties(G)

        return G, stats


def main():
    """메인 실행 함수"""
    from src import RAW_EXTRACTIONS_DIR, GRAPHS_DIR

    # 필요한 파일들 확인
    citation_file = RAW_EXTRACTIONS_DIR / "citation_network_simple.json"
    metadata_file = RAW_EXTRACTIONS_DIR / "integrated_papers_metadata.json"

    if not citation_file.exists():
        print(f"❌ Citation data not found: {citation_file}")
        print("Please run reference_extractor.py first.")
        return

    if not metadata_file.exists():
        print(f"❌ Metadata not found: {metadata_file}")
        print("Please run data processing pipeline first.")
        return

    print(f"📂 Citation file: {citation_file}")
    print(f"📂 Metadata file: {metadata_file}")

    # Citation Graph Builder 초기화
    builder = CitationGraphBuilder(min_citations=1)

    # 전체 처리
    G, stats = builder.process_citation_data(citation_file, metadata_file)

    # 결과 저장
    output_file = builder.save_citation_graph_and_analysis(G, stats, GRAPHS_DIR)

    print(f"\n✅ Citation network graph construction completed!")
    print(f"📁 Main output: {output_file}")

    # 요약 통계 출력
    print(f"\n📊 Final Summary:")
    print(f"   📄 Total papers: {G.number_of_nodes()}")
    print(f"   🔗 Citation relationships: {G.number_of_edges()}")
    print(f"   📈 Graph density: {nx.density(G):.6f}")

    if stats.get("degree_stats"):
        top_cited = stats["degree_stats"]["top_cited_by_count"][:3]
        print(f"   🏆 Top 3 most cited papers:")
        for i, (paper_id, citation_count) in enumerate(top_cited):
            title = G.nodes[paper_id].get("title", "Unknown")[:50] + "..."
            print(f"      {i+1}. {title} ({citation_count} citations)")

    return G, output_file


if __name__ == "__main__":
    main()
