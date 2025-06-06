"""
저자 협업 네트워크 그래프 구축 모듈
Author Collaboration Network Graph Construction Module
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from tqdm import tqdm
import re


class AuthorCollaborationGraphBuilder:
    """저자 협업 네트워크 그래프를 구축하는 클래스"""

    def __init__(self, min_collaborations=1, min_papers=1):
        """
        Args:
            min_collaborations (int): 그래프에 포함할 최소 협업 횟수
            min_papers (int): 그래프에 포함할 저자의 최소 논문 수
        """
        self.min_collaborations = min_collaborations
        self.min_papers = min_papers

        # 결과 저장용
        self.author_papers = defaultdict(list)  # 저자별 논문 목록
        self.paper_authors = {}  # 논문별 저자 목록
        self.collaboration_matrix = defaultdict(lambda: defaultdict(int))
        self.author_stats = {}
        self.papers_metadata = None

    def clean_author_name(self, author_name):
        """저자명 정제 함수"""
        if not author_name or not isinstance(author_name, str):
            return None

        # 기본 정제
        author_name = author_name.strip()

        # 너무 짧은 이름 제외
        if len(author_name) < 2:
            return None

        # 숫자만 있는 경우 제외
        if author_name.isdigit():
            return None

        # 기본적인 정제 (특수문자 정리)
        author_name = re.sub(r"[^\w\s\.\-]", " ", author_name)
        author_name = re.sub(r"\s+", " ", author_name).strip()

        # 일반적인 형태로 정규화
        # "Smith, John" → "John Smith" 형태로 변환
        if "," in author_name:
            parts = author_name.split(",", 1)
            if len(parts) == 2:
                last_name = parts[0].strip()
                first_name = parts[1].strip()
                author_name = f"{first_name} {last_name}".strip()

        # 연속된 공백 제거
        author_name = re.sub(r"\s+", " ", author_name)

        return author_name

    def normalize_author_name(self, author_name):
        """저자명 정규화 (동일 인물 인식을 위한 고급 정제)"""
        if not author_name:
            return None

        # 기본 정제
        clean_name = self.clean_author_name(author_name)
        if not clean_name:
            return None

        # 소문자 변환
        normalized = clean_name.lower()

        # 점과 공백 정리
        normalized = re.sub(r"\.", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    def extract_author_collaborations(self, papers_metadata):
        """논문 메타데이터에서 저자 협업 관계 추출"""
        print("🔍 Extracting author collaborations from papers...")

        valid_papers = 0
        total_authors = 0
        collaboration_pairs = 0

        for i, paper in enumerate(tqdm(papers_metadata, desc="Processing papers")):
            paper_id = f"paper_{i}"
            title = paper.get("title", "")
            authors = paper.get("authors", [])
            year = paper.get("year", "")
            journal = paper.get("journal", "")

            if not authors or len(authors) < 1:
                continue

            # 저자명 정제
            clean_authors = []
            for author in authors:
                clean_author = self.clean_author_name(author)
                if clean_author:
                    clean_authors.append(clean_author)

            if len(clean_authors) < 1:
                continue

            # 논문-저자 매핑 저장
            self.paper_authors[paper_id] = {
                "title": title,
                "authors": clean_authors,
                "year": year,
                "journal": journal,
                "author_count": len(clean_authors),
            }

            # 저자-논문 매핑 저장
            for author in clean_authors:
                self.author_papers[author].append(
                    {
                        "paper_id": paper_id,
                        "title": title,
                        "year": year,
                        "journal": journal,
                        "co_authors": [a for a in clean_authors if a != author],
                    }
                )

            # 협업 관계 계산 (논문 내 모든 저자 쌍)
            if len(clean_authors) > 1:
                for author1, author2 in combinations(clean_authors, 2):
                    # 사전순 정렬로 중복 방지
                    if author1 > author2:
                        author1, author2 = author2, author1

                    self.collaboration_matrix[author1][author2] += 1
                    collaboration_pairs += 1

            valid_papers += 1
            total_authors += len(clean_authors)

        print(f"✅ Author collaboration extraction completed:")
        print(f"   📄 Papers with authors: {valid_papers}")
        print(f"   👥 Total author instances: {total_authors}")
        print(f"   🤝 Unique authors: {len(self.author_papers)}")
        print(f"   🔗 Collaboration pairs: {collaboration_pairs}")
        print(f"   📈 Avg authors per paper: {total_authors/valid_papers:.1f}")

    def calculate_author_statistics(self):
        """저자별 통계 계산"""
        print("📊 Calculating author statistics...")

        for author, papers in self.author_papers.items():
            # 기본 통계
            paper_count = len(papers)
            years = [p["year"] for p in papers if p["year"]]
            journals = [p["journal"] for p in papers if p["journal"]]

            # 활동 기간
            if years:
                try:
                    year_ints = [int(y) for y in years if str(y).isdigit()]
                    if year_ints:
                        first_year = min(year_ints)
                        last_year = max(year_ints)
                        active_years = last_year - first_year + 1
                    else:
                        first_year = last_year = active_years = 0
                except:
                    first_year = last_year = active_years = 0
            else:
                first_year = last_year = active_years = 0

            # 협업자 수 계산
            collaborators = set()
            for paper in papers:
                collaborators.update(paper["co_authors"])

            # 저널 다양성
            unique_journals = len(set(journals)) if journals else 0

            self.author_stats[author] = {
                "paper_count": paper_count,
                "collaborator_count": len(collaborators),
                "first_year": first_year,
                "last_year": last_year,
                "active_years": active_years,
                "unique_journals": unique_journals,
                "most_frequent_journal": (
                    Counter(journals).most_common(1)[0][0] if journals else ""
                ),
                "collaborators": list(collaborators),
                "papers": papers,
            }

        print(f"✅ Author statistics calculated for {len(self.author_stats)} authors")

    def filter_authors_by_activity(self):
        """활동 수준 기준으로 저자 필터링"""
        print(f"🔍 Filtering authors (min papers: {self.min_papers})...")

        # 최소 논문 수 이상의 저자만 선택
        active_authors = {
            author
            for author, stats in self.author_stats.items()
            if stats["paper_count"] >= self.min_papers
        }

        print(f"   📊 Authors before filtering: {len(self.author_stats)}")
        print(f"   ✅ Authors after filtering: {len(active_authors)}")

        # 협업 관계도 필터링
        filtered_collaborations = defaultdict(lambda: defaultdict(int))
        for author1, collaborations in self.collaboration_matrix.items():
            if author1 in active_authors:
                for author2, count in collaborations.items():
                    if author2 in active_authors and count >= self.min_collaborations:
                        filtered_collaborations[author1][author2] = count

        self.collaboration_matrix = filtered_collaborations

        # 저자 통계도 필터링
        self.author_stats = {
            author: stats
            for author, stats in self.author_stats.items()
            if author in active_authors
        }

        self.author_papers = {
            author: papers
            for author, papers in self.author_papers.items()
            if author in active_authors
        }

        print(f"   📄 Active authors remaining: {len(self.author_stats)}")
        return active_authors

    def build_collaboration_graph(self):
        """협업 관계 그래프 구축"""
        print(f"🔗 Building author collaboration graph...")

        # 무방향 그래프 생성 (협업은 상호적)
        G = nx.Graph()

        # 저자 노드 추가
        for author, stats in self.author_stats.items():
            G.add_node(
                author,
                node_type="author",
                name=author,
                paper_count=stats["paper_count"],
                collaborator_count=stats["collaborator_count"],
                first_year=stats["first_year"],
                last_year=stats["last_year"],
                active_years=stats["active_years"],
                unique_journals=stats["unique_journals"],
                most_frequent_journal=stats["most_frequent_journal"],
            )

        # 협업 엣지 추가
        edges_added = 0
        collaboration_weights = []

        for author1, collaborations in self.collaboration_matrix.items():
            for author2, count in collaborations.items():
                if count >= self.min_collaborations:
                    if G.has_node(author1) and G.has_node(author2):
                        # 공동 논문 제목들 수집
                        common_papers = []
                        for paper_info in self.author_papers[author1]:
                            if author2 in paper_info["co_authors"]:
                                common_papers.append(
                                    {
                                        "paper_id": paper_info["paper_id"],
                                        "title": paper_info["title"],
                                        "year": paper_info["year"],
                                    }
                                )

                        G.add_edge(
                            author1,
                            author2,
                            edge_type="collaboration",
                            collaboration_count=count,
                            weight=count,  # 시각화용
                            common_papers=common_papers[:5],
                        )  # 처음 5개만 저장

                        edges_added += 1
                        collaboration_weights.append(count)

        print(f"✅ Author collaboration graph constructed:")
        print(f"   👥 Nodes (authors): {G.number_of_nodes()}")
        print(f"   🤝 Edges (collaborations): {G.number_of_edges()}")
        print(f"   📈 Graph density: {nx.density(G):.4f}")

        if collaboration_weights:
            print(f"   🎯 Collaboration count stats:")
            print(f"      Min: {min(collaboration_weights)}")
            print(f"      Max: {max(collaboration_weights)}")
            print(f"      Mean: {np.mean(collaboration_weights):.1f}")
            print(f"      Median: {np.median(collaboration_weights):.1f}")

        return G

    def analyze_graph_properties(self, G):
        """그래프 속성 분석"""
        print("📈 Analyzing author collaboration graph properties...")

        # 기본 통계
        stats = {
            "basic_stats": {
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "density": float(nx.density(G)),
                "is_connected": nx.is_connected(G),
            }
        }

        # 연결 성분 분석
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            stats["connectivity"] = {
                "num_components": len(components),
                "largest_component_size": len(max(components, key=len)),
                "component_sizes": [
                    len(comp) for comp in components[:10]
                ],  # 상위 10개만
            }

        # 중심성 분석
        if G.number_of_nodes() > 0:
            degree_centrality = nx.degree_centrality(G)

            # 베트위너스 중심성 (큰 그래프의 경우 샘플링)
            if G.number_of_nodes() < 1000:
                betweenness_centrality = nx.betweenness_centrality(G)
            else:
                betweenness_centrality = nx.betweenness_centrality(
                    G, k=min(500, G.number_of_nodes())
                )

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
            }

            # 실제 협업 수로도 정렬
            degrees = dict(G.degree())

            stats["degree_stats"] = {
                "top_collaborators_by_count": sorted(
                    degrees.items(), key=lambda x: x[1], reverse=True
                )[:10],
                "avg_degree": float(np.mean(list(degrees.values()))),
                "max_degree": max(degrees.values()),
                "min_degree": min(degrees.values()),
            }

        # 클러스터링 분석
        if G.number_of_nodes() > 0:
            clustering_coeffs = nx.clustering(G)
            stats["clustering"] = {
                "average_clustering": float(nx.average_clustering(G)),
                "global_clustering": float(nx.transitivity(G)),
                "top_clustered_authors": [
                    (k, float(v))
                    for k, v in sorted(
                        clustering_coeffs.items(), key=lambda x: x[1], reverse=True
                    )[:10]
                ],
            }

        # 협업 패턴 분석
        collaboration_analysis = self.analyze_collaboration_patterns(G)
        stats["collaboration_patterns"] = collaboration_analysis

        return stats

    def analyze_collaboration_patterns(self, G):
        """협업 패턴 분석"""
        print("🔍 Analyzing collaboration patterns...")

        # 논문 수별 저자 분포
        paper_counts = [G.nodes[author]["paper_count"] for author in G.nodes()]

        # 협업 강도 분포
        collaboration_strengths = []
        for edge in G.edges():
            collaboration_strengths.append(G.edges[edge]["collaboration_count"])

        # 활동 기간 분석
        active_years = [
            G.nodes[author]["active_years"]
            for author in G.nodes()
            if G.nodes[author]["active_years"] > 0
        ]

        # 저널 다양성 분석
        journal_diversity = [G.nodes[author]["unique_journals"] for author in G.nodes()]

        patterns = {
            "paper_count_distribution": {
                "mean": float(np.mean(paper_counts)),
                "median": float(np.median(paper_counts)),
                "max": int(max(paper_counts)),
                "min": int(min(paper_counts)),
            },
            "collaboration_strength_distribution": {
                "mean": (
                    float(np.mean(collaboration_strengths))
                    if collaboration_strengths
                    else 0
                ),
                "median": (
                    float(np.median(collaboration_strengths))
                    if collaboration_strengths
                    else 0
                ),
                "max": (
                    int(max(collaboration_strengths)) if collaboration_strengths else 0
                ),
                "min": (
                    int(min(collaboration_strengths)) if collaboration_strengths else 0
                ),
            },
            "activity_period_distribution": {
                "mean": float(np.mean(active_years)) if active_years else 0,
                "median": float(np.median(active_years)) if active_years else 0,
                "max": int(max(active_years)) if active_years else 0,
            },
            "journal_diversity_distribution": {
                "mean": float(np.mean(journal_diversity)),
                "median": float(np.median(journal_diversity)),
                "max": int(max(journal_diversity)),
            },
        }

        return patterns

    def create_author_info_map(self, G):
        """저자 정보를 쉽게 조회할 수 있는 맵 생성"""
        author_info = {}

        for author in G.nodes():
            node_data = G.nodes[author]
            author_info[author] = {
                "name": author,
                "paper_count": node_data.get("paper_count", 0),
                "collaborator_count": node_data.get("collaborator_count", 0),
                "degree": G.degree(author),  # 실제 그래프에서의 연결 수
                "first_year": node_data.get("first_year", ""),
                "last_year": node_data.get("last_year", ""),
                "active_years": node_data.get("active_years", 0),
                "unique_journals": node_data.get("unique_journals", 0),
                "most_frequent_journal": node_data.get("most_frequent_journal", ""),
            }

        return author_info

    def save_collaboration_graph_and_analysis(self, G, stats, output_dir):
        """Author collaboration 그래프와 분석 결과 저장"""
        output_dir = Path(output_dir)

        # 1. NetworkX 그래프를 JSON으로 저장 (GraphRAG용)
        graph_data = {"nodes": [], "edges": []}

        # 노드 정보
        for node in G.nodes():
            node_data = G.nodes[node].copy()
            node_data["id"] = node
            graph_data["nodes"].append(node_data)

        # 엣지 정보
        for edge in G.edges():
            edge_data = G.edges[edge].copy()
            edge_data["source"] = edge[0]
            edge_data["target"] = edge[1]
            # common_papers 리스트 유지 (JSON 직렬화 가능)
            graph_data["edges"].append(edge_data)

        # JSON 파일로 저장
        graph_file = output_dir / "author_collaboration_graph.json"
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        # 2. GraphML 파일로 저장
        try:
            # GraphML 호환을 위해 그래프 복사 및 속성 변환
            G_graphml = G.copy()

            # 엣지의 common_papers 리스트를 문자열로 변환
            for edge in G_graphml.edges():
                if "common_papers" in G_graphml.edges[edge]:
                    common_papers = G_graphml.edges[edge]["common_papers"]
                    if isinstance(common_papers, list):
                        # 논문 정보를 간단한 문자열로 변환
                        paper_titles = [p.get("title", "")[:50] for p in common_papers]
                        G_graphml.edges[edge]["common_papers"] = ";".join(paper_titles)

                # 기타 복잡한 타입들 변환
                for attr_name, attr_value in G_graphml.edges[edge].items():
                    if (
                        isinstance(attr_value, (list, dict))
                        and attr_name != "common_papers"
                    ):
                        if isinstance(attr_value, dict):
                            G_graphml.edges[edge][attr_name] = json.dumps(attr_value)
                        else:
                            G_graphml.edges[edge][attr_name] = ";".join(
                                str(v) for v in attr_value
                            )
                    elif not isinstance(attr_value, (str, int, float, bool)):
                        G_graphml.edges[edge][attr_name] = str(attr_value)

            graphml_file = output_dir / "author_collaboration_graph.graphml"
            nx.write_graphml(G_graphml, graphml_file)

        except Exception as e:
            print(f"⚠️  GraphML 저장 중 오류 발생: {e}")
            print(f"📄 JSON 파일은 정상적으로 저장되었습니다: {graph_file}")
            graphml_file = None

        # 3. 분석 결과 저장
        stats_file = output_dir / "author_collaboration_analysis.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # 4. 저자 정보 테이블 저장
        author_info = self.create_author_info_map(G)
        author_info_file = output_dir / "author_collaboration_info.json"
        with open(author_info_file, "w", encoding="utf-8") as f:
            json.dump(author_info, f, ensure_ascii=False, indent=2)

        # 5. CSV 형태의 엣지 리스트 저장
        edge_list = []
        for edge in G.edges():
            edge_info = G.edges[edge].copy()
            edge_info.update(
                {
                    "author1": edge[0],
                    "author2": edge[1],
                    "author1_papers": G.nodes[edge[0]].get("paper_count", 0),
                    "author2_papers": G.nodes[edge[1]].get("paper_count", 0),
                    "author1_first_year": G.nodes[edge[0]].get("first_year", ""),
                    "author2_first_year": G.nodes[edge[1]].get("first_year", ""),
                }
            )
            # common_papers 필드 간소화
            if "common_papers" in edge_info:
                edge_info["common_papers_count"] = len(edge_info["common_papers"])
                edge_info.pop("common_papers")  # CSV에서는 제외
            edge_list.append(edge_info)

        edge_df = pd.DataFrame(edge_list)
        edge_file = output_dir / "author_collaboration_edges.csv"
        edge_df.to_csv(edge_file, index=False, encoding="utf-8")

        print(f"💾 Author collaboration graph results saved:")
        print(f"   🔗 Graph (JSON): {graph_file}")
        if graphml_file:
            print(f"   🔗 Graph (GraphML): {graphml_file}")
        print(f"   📊 Analysis: {stats_file}")
        print(f"   👥 Author Info: {author_info_file}")
        print(f"   📈 Edge List: {edge_file}")

        return graph_file

    def process_papers(self, papers_metadata):
        """전체 처리 파이프라인"""
        print("🚀 Starting author collaboration graph construction...")

        # 메타데이터 저장
        self.papers_metadata = papers_metadata

        # 1. 저자 협업 관계 추출
        self.extract_author_collaborations(papers_metadata)

        # 2. 저자 통계 계산
        self.calculate_author_statistics()

        # 3. 활동 수준 기준 필터링
        active_authors = self.filter_authors_by_activity()

        # 4. 협업 그래프 구축
        G = self.build_collaboration_graph()

        # 5. 그래프 분석
        stats = self.analyze_graph_properties(G)

        return G, stats


def main():
    """메인 실행 함수"""
    from src import RAW_EXTRACTIONS_DIR, GRAPHS_DIR

    # 통합 메타데이터 로드
    metadata_file = RAW_EXTRACTIONS_DIR / "integrated_papers_metadata.json"

    if not metadata_file.exists():
        print(f"❌ Metadata not found: {metadata_file}")
        print("Please run data processing pipeline first.")
        return

    with open(metadata_file, "r", encoding="utf-8") as f:
        papers_metadata = json.load(f)

    print(f"📄 Loaded {len(papers_metadata)} papers metadata")

    # Author Collaboration Graph Builder 초기화
    builder = AuthorCollaborationGraphBuilder(
        min_collaborations=1,  # 최소 1번 이상 협업
        min_papers=1,  # 최소 1편 이상 논문 (모든 저자 포함)
    )

    # 전체 처리
    G, stats = builder.process_papers(papers_metadata)

    # 결과 저장
    output_file = builder.save_collaboration_graph_and_analysis(G, stats, GRAPHS_DIR)

    print(f"\n✅ Author collaboration graph construction completed!")
    print(f"📁 Main output: {output_file}")

    # 요약 통계 출력
    print(f"\n📊 Final Summary:")
    print(f"   👥 Total authors: {G.number_of_nodes()}")
    print(f"   🤝 Collaboration relationships: {G.number_of_edges()}")
    print(f"   📈 Graph density: {nx.density(G):.4f}")

    if stats.get("degree_stats"):
        top_collaborators = stats["degree_stats"]["top_collaborators_by_count"][:3]
        print(f"   🏆 Top 3 most collaborative authors:")
        for i, (author, collab_count) in enumerate(top_collaborators):
            paper_count = G.nodes[author].get("paper_count", 0)
            print(
                f"      {i+1}. {author} ({collab_count} collaborations, {paper_count} papers)"
            )

    return G, output_file


if __name__ == "__main__":
    main()
