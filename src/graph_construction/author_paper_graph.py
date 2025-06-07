"""
저자-논문 이분 그래프 구축 모듈
Author-Paper Bipartite Graph Construction Module
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import re


class AuthorPaperGraphBuilder:
    """저자-논문 이분 그래프를 구축하는 클래스"""

    def __init__(self, min_papers_per_author=1, min_authors_per_paper=1):
        """
        Args:
            min_papers_per_author (int): 그래프에 포함할 저자의 최소 논문 수
            min_authors_per_paper (int): 그래프에 포함할 논문의 최소 저자 수
        """
        self.min_papers_per_author = min_papers_per_author
        self.min_authors_per_paper = min_authors_per_paper

        # 결과 저장용
        self.author_papers = defaultdict(list)  # 저자별 논문 목록
        self.paper_authors = {}  # 논문별 저자 목록
        self.author_stats = {}
        self.paper_stats = {}
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

    def extract_author_paper_relationships(self, papers_metadata):
        """논문 메타데이터에서 저자-논문 관계 추출"""
        print("🔍 Extracting author-paper relationships...")

        valid_papers = 0
        valid_authors = 0
        total_authorship_relations = 0

        for i, paper in enumerate(tqdm(papers_metadata, desc="Processing papers")):
            paper_id = f"paper_{i}"
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            authors = paper.get("authors", [])
            year = paper.get("year", "")
            journal = paper.get("journal", "")
            keywords = paper.get("keywords", [])

            if not authors:
                continue

            # 저자명 정제
            clean_authors = []
            for author in authors:
                clean_author = self.clean_author_name(author)
                if clean_author:
                    clean_authors.append(clean_author)

            if len(clean_authors) < self.min_authors_per_paper:
                continue

            # 논문-저자 매핑 저장
            self.paper_authors[paper_id] = {
                "title": title,
                "abstract": abstract,
                "authors": clean_authors,
                "year": year,
                "journal": journal,
                "keywords": keywords,
                "author_count": len(clean_authors),
                "keyword_count": len(keywords) if keywords else 0,
            }

            # 저자-논문 매핑 저장
            for author in clean_authors:
                self.author_papers[author].append(
                    {
                        "paper_id": paper_id,
                        "title": title,
                        "abstract": abstract,
                        "year": year,
                        "journal": journal,
                        "keywords": keywords,
                        "authorship_position": clean_authors.index(author),  # 저자 순서
                        "is_first_author": clean_authors.index(author) == 0,
                        "is_last_author": clean_authors.index(author)
                        == len(clean_authors) - 1,
                        "co_author_count": len(clean_authors) - 1,
                    }
                )
                total_authorship_relations += 1

            valid_papers += 1

        valid_authors = len(self.author_papers)

        print(f"✅ Author-paper relationship extraction completed:")
        print(f"   📄 Papers with authors: {valid_papers}")
        print(f"   👥 Unique authors: {valid_authors}")
        print(f"   🔗 Total authorship relations: {total_authorship_relations}")
        print(
            f"   📈 Avg authors per paper: {total_authorship_relations/valid_papers:.1f}"
        )
        print(
            f"   📈 Avg papers per author: {total_authorship_relations/valid_authors:.1f}"
        )

    def calculate_author_statistics(self):
        """저자별 통계 계산"""
        print("📊 Calculating author statistics...")

        for author, papers in self.author_papers.items():
            # 기본 통계
            paper_count = len(papers)
            first_author_count = sum(1 for p in papers if p["is_first_author"])
            last_author_count = sum(1 for p in papers if p["is_last_author"])
            single_author_count = sum(1 for p in papers if p["co_author_count"] == 0)

            # 활동 기간
            years = [p["year"] for p in papers if p["year"]]
            if years:
                try:
                    year_ints = [int(y) for y in years if str(y).isdigit()]
                    if year_ints:
                        first_year = min(year_ints)
                        last_year = max(year_ints)
                        active_years = last_year - first_year + 1
                        papers_by_year = Counter(year_ints)
                    else:
                        first_year = last_year = active_years = 0
                        papers_by_year = {}
                except:
                    first_year = last_year = active_years = 0
                    papers_by_year = {}
            else:
                first_year = last_year = active_years = 0
                papers_by_year = {}

            # 저널 분석
            journals = [p["journal"] for p in papers if p["journal"]]
            unique_journals = len(set(journals)) if journals else 0
            most_frequent_journal = (
                Counter(journals).most_common(1)[0][0] if journals else ""
            )

            # 키워드 분석
            all_keywords = []
            for paper in papers:
                all_keywords.extend(paper.get("keywords", []))
            keyword_frequency = Counter(all_keywords)
            top_keywords = keyword_frequency.most_common(10)

            # 협업 패턴
            total_co_authors = sum(p["co_author_count"] for p in papers)
            avg_co_authors = total_co_authors / paper_count if paper_count > 0 else 0

            self.author_stats[author] = {
                "name": author,
                "paper_count": paper_count,
                "first_author_count": first_author_count,
                "last_author_count": last_author_count,
                "single_author_count": single_author_count,
                "first_year": first_year,
                "last_year": last_year,
                "active_years": active_years,
                "unique_journals": unique_journals,
                "most_frequent_journal": most_frequent_journal,
                "avg_co_authors": avg_co_authors,
                "papers_by_year": dict(papers_by_year),
                "top_keywords": top_keywords,
                "productivity_type": self.classify_author_productivity(
                    paper_count, active_years, first_author_count
                ),
            }

        print(f"✅ Author statistics calculated for {len(self.author_stats)} authors")

    def calculate_paper_statistics(self):
        """논문별 통계 계산"""
        print("📊 Calculating paper statistics...")

        for paper_id, paper_info in self.paper_authors.items():
            authors = paper_info["authors"]
            author_count = len(authors)

            # 논문 타입 분류
            if author_count == 1:
                collaboration_type = "Single Author"
            elif author_count <= 3:
                collaboration_type = "Small Team"
            elif author_count <= 6:
                collaboration_type = "Medium Team"
            else:
                collaboration_type = "Large Team"

            # 저자들의 경력 분석 (이전 논문 수 기준)
            author_experience_levels = []
            for author in authors:
                author_paper_count = len(self.author_papers.get(author, []))
                if author_paper_count == 1:
                    author_experience_levels.append("Novice")
                elif author_paper_count <= 5:
                    author_experience_levels.append("Intermediate")
                else:
                    author_experience_levels.append("Experienced")

            experience_distribution = Counter(author_experience_levels)

            self.paper_stats[paper_id] = {
                "title": paper_info["title"],
                "abstract": paper_info["abstract"],
                "author_count": author_count,
                "collaboration_type": collaboration_type,
                "year": paper_info["year"],
                "journal": paper_info["journal"],
                "keyword_count": paper_info["keyword_count"],
                "authors": authors,
                "experience_distribution": dict(experience_distribution),
                "has_experienced_authors": "Experienced" in author_experience_levels,
                "is_interdisciplinary": self.assess_interdisciplinarity(
                    paper_info["keywords"]
                ),
            }

        print(f"✅ Paper statistics calculated for {len(self.paper_stats)} papers")

    def classify_author_productivity(
        self, paper_count, active_years, first_author_count
    ):
        """저자 생산성 타입 분류"""
        if paper_count >= 10 and active_years >= 5:
            return "Highly Productive"
        elif paper_count >= 5 and first_author_count >= 2:
            return "Leading Researcher"
        elif paper_count >= 3:
            return "Active Researcher"
        elif paper_count == 1:
            return "Newcomer"
        else:
            return "Occasional Contributor"

    def assess_interdisciplinarity(self, keywords):
        """키워드 기반 학제간 연구 여부 평가"""
        if not keywords or len(keywords) < 3:
            return False

        # 간단한 도메인 분류
        ai_keywords = {
            "machine learning",
            "deep learning",
            "neural network",
            "ai",
            "artificial intelligence",
        }
        energy_keywords = {"battery", "energy", "power", "electric", "charging"}
        engineering_keywords = {
            "system",
            "control",
            "optimization",
            "design",
            "manufacturing",
        }

        keyword_set = set(kw.lower() for kw in keywords)

        domains = 0
        if ai_keywords & keyword_set:
            domains += 1
        if energy_keywords & keyword_set:
            domains += 1
        if engineering_keywords & keyword_set:
            domains += 1

        return domains >= 2

    def filter_by_activity(self):
        """활동 수준 기준으로 저자/논문 필터링"""
        print(
            f"🔍 Filtering authors/papers (min papers per author: {self.min_papers_per_author})..."
        )

        # 최소 논문 수 이상의 저자만 선택
        active_authors = {
            author
            for author, stats in self.author_stats.items()
            if stats["paper_count"] >= self.min_papers_per_author
        }

        print(f"   📊 Authors before filtering: {len(self.author_stats)}")
        print(f"   ✅ Authors after filtering: {len(active_authors)}")

        # 저자 통계 필터링
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

        # 필터링된 저자들과 관련된 논문들만 유지
        active_papers = set()
        for author in active_authors:
            for paper_info in self.author_papers[author]:
                active_papers.add(paper_info["paper_id"])

        filtered_paper_authors = {}
        filtered_paper_stats = {}

        for paper_id in active_papers:
            if paper_id in self.paper_authors:
                # 논문의 저자 목록을 active_authors로 필터링
                original_authors = self.paper_authors[paper_id]["authors"]
                filtered_authors = [a for a in original_authors if a in active_authors]

                if len(filtered_authors) >= self.min_authors_per_paper:
                    paper_info = self.paper_authors[paper_id].copy()
                    paper_info["authors"] = filtered_authors
                    paper_info["author_count"] = len(filtered_authors)

                    filtered_paper_authors[paper_id] = paper_info

                    if paper_id in self.paper_stats:
                        filtered_paper_stats[paper_id] = self.paper_stats[
                            paper_id
                        ].copy()
                        filtered_paper_stats[paper_id]["author_count"] = len(
                            filtered_authors
                        )
                        filtered_paper_stats[paper_id]["authors"] = filtered_authors

        self.paper_authors = filtered_paper_authors
        self.paper_stats = filtered_paper_stats

        print(f"   📄 Active authors remaining: {len(self.author_stats)}")
        print(f"   📰 Active papers remaining: {len(self.paper_authors)}")

        return active_authors

    def build_author_paper_graph(self):
        """저자-논문 이분 그래프 구축"""
        print(f"🔗 Building author-paper bipartite graph...")

        # 무방향 그래프 생성 (저자-논문 관계는 방향성이 없음)
        G = nx.Graph()

        # 저자 노드 추가
        for author, stats in self.author_stats.items():
            G.add_node(
                author,
                node_type="author",
                name=author,
                paper_count=stats["paper_count"],
                first_author_count=stats["first_author_count"],
                last_author_count=stats["last_author_count"],
                single_author_count=stats["single_author_count"],
                first_year=stats["first_year"],
                last_year=stats["last_year"],
                active_years=stats["active_years"],
                unique_journals=stats["unique_journals"],
                most_frequent_journal=stats["most_frequent_journal"],
                avg_co_authors=stats["avg_co_authors"],
                productivity_type=stats["productivity_type"],
                top_keywords=[kw for kw, count in stats["top_keywords"]][
                    :5
                ],  # 상위 5개만
            )

        # 논문 노드 추가
        for paper_id, stats in self.paper_stats.items():
            G.add_node(
                paper_id,
                node_type="paper",
                title=stats["title"],
                abstract=stats.get("abstract", ""),  # ✅ Abstract 추가
                author_count=stats["author_count"],
                collaboration_type=stats["collaboration_type"],
                year=stats["year"],
                journal=stats["journal"],
                keyword_count=stats["keyword_count"],
                has_experienced_authors=stats["has_experienced_authors"],
                is_interdisciplinary=stats["is_interdisciplinary"],
            )

        # 저자-논문 엣지 추가
        edges_added = 0

        for author, papers in self.author_papers.items():
            if G.has_node(author):
                for paper_info in papers:
                    paper_id = paper_info["paper_id"]
                    if G.has_node(paper_id):
                        G.add_edge(
                            author,
                            paper_id,
                            edge_type="authored",
                            authorship_position=paper_info["authorship_position"],
                            is_first_author=paper_info["is_first_author"],
                            is_last_author=paper_info["is_last_author"],
                            co_author_count=paper_info["co_author_count"],
                            year=paper_info["year"],
                            weight=1.0,  # 모든 authorship 관계는 동등한 가중치
                        )
                        edges_added += 1

        print(f"✅ Author-paper bipartite graph constructed:")
        print(
            f"   👥 Author nodes: {sum(1 for n in G.nodes() if G.nodes[n]['node_type'] == 'author')}"
        )
        print(
            f"   📄 Paper nodes: {sum(1 for n in G.nodes() if G.nodes[n]['node_type'] == 'paper')}"
        )
        print(f"   🔗 Authorship edges: {G.number_of_edges()}")
        print(f"   📈 Graph density: {nx.density(G):.6f}")

        return G

    def analyze_graph_properties(self, G):
        """그래프 속성 분석"""
        print("📈 Analyzing author-paper graph properties...")

        # 노드별 통계
        author_nodes = [n for n in G.nodes() if G.nodes[n]["node_type"] == "author"]
        paper_nodes = [n for n in G.nodes() if G.nodes[n]["node_type"] == "paper"]

        # 기본 통계
        stats = {
            "basic_stats": {
                "num_authors": len(author_nodes),
                "num_papers": len(paper_nodes),
                "num_edges": G.number_of_edges(),
                "density": float(nx.density(G)),
                "avg_papers_per_author": (
                    G.number_of_edges() / len(author_nodes) if author_nodes else 0
                ),
                "avg_authors_per_paper": (
                    G.number_of_edges() / len(paper_nodes) if paper_nodes else 0
                ),
            }
        }

        # 저자별 논문 수 분석
        author_paper_counts = {}
        for author in author_nodes:
            paper_count = G.degree(author)  # 연결된 논문 수
            author_paper_counts[author] = paper_count

        # 논문별 저자 수 분석
        paper_author_counts = {}
        for paper in paper_nodes:
            author_count = G.degree(paper)  # 연결된 저자 수
            paper_author_counts[paper] = author_count

        stats["productivity_analysis"] = {
            "top_authors_by_papers": sorted(
                author_paper_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "author_productivity_distribution": {
                "mean": float(np.mean(list(author_paper_counts.values()))),
                "median": float(np.median(list(author_paper_counts.values()))),
                "max": max(author_paper_counts.values()),
                "min": min(author_paper_counts.values()),
            },
            "collaboration_size_distribution": {
                "mean": float(np.mean(list(paper_author_counts.values()))),
                "median": float(np.median(list(paper_author_counts.values()))),
                "max": max(paper_author_counts.values()),
                "min": min(paper_author_counts.values()),
            },
        }

        # 저자 생산성 타입별 분석
        productivity_analysis = self.analyze_productivity_types(G)
        stats["productivity_types"] = productivity_analysis

        # 협업 패턴 분석
        collaboration_analysis = self.analyze_collaboration_patterns(G)
        stats["collaboration_patterns"] = collaboration_analysis

        # 시간적 분석
        temporal_analysis = self.analyze_temporal_patterns(G)
        stats["temporal_patterns"] = temporal_analysis

        return stats

    def analyze_productivity_types(self, G):
        """저자 생산성 타입별 분석"""
        type_counts = Counter()
        type_papers = defaultdict(int)

        for node in G.nodes():
            if G.nodes[node]["node_type"] == "author":
                productivity_type = G.nodes[node]["productivity_type"]
                paper_count = G.degree(node)

                type_counts[productivity_type] += 1
                type_papers[productivity_type] += paper_count

        return {
            "type_distribution": dict(type_counts),
            "papers_by_type": dict(type_papers),
        }

    def analyze_collaboration_patterns(self, G):
        """협업 패턴 분석"""
        collaboration_types = Counter()

        for node in G.nodes():
            if G.nodes[node]["node_type"] == "paper":
                collab_type = G.nodes[node]["collaboration_type"]
                collaboration_types[collab_type] += 1

        # 첫 번째/마지막 저자 분석
        first_author_papers = 0
        last_author_papers = 0
        single_author_papers = 0

        for edge in G.edges():
            edge_data = G.edges[edge]
            if edge_data.get("is_first_author", False):
                first_author_papers += 1
            if edge_data.get("is_last_author", False):
                last_author_papers += 1
            if edge_data.get("co_author_count", 0) == 0:
                single_author_papers += 1

        return {
            "collaboration_type_distribution": dict(collaboration_types),
            "authorship_patterns": {
                "first_author_papers": first_author_papers,
                "last_author_papers": last_author_papers,
                "single_author_papers": single_author_papers,
            },
        }

    def analyze_temporal_patterns(self, G):
        """시간적 패턴 분석"""
        papers_by_year = defaultdict(int)
        authors_by_year = defaultdict(set)

        for edge in G.edges():
            year = G.edges[edge].get("year", "")
            if year and str(year).isdigit():
                year_int = int(year)
                papers_by_year[year_int] += 1

                # 엣지의 저자와 논문 찾기
                node1, node2 = edge
                author = node1 if G.nodes[node1]["node_type"] == "author" else node2
                authors_by_year[year_int].add(author)

        # 년도별 활동 저자 수 계산
        authors_count_by_year = {
            year: len(authors) for year, authors in authors_by_year.items()
        }

        return {
            "papers_by_year": dict(papers_by_year),
            "authors_count_by_year": authors_count_by_year,
            "active_years": list(papers_by_year.keys()) if papers_by_year else [],
        }

    def create_node_info_maps(self, G):
        """노드 정보를 쉽게 조회할 수 있는 맵 생성"""
        author_info = {}
        paper_info = {}

        # 저자 정보
        for node in G.nodes():
            if G.nodes[node]["node_type"] == "author":
                node_data = G.nodes[node]
                author_info[node] = {
                    "name": node,
                    "productivity_type": node_data.get("productivity_type", ""),
                    "paper_count": G.degree(node),
                    "first_author_count": node_data.get("first_author_count", 0),
                    "last_author_count": node_data.get("last_author_count", 0),
                    "single_author_count": node_data.get("single_author_count", 0),
                    "first_year": node_data.get("first_year", ""),
                    "last_year": node_data.get("last_year", ""),
                    "active_years": node_data.get("active_years", 0),
                    "unique_journals": node_data.get("unique_journals", 0),
                    "most_frequent_journal": node_data.get("most_frequent_journal", ""),
                    "top_keywords": node_data.get("top_keywords", [])[:3],  # 상위 3개만
                }

        # 논문 정보
        for node in G.nodes():
            if G.nodes[node]["node_type"] == "paper":
                node_data = G.nodes[node]

                # 해당 논문의 저자들 찾기
                authors = [
                    neighbor
                    for neighbor in G.neighbors(node)
                    if G.nodes[neighbor]["node_type"] == "author"
                ]

                paper_info[node] = {
                    "title": node_data.get("title", ""),
                    "abstract": node_data.get("abstract", ""),
                    "author_count": G.degree(node),
                    "collaboration_type": node_data.get("collaboration_type", ""),
                    "year": node_data.get("year", ""),
                    "journal": node_data.get("journal", ""),
                    "keyword_count": node_data.get("keyword_count", 0),
                    "has_experienced_authors": node_data.get(
                        "has_experienced_authors", False
                    ),
                    "is_interdisciplinary": node_data.get(
                        "is_interdisciplinary", False
                    ),
                    "authors": authors[:5],  # 처음 5명만
                }

        return author_info, paper_info

    def save_author_paper_graph_and_analysis(self, G, stats, output_dir):
        """Author-paper 그래프와 분석 결과 저장"""
        output_dir = Path(output_dir)

        # 1. NetworkX 그래프를 JSON으로 저장 (GraphRAG용)
        graph_data = {"nodes": [], "edges": []}

        # 노드 정보
        for node in G.nodes():
            node_data = G.nodes[node].copy()
            node_data["id"] = node
            # 리스트 필드들 JSON 직렬화 가능하게 유지
            graph_data["nodes"].append(node_data)

        # 엣지 정보
        for edge in G.edges():
            edge_data = G.edges[edge].copy()
            edge_data["source"] = edge[0]
            edge_data["target"] = edge[1]
            graph_data["edges"].append(edge_data)

        # JSON 파일로 저장
        graph_file = output_dir / "author_paper_graph.json"
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        # 2. GraphML 파일로 저장
        try:
            # GraphML 호환을 위해 그래프 복사 및 속성 변환
            G_graphml = G.copy()

            # 리스트 타입 속성들을 문자열로 변환
            for node in G_graphml.nodes():
                for attr_name, attr_value in G_graphml.nodes[node].items():
                    if isinstance(attr_value, list):
                        G_graphml.nodes[node][attr_name] = ";".join(
                            str(v) for v in attr_value
                        )
                    elif not isinstance(attr_value, (str, int, float, bool)):
                        G_graphml.nodes[node][attr_name] = str(attr_value)

            graphml_file = output_dir / "author_paper_graph.graphml"
            nx.write_graphml(G_graphml, graphml_file)

        except Exception as e:
            print(f"⚠️  GraphML 저장 중 오류 발생: {e}")
            print(f"📄 JSON 파일은 정상적으로 저장되었습니다: {graph_file}")
            graphml_file = None

        # 3. 분석 결과 저장
        stats_file = output_dir / "author_paper_analysis.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # 4. 노드 정보 테이블들 저장
        author_info, paper_info = self.create_node_info_maps(G)

        author_info_file = output_dir / "author_paper_author_info.json"
        with open(author_info_file, "w", encoding="utf-8") as f:
            json.dump(author_info, f, ensure_ascii=False, indent=2)

        paper_info_file = output_dir / "author_paper_paper_info.json"
        with open(paper_info_file, "w", encoding="utf-8") as f:
            json.dump(paper_info, f, ensure_ascii=False, indent=2)

        # 5. CSV 형태의 엣지 리스트 저장
        edge_list = []
        for edge in G.edges():
            edge_info = G.edges[edge].copy()

            # 저자와 논문 구분
            node1, node2 = edge
            if G.nodes[node1]["node_type"] == "author":
                author_id, paper_id = node1, node2
            else:
                author_id, paper_id = node2, node1

            edge_info.update(
                {
                    "author_id": author_id,
                    "paper_id": paper_id,
                    "author_name": author_id,
                    "paper_title": G.nodes[paper_id].get("title", ""),
                    "paper_year": G.nodes[paper_id].get("year", ""),
                    "author_productivity_type": G.nodes[author_id].get(
                        "productivity_type", ""
                    ),
                    "collaboration_type": G.nodes[paper_id].get(
                        "collaboration_type", ""
                    ),
                }
            )
            edge_list.append(edge_info)

        edge_df = pd.DataFrame(edge_list)
        edge_file = output_dir / "author_paper_edges.csv"
        edge_df.to_csv(edge_file, index=False, encoding="utf-8")

        print(f"💾 Author-paper graph results saved:")
        print(f"   🔗 Graph (JSON): {graph_file}")
        if graphml_file:
            print(f"   🔗 Graph (GraphML): {graphml_file}")
        print(f"   📊 Analysis: {stats_file}")
        print(f"   👥 Author Info: {author_info_file}")
        print(f"   📄 Paper Info: {paper_info_file}")
        print(f"   📈 Edge List: {edge_file}")

        return graph_file

    def process_papers(self, papers_metadata):
        """전체 처리 파이프라인"""
        print("🚀 Starting author-paper graph construction...")

        # 메타데이터 저장
        self.papers_metadata = papers_metadata

        # 1. 저자-논문 관계 추출
        self.extract_author_paper_relationships(papers_metadata)

        # 2. 저자 통계 계산
        self.calculate_author_statistics()

        # 3. 논문 통계 계산
        self.calculate_paper_statistics()

        # 4. 활동 수준 기준 필터링
        active_authors = self.filter_by_activity()

        # 5. 이분 그래프 구축
        G = self.build_author_paper_graph()

        # 6. 그래프 분석
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

    # Author-Paper Graph Builder 초기화
    builder = AuthorPaperGraphBuilder(
        min_papers_per_author=1,  # 최소 1편 이상 논문이 있는 저자만 포함
        min_authors_per_paper=1,  # 최소 1명 이상 저자가 있는 논문만 포함
    )

    # 전체 처리
    G, stats = builder.process_papers(papers_metadata)

    # 결과 저장
    output_file = builder.save_author_paper_graph_and_analysis(G, stats, GRAPHS_DIR)

    print(f"\n✅ Author-paper graph construction completed!")
    print(f"📁 Main output: {output_file}")

    # 요약 통계 출력
    print(f"\n📊 Final Summary:")
    print(f"   👥 Total authors: {stats['basic_stats']['num_authors']}")
    print(f"   📄 Total papers: {stats['basic_stats']['num_papers']}")
    print(f"   🔗 Authorship relationships: {stats['basic_stats']['num_edges']}")
    print(
        f"   📈 Avg papers per author: {stats['basic_stats']['avg_papers_per_author']:.1f}"
    )
    print(
        f"   📈 Avg authors per paper: {stats['basic_stats']['avg_authors_per_paper']:.1f}"
    )

    if stats.get("productivity_types"):
        type_dist = stats["productivity_types"]["type_distribution"]
        print(f"   📋 Author productivity types:")
        for prod_type, count in sorted(
            type_dist.items(), key=lambda x: x[1], reverse=True
        ):
            papers_count = stats["productivity_types"]["papers_by_type"].get(
                prod_type, 0
            )
            print(f"      {prod_type}: {count} authors, {papers_count} papers")

    if stats.get("collaboration_patterns"):
        collab_dist = stats["collaboration_patterns"]["collaboration_type_distribution"]
        print(f"   🤝 Collaboration patterns:")
        for collab_type, count in sorted(
            collab_dist.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"      {collab_type}: {count} papers")

    if stats.get("productivity_analysis"):
        top_authors = stats["productivity_analysis"]["top_authors_by_papers"][:3]
        print(f"   🏆 Top 3 most productive authors:")
        for i, (author, paper_count) in enumerate(top_authors):
            productivity_type = G.nodes[author].get("productivity_type", "")
            print(f"      {i+1}. {author} ({productivity_type}) - {paper_count} papers")

    return G, output_file


if __name__ == "__main__":
    main()
