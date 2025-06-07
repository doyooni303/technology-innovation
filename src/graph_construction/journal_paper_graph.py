"""
저널-논문 이분 그래프 구축 모듈
Journal-Paper Bipartite Graph Construction Module
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import re


class JournalPaperGraphBuilder:
    """저널-논문 이분 그래프를 구축하는 클래스"""

    def __init__(self, min_papers_per_journal=1):
        """
        Args:
            min_papers_per_journal (int): 그래프에 포함할 저널의 최소 논문 수
        """
        self.min_papers_per_journal = min_papers_per_journal

        # 결과 저장용
        self.journal_papers = defaultdict(list)  # 저널별 논문 목록
        self.paper_journals = {}  # 논문별 저널 정보
        self.journal_stats = {}
        self.papers_metadata = None

        # 저널명 약어 -> Full name 매핑
        self.journal_abbreviations = {
            # IEEE 저널들
            r"IEEE Trans\.?\s*Power\s*Syst\.?": "IEEE Transactions on Power Systems",
            r"IEEE Trans\.?\s*Ind\.?\s*Electron\.?": "IEEE Transactions on Industrial Electronics",
            r"IEEE Trans\.?\s*Energy\s*Convers\.?": "IEEE Transactions on Energy Conversion",
            r"IEEE Trans\.?\s*Smart\s*Grid": "IEEE Transactions on Smart Grid",
            r"IEEE Trans\.?\s*Veh\.?\s*Technol\.?": "IEEE Transactions on Vehicular Technology",
            r"IEEE Trans\.?\s*Neural\s*Netw\.?": "IEEE Transactions on Neural Networks",
            r"IEEE Trans\.?\s*Pattern\s*Anal\.?": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
            r"IEEE Trans\.?\s*Autom\.?\s*Sci\.?\s*Eng\.?": "IEEE Transactions on Automation Science and Engineering",
            # 기타 주요 저널들
            r"J\.?\s*Power\s*Sources": "Journal of Power Sources",
            r"Appl\.?\s*Energy": "Applied Energy",
            r"Renew\.?\s*Sustain\.?\s*Energy\s*Rev\.?": "Renewable and Sustainable Energy Reviews",
            r"Energy\s*Build\.?": "Energy and Buildings",
            r"Int\.?\s*J\.?\s*Electr\.?\s*Power": "International Journal of Electrical Power & Energy Systems",
            # Nature, Science 계열
            r"Nat\.?\s*Energy": "Nature Energy",
            r"Nat\.?\s*Commun\.?": "Nature Communications",
            r"Sci\.?\s*Rep\.?": "Scientific Reports",
            # Elsevier 저널들
            r"Expert\s*Syst\.?\s*Appl\.?": "Expert Systems with Applications",
            r"Neurocomputing": "Neurocomputing",
            r"Comput\.?\s*Chem\.?\s*Eng\.?": "Computers & Chemical Engineering",
        }

    def normalize_journal_name(self, journal_name):
        """저널명 정규화 (약어 → Full name)"""
        if not journal_name or not isinstance(journal_name, str):
            return None

        # 기본 정제
        journal_name = journal_name.strip()

        # 너무 짧은 이름 제외
        if len(journal_name) < 3:
            return None

        # 약어를 Full name으로 변환
        normalized = journal_name
        for abbrev_pattern, full_name in self.journal_abbreviations.items():
            if re.search(abbrev_pattern, journal_name, re.IGNORECASE):
                normalized = full_name
                break

        # 기본적인 정제
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # 첫 글자를 대문자로
        if normalized:
            normalized = normalized[0].upper() + normalized[1:]

        return normalized

    def extract_journal_paper_relationships(self, papers_metadata):
        """논문 메타데이터에서 저널-논문 관계 추출"""
        print("🔍 Extracting journal-paper relationships...")

        valid_papers = 0
        papers_with_journals = 0
        unknown_journals = 0

        for i, paper in enumerate(tqdm(papers_metadata, desc="Processing papers")):
            paper_id = f"paper_{i}"
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            journal = paper.get("journal", "")
            year = paper.get("year", "")
            authors = paper.get("authors", [])
            keywords = paper.get("keywords", [])

            # 저널 정보 정제
            if journal:
                normalized_journal = self.normalize_journal_name(journal)
                if normalized_journal:
                    papers_with_journals += 1
                else:
                    unknown_journals += 1
                    normalized_journal = journal  # 정제 실패시 원본 사용
            else:
                unknown_journals += 1
                normalized_journal = "Unknown Journal"

            # 논문-저널 매핑 저장
            self.paper_journals[paper_id] = {
                "title": title,
                "abstract": abstract,
                "journal": normalized_journal,
                "original_journal": journal,
                "year": year,
                "authors": authors,
                "keywords": keywords,
                "author_count": len(authors) if authors else 0,
                "keyword_count": len(keywords) if keywords else 0,
                "has_abstract": bool(abstract.strip()),
                "abstract_length": len(abstract),
            }

            # 저널-논문 매핑 저장
            self.journal_papers[normalized_journal].append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "authors": authors,
                    "keywords": keywords,
                }
            )

            valid_papers += 1

        print(f"✅ Journal-paper relationship extraction completed:")
        print(f"   📄 Total papers processed: {valid_papers}")
        print(f"   📰 Papers with journals: {papers_with_journals}")
        print(f"   ❓ Papers with unknown/missing journals: {unknown_journals}")
        print(f"   📚 Unique journals: {len(self.journal_papers)}")
        print(
            f"   📈 Avg papers per journal: {valid_papers/len(self.journal_papers):.1f}"
        )

    def calculate_journal_statistics(self):
        """저널별 통계 계산"""
        print("📊 Calculating journal statistics...")

        for journal, papers in self.journal_papers.items():
            # 기본 통계
            paper_count = len(papers)
            years = [p["year"] for p in papers if p["year"]]
            authors_all = [author for p in papers for author in p.get("authors", [])]
            keywords_all = [kw for p in papers for kw in p.get("keywords", [])]

            # 활동 기간
            if years:
                try:
                    year_ints = [int(y) for y in years if str(y).isdigit()]
                    if year_ints:
                        first_year = min(year_ints)
                        last_year = max(year_ints)
                        active_years = last_year - first_year + 1
                        # 년도별 논문 수
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

            # 저자 통계
            unique_authors = len(set(authors_all)) if authors_all else 0
            avg_authors_per_paper = (
                len(authors_all) / paper_count if paper_count > 0 else 0
            )

            # 키워드 통계
            unique_keywords = len(set(keywords_all)) if keywords_all else 0
            keyword_frequency = Counter(keywords_all)
            top_keywords = keyword_frequency.most_common(10)

            # 저널 타입 분류
            journal_type = self.classify_journal_type(journal)

            self.journal_stats[journal] = {
                "name": journal,
                "paper_count": paper_count,
                "unique_authors": unique_authors,
                "unique_keywords": unique_keywords,
                "first_year": first_year,
                "last_year": last_year,
                "active_years": active_years,
                "avg_authors_per_paper": avg_authors_per_paper,
                "journal_type": journal_type,
                "papers_by_year": dict(papers_by_year),
                "top_keywords": top_keywords,
                "papers": papers,
            }

        print(
            f"✅ Journal statistics calculated for {len(self.journal_stats)} journals"
        )

    def classify_journal_type(self, journal_name):
        """저널 타입 분류"""
        name_lower = journal_name.lower()

        if "ieee" in name_lower:
            return "IEEE"
        elif any(word in name_lower for word in ["nature", "science"]):
            return "High-Impact"
        elif any(word in name_lower for word in ["energy", "power", "electric"]):
            return "Energy & Power"
        elif any(
            word in name_lower
            for word in ["neural", "machine", "artificial", "intelligence"]
        ):
            return "AI & ML"
        elif any(word in name_lower for word in ["computer", "computing", "software"]):
            return "Computer Science"
        elif any(
            word in name_lower for word in ["engineering", "industrial", "automation"]
        ):
            return "Engineering"
        elif any(
            word in name_lower for word in ["applied", "international", "journal"]
        ):
            return "Applied Science"
        else:
            return "Other"

    def filter_journals_by_activity(self):
        """활동 수준 기준으로 저널 필터링"""
        print(f"🔍 Filtering journals (min papers: {self.min_papers_per_journal})...")

        # 최소 논문 수 이상의 저널만 선택
        active_journals = {
            journal
            for journal, stats in self.journal_stats.items()
            if stats["paper_count"] >= self.min_papers_per_journal
        }

        print(f"   📊 Journals before filtering: {len(self.journal_stats)}")
        print(f"   ✅ Journals after filtering: {len(active_journals)}")

        # 저널 통계도 필터링
        self.journal_stats = {
            journal: stats
            for journal, stats in self.journal_stats.items()
            if journal in active_journals
        }

        self.journal_papers = {
            journal: papers
            for journal, papers in self.journal_papers.items()
            if journal in active_journals
        }

        # 논문-저널 매핑도 필터링
        filtered_paper_journals = {}
        for paper_id, paper_info in self.paper_journals.items():
            if paper_info["journal"] in active_journals:
                filtered_paper_journals[paper_id] = paper_info

        self.paper_journals = filtered_paper_journals

        print(f"   📄 Active journals remaining: {len(self.journal_stats)}")
        print(f"   📰 Papers with active journals: {len(self.paper_journals)}")

        return active_journals

    def build_journal_paper_graph(self):
        """저널-논문 이분 그래프 구축"""
        print(f"🔗 Building journal-paper bipartite graph...")

        # 방향 그래프 생성 (Paper → Journal)
        G = nx.DiGraph()

        # 저널 노드 추가
        for journal, stats in self.journal_stats.items():
            G.add_node(
                journal,
                node_type="journal",
                name=journal,
                paper_count=stats["paper_count"],
                unique_authors=stats["unique_authors"],
                unique_keywords=stats["unique_keywords"],
                first_year=stats["first_year"],
                last_year=stats["last_year"],
                active_years=stats["active_years"],
                avg_authors_per_paper=stats["avg_authors_per_paper"],
                journal_type=stats["journal_type"],
                top_keywords=[kw for kw, count in stats["top_keywords"]],
            )

        # 논문 노드 추가
        for paper_id, paper_info in self.paper_journals.items():
            G.add_node(
                paper_id,
                node_type="paper",
                title=paper_info["title"],
                abstract=paper_info.get("abstract", ""),
                year=paper_info["year"],
                author_count=paper_info["author_count"],
                keyword_count=paper_info["keyword_count"],
                authors=paper_info["authors"][:5],  # 처음 5명만 저장
                keywords=paper_info["keywords"][:10],
                has_abstract=paper_info.get("has_abstract", False),
                abstract_length=paper_info.get("abstract_length", 0),
            )  # 처음 10개만 저장

        # Published_in 엣지 추가 (Paper → Journal)
        edges_added = 0

        for paper_id, paper_info in self.paper_journals.items():
            journal = paper_info["journal"]
            year = paper_info["year"]

            if G.has_node(paper_id) and G.has_node(journal):
                G.add_edge(
                    paper_id, journal, edge_type="published_in", year=year, weight=1.0
                )  # 단순 관계이므로 가중치는 1

                edges_added += 1

        print(f"✅ Journal-paper bipartite graph constructed:")
        print(
            f"   📰 Journal nodes: {sum(1 for n in G.nodes() if G.nodes[n]['node_type'] == 'journal')}"
        )
        print(
            f"   📄 Paper nodes: {sum(1 for n in G.nodes() if G.nodes[n]['node_type'] == 'paper')}"
        )
        print(f"   🔗 Published_in edges: {G.number_of_edges()}")
        print(f"   📈 Graph density: {nx.density(G):.6f}")

        return G

    def analyze_graph_properties(self, G):
        """그래프 속성 분석"""
        print("📈 Analyzing journal-paper graph properties...")

        # 노드별 통계
        journal_nodes = [n for n in G.nodes() if G.nodes[n]["node_type"] == "journal"]
        paper_nodes = [n for n in G.nodes() if G.nodes[n]["node_type"] == "paper"]

        # 기본 통계
        stats = {
            "basic_stats": {
                "num_journals": len(journal_nodes),
                "num_papers": len(paper_nodes),
                "num_edges": G.number_of_edges(),
                "density": float(nx.density(G)),
                "avg_papers_per_journal": (
                    G.number_of_edges() / len(journal_nodes) if journal_nodes else 0
                ),
            }
        }

        # 저널별 논문 수 분석
        journal_paper_counts = {}
        for journal in journal_nodes:
            paper_count = G.in_degree(journal)  # 들어오는 엣지 = 발표된 논문 수
            journal_paper_counts[journal] = paper_count

        stats["journal_analysis"] = {
            "top_journals_by_papers": sorted(
                journal_paper_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "journal_paper_distribution": {
                "mean": float(np.mean(list(journal_paper_counts.values()))),
                "median": float(np.median(list(journal_paper_counts.values()))),
                "max": max(journal_paper_counts.values()),
                "min": min(journal_paper_counts.values()),
            },
        }

        # 저널 타입별 분석
        journal_type_analysis = self.analyze_journal_types(G)
        stats["journal_types"] = journal_type_analysis

        # 시간적 분석
        temporal_analysis = self.analyze_temporal_patterns(G)
        stats["temporal_patterns"] = temporal_analysis

        return stats

    def analyze_journal_types(self, G):
        """저널 타입별 분석"""
        type_counts = Counter()
        type_papers = defaultdict(int)

        for node in G.nodes():
            if G.nodes[node]["node_type"] == "journal":
                journal_type = G.nodes[node]["journal_type"]
                paper_count = G.in_degree(node)

                type_counts[journal_type] += 1
                type_papers[journal_type] += paper_count

        return {
            "type_distribution": dict(type_counts),
            "papers_by_type": dict(type_papers),
        }

    def analyze_temporal_patterns(self, G):
        """시간적 패턴 분석"""
        papers_by_year = defaultdict(int)
        journals_by_year = defaultdict(set)

        for paper_id in G.nodes():
            if G.nodes[paper_id]["node_type"] == "paper":
                year = G.nodes[paper_id].get("year", "")
                if year and str(year).isdigit():
                    year_int = int(year)
                    papers_by_year[year_int] += 1

                    # 해당 논문이 발표된 저널 찾기
                    for journal in G.successors(paper_id):
                        journals_by_year[year_int].add(journal)

        # 년도별 저널 수 계산
        journals_count_by_year = {
            year: len(journals) for year, journals in journals_by_year.items()
        }

        return {
            "papers_by_year": dict(papers_by_year),
            "journals_count_by_year": journals_count_by_year,
            "active_years": list(papers_by_year.keys()) if papers_by_year else [],
        }

    def create_node_info_maps(self, G):
        """노드 정보를 쉽게 조회할 수 있는 맵 생성"""
        journal_info = {}
        paper_info = {}

        # 저널 정보
        for node in G.nodes():
            if G.nodes[node]["node_type"] == "journal":
                node_data = G.nodes[node]
                journal_info[node] = {
                    "name": node,
                    "journal_type": node_data.get("journal_type", ""),
                    "paper_count": G.in_degree(node),
                    "unique_authors": node_data.get("unique_authors", 0),
                    "first_year": node_data.get("first_year", ""),
                    "last_year": node_data.get("last_year", ""),
                    "active_years": node_data.get("active_years", 0),
                    "top_keywords": node_data.get("top_keywords", [])[:5],  # 상위 5개만
                }

        # 논문 정보
        for node in G.nodes():
            if G.nodes[node]["node_type"] == "paper":
                node_data = G.nodes[node]
                # 해당 논문의 저널 찾기
                journals = list(G.successors(node))
                journal_name = journals[0] if journals else "Unknown"

                paper_info[node] = {
                    "title": node_data.get("title", ""),
                    "abstract": node_data.get("abstract", ""),
                    "journal": journal_name,
                    "year": node_data.get("year", ""),
                    "author_count": node_data.get("author_count", 0),
                    "keyword_count": node_data.get("keyword_count", 0),
                    "authors": node_data.get("authors", []),
                    "has_abstract": node_data.get("has_abstract", False),
                    "abstract_length": node_data.get("abstract_length", 0),
                }

        return journal_info, paper_info

    def save_journal_paper_graph_and_analysis(self, G, stats, output_dir):
        """Journal-paper 그래프와 분석 결과 저장"""
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
        graph_file = output_dir / "journal_paper_graph.json"
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        # 2. ✅ GraphML 파일로 저장 (XML 호환성 개선)
        try:
            G_graphml = G.copy()

            # XML 호환 문자열 정제 함수
            def clean_xml_string(text):
                if not isinstance(text, str):
                    text = str(text)

                # NULL 바이트 및 제어 문자 제거
                import re

                # XML 1.0에서 허용되지 않는 문자들 제거
                text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

                # XML 특수 문자 이스케이프
                text = text.replace("&", "&amp;")
                text = text.replace("<", "&lt;")
                text = text.replace(">", "&gt;")
                text = text.replace('"', "&quot;")
                text = text.replace("'", "&apos;")

                return text

            # 모든 노드 속성 정제
            for node in G_graphml.nodes():
                for attr_name, attr_value in G_graphml.nodes[node].items():
                    if isinstance(attr_value, list):
                        # 리스트를 세미콜론으로 구분된 문자열로 변환 후 정제
                        cleaned_list = [clean_xml_string(str(v)) for v in attr_value]
                        G_graphml.nodes[node][attr_name] = ";".join(cleaned_list)
                    elif isinstance(attr_value, str):
                        # 문자열 정제 (Abstract 포함)
                        G_graphml.nodes[node][attr_name] = clean_xml_string(attr_value)
                    elif not isinstance(attr_value, (int, float, bool)):
                        # 기타 타입을 문자열로 변환 후 정제
                        G_graphml.nodes[node][attr_name] = clean_xml_string(
                            str(attr_value)
                        )

            # 엣지 속성도 정제
            for u, v in G_graphml.edges():
                for attr_name, attr_value in G_graphml.edges[u, v].items():
                    if isinstance(attr_value, str):
                        G_graphml.edges[u, v][attr_name] = clean_xml_string(attr_value)
                    elif not isinstance(attr_value, (int, float, bool)):
                        G_graphml.edges[u, v][attr_name] = clean_xml_string(
                            str(attr_value)
                        )

            graphml_file = output_dir / "journal_paper_graph.graphml"
            nx.write_graphml(
                G_graphml, graphml_file, encoding="utf-8", prettyprint=True
            )
            print(f"   🔗 Graph (GraphML): {graphml_file}")

        except Exception as e:
            print(f"⚠️  GraphML 저장 중 오류 발생: {e}")
            print(f"📄 JSON 파일은 정상적으로 저장되었습니다: {graph_file}")
            graphml_file = None

        # 3. 분석 결과 저장
        stats_file = output_dir / "journal_paper_analysis.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # 4. 노드 정보 테이블들 저장
        journal_info, paper_info = self.create_node_info_maps(G)

        journal_info_file = output_dir / "journal_paper_journal_info.json"
        with open(journal_info_file, "w", encoding="utf-8") as f:
            json.dump(journal_info, f, ensure_ascii=False, indent=2)

        paper_info_file = output_dir / "journal_paper_paper_info.json"
        with open(paper_info_file, "w", encoding="utf-8") as f:
            json.dump(paper_info, f, ensure_ascii=False, indent=2)

        # 5. CSV 형태의 엣지 리스트 저장
        edge_list = []
        for edge in G.edges():
            edge_info = G.edges[edge].copy()
            edge_info.update(
                {
                    "paper_id": edge[0],
                    "journal_name": edge[1],
                    "paper_title": G.nodes[edge[0]].get("title", ""),
                    "journal_type": G.nodes[edge[1]].get("journal_type", ""),
                    "paper_year": G.nodes[edge[0]].get("year", ""),
                    "has_abstract": G.nodes[edge[0]].get("has_abstract", False),
                    "abstract_length": G.nodes[edge[0]].get("abstract_length", 0),
                }
            )
            edge_list.append(edge_info)

        edge_df = pd.DataFrame(edge_list)
        edge_file = output_dir / "journal_paper_edges.csv"
        edge_df.to_csv(edge_file, index=False, encoding="utf-8")

        print(f"💾 Journal-paper graph results saved:")
        print(f"   🔗 Graph (JSON): {graph_file}")
        if graphml_file:
            print(f"   🔗 Graph (GraphML): {graphml_file}")
        print(f"   📊 Analysis: {stats_file}")
        print(f"   📰 Journal Info: {journal_info_file}")
        print(f"   📄 Paper Info: {paper_info_file}")
        print(f"   📈 Edge List: {edge_file}")

        return graph_file

    def process_papers(self, papers_metadata):
        """전체 처리 파이프라인"""
        print("🚀 Starting journal-paper graph construction...")

        # 메타데이터 저장
        self.papers_metadata = papers_metadata

        # 1. 저널-논문 관계 추출
        self.extract_journal_paper_relationships(papers_metadata)

        # 2. 저널 통계 계산
        self.calculate_journal_statistics()

        # 3. 활동 수준 기준 필터링
        active_journals = self.filter_journals_by_activity()

        # 4. 이분 그래프 구축
        G = self.build_journal_paper_graph()

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

    # Journal-Paper Graph Builder 초기화
    builder = JournalPaperGraphBuilder(
        min_papers_per_journal=1  # 최소 1편 이상 논문이 있는 저널만 포함
    )

    # 전체 처리
    G, stats = builder.process_papers(papers_metadata)

    # 결과 저장
    output_file = builder.save_journal_paper_graph_and_analysis(G, stats, GRAPHS_DIR)

    print(f"\n✅ Journal-paper graph construction completed!")
    print(f"📁 Main output: {output_file}")

    # 요약 통계 출력
    print(f"\n📊 Final Summary:")
    print(f"   📰 Total journals: {stats['basic_stats']['num_journals']}")
    print(f"   📄 Total papers: {stats['basic_stats']['num_papers']}")
    print(f"   🔗 Published_in relationships: {stats['basic_stats']['num_edges']}")
    print(
        f"   📈 Avg papers per journal: {stats['basic_stats']['avg_papers_per_journal']:.1f}"
    )

    if stats.get("journal_types"):
        type_dist = stats["journal_types"]["type_distribution"]
        print(f"   📋 Journal types:")
        for journal_type, count in sorted(
            type_dist.items(), key=lambda x: x[1], reverse=True
        ):
            papers_count = stats["journal_types"]["papers_by_type"].get(journal_type, 0)
            print(f"      {journal_type}: {count} journals, {papers_count} papers")

    if stats.get("journal_analysis"):
        top_journals = stats["journal_analysis"]["top_journals_by_papers"][:3]
        print(f"   🏆 Top 3 most active journals:")
        for i, (journal, paper_count) in enumerate(top_journals):
            journal_type = G.nodes[journal].get("journal_type", "")
            print(f"      {i+1}. {journal} ({journal_type}) - {paper_count} papers")

    return G, output_file


if __name__ == "__main__":
    main()
