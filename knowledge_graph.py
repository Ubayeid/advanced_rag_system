import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Tuple
import numpy as np
from collections import defaultdict, Counter
import json
from datetime import datetime
import spacy
import os
import logging

logger = logging.getLogger(__name__)

class KnowledgeGraphVisualizer:
    """Enhanced visualizer for the NetworkX-based knowledge graph"""

    def __init__(self, kg_manager):
        self.kg_manager = kg_manager
        self.graph = kg_manager.graph
        try:
            self.nlp = spacy.load(os.getenv("SPACY_MODEL", "en_core_web_sm"))
            self.nlp.max_length = 2_000_000
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found for KnowledgeGraphVisualizer. Visualization might be limited. Please install with: python -m spacy download en_core_web_sm")
            self.nlp = None

    def visualize_graph_static(self, figsize=(15, 10), save_path=None):
        """Create a static visualization using matplotlib"""

        if len(self.graph.nodes()) == 0:
            logger.warning("No nodes in the graph to visualize for static plot.")
            return

        plt.figure(figsize=figsize)

        pos = nx.spring_layout(self.graph, k=0.5, iterations=50, seed=42)

        entity_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n].get('type') not in ['CHUNK', 'Document']]
        chunk_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n].get('type') == 'CHUNK']

        # Draw nodes
        node_labels = {}
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type')
            if node_type == 'CHUNK':
                content_snippet = self.graph.nodes[node].get('content', '')
                node_labels[node] = content_snippet.split('...')[0] if '...' in content_snippet else content_snippet[:20] + '...' # Safer snippet
            else:
                node_labels[node] = self.graph.nodes[node].get('name', node)

        nx.draw_networkx_nodes(self.graph, pos, nodelist=entity_nodes, node_color='lightblue', node_size=500, alpha=0.8, label='Entities', edgecolors='black')
        nx.draw_networkx_nodes(self.graph, pos, nodelist=chunk_nodes, node_color='lightcoral', node_size=300, alpha=0.8, label='Chunks', edgecolors='black')

        # Draw edges
        edge_labels = nx.get_edge_attributes(self.graph, 'relation_type')
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, width=1, arrowsize=10)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=7, alpha=0.7)

        # Draw node labels
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=7, font_color='black')

        plt.title("Knowledge Graph Visualization", size=16)
        plt.legend()
        plt.axis('off')
        plt.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Static graph saved to {save_path}")
            except Exception as e:
                logger.error(f"Error saving static graph to {save_path}: {e}")

    def visualize_graph_interactive(self, height=800):
        """Create an interactive visualization using Plotly"""

        if len(self.graph.nodes()) == 0:
            logger.warning("No nodes in the graph to visualize for interactive plot.")
            return None

        # Create layout
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50, seed=42) # Fixed seed for consistency

        # Prepare node data
        node_x = []
        node_y = []
        node_text = [] # For hover text
        node_size = []
        node_color = []
        node_type_map = {
            'CHUNK': 'lightcoral', 'PERSON': 'blue', 'ORG': 'green', 'GPE': 'purple',
            'STUDY': 'orange', 'WATER_BODY': 'cyan', 'HYDRO_MEASUREMENT': 'yellow',
            'POLLUTANT': 'red', 'HYDRO_EVENT': 'magenta', 'HYDRO_INFRASTRUCTURE': 'brown',
            'HYDRO_MODEL': 'lime', 'MISC': 'gray', 'DATE': 'darkblue', 'MONEY': 'darkgreen',
            'DATASET': 'teal', # Added DATASET
            'DEFAULT': 'lightgray'
        } # More specific colors

        for node_id in self.graph.nodes():
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)

            node_info = self.graph.nodes[node_id]
            node_type = node_info.get('type', 'DEFAULT')
            node_name = node_info.get('name', node_id) # Use name for entities, ID for chunks

            # For chunks, use a content snippet in hover text
            if node_type == 'CHUNK':
                display_name = f"Chunk: {node_id[:8]}..."
                hover_content = (
                    f"<b>ID:</b> {node_id}<br>"
                    f"<b>Type:</b> {node_type}<br>"
                    f"<b>Document:</b> {node_info.get('document_id', 'N/A')}<br>"
                    f"<b>Page:</b> {node_info.get('page_number', 'N/A')}<br>"
                    f"<b>Section:</b> {node_info.get('section_title', 'N/A')}<br>"
                    f"<b>Content:</b> {node_info.get('content', 'N/A')}"
                )
            else:
                display_name = f"{node_name}" # Display full name for entities
                hover_content = (
                    f"<b>Name:</b> {node_name}<br>"
                    f"<b>ID:</b> {node_id}<br>"
                    f"<b>Type:</b> {node_type}<br>"
                    f"<b>Documents:</b> {', '.join(node_info.get('document_ids', ['N/A']))}"
                )

            # Add relations that this node is part of to hover info
            incoming_edges = self.graph.in_edges(node_id, data=True)
            outgoing_edges = self.graph.out_edges(node_id, data=True)

            if incoming_edges or outgoing_edges:
                hover_content += "<br><b>Relations:</b>"
                for u, v, data in incoming_edges:
                    source_node_display = self.graph.nodes[u].get('name', u[:8] + '...') if self.graph.nodes[u].get('type') != 'CHUNK' else f"Chunk:{u[:8]}..."
                    hover_content += f"<br> &larr; {data.get('relation_type', 'related to')} {source_node_display}"
                for u, v, data in outgoing_edges:
                    target_node_display = self.graph.nodes[v].get('name', v[:8] + '...') if self.graph.nodes[v].get('type') != 'CHUNK' else f"Chunk:{v[:8]}..."
                    hover_content += f"<br> &rarr; {data.get('relation_type', 'related to')} {target_node_display}"

            node_text.append(hover_content)

            connections = self.graph.degree(node_id) # Total number of connections
            node_size.append(10 + connections * 3) # Scale size by connections
            node_color.append(node_type_map.get(node_type, node_type_map['DEFAULT']))

        node_trace = go.Scatter(
            x=node_x, y=node_y, text=node_text, mode='markers',
            hoverinfo='text',
            marker=dict(size=node_size, color=node_color, line_width=1, line_color='DarkSlateGrey')
        )

        # Prepare edge data with arrow annotations for direction
        edge_x = []
        edge_y = []
        edge_annotations = []

        for edge in self.graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2

            # Adjust offsets for text to be slightly off the center of the line
            text_x_offset = (x1 - x0) * 0.1
            text_y_offset = (y1 - y0) * 0.1

            edge_length = np.sqrt((x1-x0)**2 + (y1-y0)**2)
            # Avoid division by zero if nodes are at the same position
            if edge_length > 0:
                # Calculate angle for text rotation
                angle_rad = np.arctan2(y1 - y0, x1 - x0)
                angle_deg = np.degrees(angle_rad)
                # Adjust for text upside down if necessary
                if angle_deg > 90 or angle_deg < -90:
                    angle_deg += 180

                edge_annotations.append(
                    go.layout.Annotation(
                        x=(mid_x + text_x_offset), # X-coordinate for text anchor and arrow head
                        y=(mid_y + text_y_offset), # Y-coordinate for text anchor and arrow head
                        xref='x', yref='y', # Reference to the plot axes

                        ax=x0, ay=y0, # Arrow tail at source node's original x, y
                        axref='x', ayref='y', # Reference to the plot axes

                        showarrow=True,
                        arrowhead=2, # Triangle arrow
                        arrowsize=1, arrowwidth=1, arrowcolor='gray',
                        opacity=0.7,

                        text=edge[2].get('relation_type', 'rel'),
                        font=dict(size=8, color='darkslategray'),
                        textangle=angle_deg, # Use calculated angle
                        xanchor="center", yanchor="bottom",
                    )
                )

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(width=0.8, color='gray'),
            hoverinfo='none',
            opacity=0.7
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
                            title=dict(text="Interactive Knowledge Graph", font=dict(size=16)),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=edge_annotations + [ # Combine edge annotations with general annotations
                                dict(
                                    text="Node size = number of connections",
                                    showarrow=False,
                                    xref="paper", yref="paper",
                                    x=0.005, y=-0.002,
                                    xanchor="left", yanchor="bottom",
                                    font=dict(color="black", size=10)
                                ),
                                # Add legend for node colors
                                dict(
                                    text="<b>Node Types:</b><br>" + "<br>".join([
                                        f"<span style='color:{color}'>&#9632;</span> {node_type}" for node_type, color in node_type_map.items() if node_type != 'DEFAULT'
                                    ]),
                                    showarrow=False,
                                    xref="paper", yref="paper",
                                    x=1.0, y=0.98,
                                    xanchor="right", yanchor="top",
                                    align="left",
                                    bgcolor="rgba(255,255,255,0.7)",
                                    bordercolor="black", borderwidth=1,
                                    font=dict(size=9)
                                )
                            ],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=height
                        ))

        # --- IMPORTANT: We only write to HTML. Do NOT call fig.show() here. ---
        # Plotly.io.show() will try to open a browser, which fails in headless Docker containers.
        try:
            output_html_file = os.path.join(os.getenv("STORAGE_PATH", "./storage"), "knowledge_graph_interactive.html")
            fig.write_html(output_html_file)
            # logger.info(f"Interactive graph saved to {output_html_file}")
        except Exception as e:
            logger.error(f"Error saving interactive graph to HTML: {e}")
            print(f"Error saving interactive graph to HTML: {e}")
            return None

        return fig

class KnowledgeGapAnalyzer:
    """Comprehensive knowledge gap analysis"""

    def __init__(self, kg_manager):
        self.kg_manager = kg_manager
        self.graph = kg_manager.graph

    def analyze_gaps(self) -> Dict[str, Any]:
        """Comprehensive gap analysis"""
        analysis = {
            'structural_gaps': self._analyze_structural_gaps(),
            'content_gaps': self._analyze_content_gaps(),
            'connectivity_gaps': self._analyze_connectivity_gaps(),
            'entity_coverage': self._analyze_entity_coverage(),
            'recommendations': []
        }

        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)

        return analysis

    def _analyze_structural_gaps(self) -> Dict[str, Any]:
        """Analyze structural issues in the graph"""
        total_nodes = len(self.graph.nodes())
        total_edges = len(self.graph.edges())

        if total_nodes == 0:
            return {'error': 'No nodes in graph'}

        # Find isolated nodes (degree == 0 means no incoming or outgoing edges)
        isolated_nodes = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]

        components = list(nx.weakly_connected_components(self.graph))

        largest_component_size = max(len(c) for c in components) if components else 0

        # Calculate density
        density = nx.density(self.graph)

        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'isolated_nodes': len(isolated_nodes),
            'isolated_node_list': [self.graph.nodes[n].get('name', n[:20] + '...') if self.graph.nodes[n].get('type') != 'CHUNK' else f"Chunk:{n[:8]}..." for n in isolated_nodes[:10]], # Show first 10 names/snippets
            'connected_components': len(components),
            'largest_component_size': largest_component_size,
            'graph_density': density,
            'connectivity_ratio': largest_component_size / total_nodes if total_nodes > 0 else 0
        }

    def _analyze_content_gaps(self) -> Dict[str, Any]:
        """Analyze content-related gaps"""
        entity_nodes = []
        document_nodes = []

        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            # Use 'type' attribute to distinguish entities from chunks
            if node_data.get('type') and node_data.get('type') != 'CHUNK':
                entity_nodes.append(node)
            elif node_data.get('type') == 'CHUNK':
                document_nodes.append(node)

        # Entity type distribution
        entity_types = defaultdict(int)
        for node in entity_nodes:
            entity_type = self.graph.nodes[node].get('type', 'UNKNOWN')
            entity_types[entity_type] += 1

        # Find entities with few connections
        sparse_entities = []
        for entity in entity_nodes:
            connections = self.graph.degree(entity) # Use .degree() for total connections
            if connections < 2: # Entities with 0 or 1 connection
                sparse_entities.append({
                    'entity_id': entity,
                    'name': self.graph.nodes[entity].get('name', entity),
                    'connections': connections,
                    'type': self.graph.nodes[entity].get('type', 'UNKNOWN')
                })

        return {
            'total_entities': len(entity_nodes),
            'total_documents': len(document_nodes),
            'entity_type_distribution': dict(entity_types),
            'sparse_entities': sorted(sparse_entities, key=lambda x: x['connections'])[:20], # Top 20 sparse entities, sorted by connections
            'entities_per_document': len(entity_nodes) / len(document_nodes) if document_nodes else 0
        }

    def _analyze_connectivity_gaps(self) -> Dict[str, Any]:
        """Analyze connectivity patterns"""
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]

        if not degrees:
            return {'error': 'No nodes to analyze'}

        # For bridges and articulation points, convert to an undirected graph for proper calculation
        undirected_graph = self.graph.to_undirected()

        # Check if the undirected graph is connected before finding bridges
        # Bridges only make sense in a connected component
        bridges = []
        if nx.is_connected(undirected_graph):
            bridges = list(nx.bridges(undirected_graph))
        else:
            logger.info("Graph is not connected, cannot accurately identify all bridges globally. Analyzing within components.")
            # Optionally, analyze bridges per connected component if desired for more granular info
            for component_nodes in nx.connected_components(undirected_graph):
                subgraph = undirected_graph.subgraph(component_nodes)
                bridges.extend(list(nx.bridges(subgraph)))


        # Find articulation points
        articulation_points = list(nx.articulation_points(undirected_graph))

        # Calculate centrality measures
        try:
            # Use the undirected_graph for centrality measures for consistency with bridges/articulation points
            if undirected_graph.number_of_nodes() > 0 and undirected_graph.number_of_edges() > 0:
                betweenness = nx.betweenness_centrality(undirected_graph)
                closeness = nx.closeness_centrality(undirected_graph)

                top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
                top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]
            else:
                betweenness, closeness = {}, {}
                top_betweenness, top_closeness = [], []

        except Exception as e:
            logger.warning(f"Error calculating centrality measures: {e}. Returning empty lists for centrality.")
            top_betweenness = []
            top_closeness = []

        return {
            'average_degree': np.mean(degrees),
            'degree_distribution': {
                'min': min(degrees),
                'max': max(degrees),
                'mean': np.mean(degrees),
                'median': np.median(degrees),
                'std': np.std(degrees)
            },
            'bridges': len(bridges),
            'articulation_points': len(articulation_points),
            'critical_nodes_examples': [self.graph.nodes[n].get('name', n[:20] + '...') if self.graph.nodes[n].get('type') != 'CHUNK' else f"Chunk:{n[:8]}..." for n in articulation_points[:10]],
            'top_betweenness_centrality': top_betweenness,
            'top_closeness_centrality': top_closeness
        }

    def _analyze_entity_coverage(self) -> Dict[str, Any]:
        """Analyze entity coverage across documents"""
        entity_doc_map = defaultdict(set) # Entity ID -> set of Document IDs
        doc_entity_map = defaultdict(set) # Document ID -> set of Entity IDs

        # Iterate through all nodes that are entities and their 'MENTIONED_IN' edges to chunks
        for entity_id, entity_data in self.kg_manager.entities.items():
            for chunk_id in entity_data.related_chunks:
                # Ensure the chunk_id exists in document_chunks before trying to access its properties
                if chunk_id in self.kg_manager.document_chunks:
                    entity_doc_map[entity_id].add(self.kg_manager.document_chunks[chunk_id].document_id)
                    doc_entity_map[self.kg_manager.document_chunks[chunk_id].document_id].add(entity_id)
                else:
                    logger.warning(f"Chunk ID {chunk_id} not found in kg_manager.document_chunks for entity {entity_id}. Skipping related document link.")

        # Find entities mentioned in multiple documents
        cross_document_entities = {e_id: docs for e_id, docs in entity_doc_map.items() if len(docs) > 1}

        # Find documents with few entities (e.g., less than 3 unique entities linked to them)
        sparse_documents = {d_id: entities for d_id, entities in doc_entity_map.items() if len(entities) < 3}

        return {
            'cross_document_entities_count': len(cross_document_entities),
            'cross_document_entity_examples': [self.kg_manager.entities[e_id].name for e_id in list(cross_document_entities.keys())[:10]],
            'sparse_documents_count': len(sparse_documents),
            'sparse_document_examples': list(sparse_documents.keys())[:10],
            'average_entities_per_document': np.mean([len(entities) for entities in doc_entity_map.values()]) if doc_entity_map else 0
        }

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        structural = analysis.get('structural_gaps', {})
        content = analysis.get('content_gaps', {})
        connectivity = analysis.get('connectivity_gaps', {})
        coverage = analysis.get('entity_coverage', {})

        # Structural recommendations
        if structural.get('isolated_nodes', 0) > 0:
            recommendations.append(f"Address {structural['isolated_nodes']} isolated nodes ({', '.join(structural['isolated_node_list'][:3])}...) by adding more contextual information in new documents or by enriching existing ones. These nodes are not connected to the main graph.")

        if structural.get('connectivity_ratio', 0) < 0.7:
            recommendations.append(f"Improve overall graph connectivity (ratio: {structural.get('connectivity_ratio', 0):.2%}) by ensuring entities and chunks have more cross-references. This will help retrieve more related information.")

        # Content recommendations
        if content.get('sparse_entities', []):
            recommendations.append(f"Expand context for {len(content['sparse_entities'])} sparsely connected entities (e.g., {content['sparse_entities'][0]['name']} with {content['sparse_entities'][0]['connections']} connections). Focus on adding documents or relations that link these entities to others.")

        if content.get('entities_per_document', 0) < 3 and content.get('total_documents', 0) > 0:
            recommendations.append(f"The average number of entities per chunk/document is low ({content.get('entities_per_document', 0):.2f}). Consider refining entity extraction or adding richer documents.")

        # Connectivity recommendations
        if connectivity.get('articulation_points', 0) > 0:
            recommendations.append(f"Strengthen connections around {connectivity['articulation_points']} critical nodes ({', '.join(connectivity['critical_nodes_examples'][:3])}...). These nodes are crucial for graph connectivity, and their removal would partition the graph.")

        if connectivity.get('bridges', 0) > 0:
            recommendations.append(f"There are {connectivity['bridges']} bridge edges. Consider adding redundant paths to connect the components linked by these bridges for improved robustness.")

        # Coverage recommendations
        if coverage.get('cross_document_entities_count', 0) < 5 and coverage.get('total_entities', 0) > 0:
            recommendations.append(f"Only {coverage.get('cross_document_entities_count', 0)} entities appear in multiple documents. Add more documents that share common entities to improve knowledge linking and retrieval across your corpus.")

        if coverage.get('sparse_documents_count', 0) > 0:
            recommendations.append(f"There are {coverage['sparse_documents_count']} documents ({', '.join(coverage['sparse_document_examples'][:3])}...) with very few entities. Review these documents for better entity extraction or consider if they are truly relevant.")

        if not recommendations:
            recommendations.append("Knowledge graph structure looks healthy, well-connected, and rich in entities!")

        return recommendations

    def create_gap_report(self, save_path=None) -> str:
        """Create a comprehensive gap analysis report"""
        analysis = self.analyze_gaps()

        report = f"""
# Knowledge Graph Gap Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
{self._create_executive_summary(analysis)}

---

## Structural Analysis
{self._format_structural_analysis(analysis['structural_gaps'])}

---

## Content Analysis
{self._format_content_analysis(analysis['content_gaps'])}

---

## Connectivity Analysis
{self._format_connectivity_analysis(analysis['connectivity_gaps'])}

---

## Entity Coverage Analysis
{self._format_coverage_analysis(analysis['entity_coverage'])}

---

## Recommendations
{self._format_recommendations(analysis['recommendations'])}
"""

        if save_path:
            try:
                # Ensure the directory for save_path exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                # logger.info(f"Knowledge gap report saved to {save_path}")
            except Exception as e:
                logger.error(f"Error saving knowledge gap report to {save_path}: {e}")

        return report

    def _create_executive_summary(self, analysis):
        structural = analysis.get('structural_gaps', {})
        return f"""
The knowledge graph contains **{structural.get('total_nodes', 0)} nodes** and **{structural.get('total_edges', 0)} edges**.
Graph connectivity ratio: **{structural.get('connectivity_ratio', 0):.2%}**
Overall graph density: **{structural.get('graph_density', 0):.4f}**
"""

    def _format_structural_analysis(self, structural):
        return f"""
- **Total nodes**: {structural.get('total_nodes', 0)}
- **Total edges**: {structural.get('total_edges', 0)}
- **Isolated nodes**: {structural.get('isolated_nodes', 0)} (Examples: {', '.join(structural.get('isolated_node_list', []))})
- **Connected components**: {structural.get('connected_components', 0)}
- **Largest component size**: {structural.get('largest_component_size', 0)}
- **Graph density**: {structural.get('graph_density', 0):.4f}
"""

    def _format_content_analysis(self, content):
        return f"""
- **Total entities**: {content.get('total_entities', 0)}
- **Total documents (chunks)**: {content.get('total_documents', 0)}
- **Average entities per document**: {content.get('entities_per_document', 0):.2f}
- **Sparse entities** (< 2 connections): {len(content.get('sparse_entities', []))} (Examples: {', '.join([e['name'] for e in content['sparse_entities'][:3]])})
- **Entity type distribution**: {content.get('entity_type_distribution', {})}
"""

    def _format_connectivity_analysis(self, connectivity):
        return f"""
- **Average node degree**: {connectivity.get('average_degree', 0):.2f}
- **Bridge connections**: {connectivity.get('bridges', 0)}
- **Critical nodes (articulation points)**: {connectivity.get('articulation_points', 0)} (Examples: {', '.join(connectivity.get('critical_nodes_examples', []))})
- **Degree distribution**: Min={connectivity.get('degree_distribution', {}).get('min', 'N/A')}, Max={connectivity.get('degree_distribution', {}).get('max', 'N/A')}, Mean={connectivity.get('degree_distribution', {}).get('mean', 'N/A'):.2f}, Median={connectivity.get('degree_distribution', {}).get('median', 'N/A'):.2f}, Std={connectivity.get('degree_distribution', {}).get('std', 'N/A'):.2f}
"""

    def _format_coverage_analysis(self, coverage):
        return f"""
- **Cross-document entities**: {coverage.get('cross_document_entities_count', 0)} (Examples: {', '.join(coverage.get('cross_document_entity_examples', []))})
- **Sparse documents** (< 3 entities): {coverage.get('sparse_documents_count', 0)} (Examples: {', '.join(coverage.get('sparse_document_examples', []))})
- **Average entities per document (linked)**: {coverage.get('average_entities_per_document', 0):.2f}
"""

    def _format_recommendations(self, recommendations):
        return '\n'.join([f"- {rec}" for rec in recommendations])

def analyze_rag_knowledge_graph(rag_system):
    """Analyze and visualize the knowledge graph from a RAG system"""

    # Create visualizer and analyzer
    visualizer = KnowledgeGraphVisualizer(rag_system.kg_manager)
    analyzer = KnowledgeGapAnalyzer(rag_system.kg_manager)

    # print("\n--- Knowledge Graph Analysis ---")

    # 1. Create interactive visualization
    print("Creating interactive visualization (saved to knowledge_graph_interactive.html)...")
    # The visualize_graph_interactive function now handles saving to STORAGE_PATH
    # and no longer calls fig.show()
    fig = visualizer.visualize_graph_interactive()
    if fig:
        logger.info("Interactive visualization HTML file generated in your 'storage' directory.")

    # 2. Generate gap analysis report
    print("\nGenerating comprehensive gap analysis report...")
    report_save_path = os.path.join(os.getenv("STORAGE_PATH", "./storage"), "knowledge_gap_report.md")
    report = analyzer.create_gap_report(report_save_path)
    print("Knowledge gap report saved to knowledge_gap_report.md")
    # print(report)

    # 3. Get detailed analysis (returned by this function if needed elsewhere)
    detailed_analysis = analyzer.analyze_gaps()

    return {
        'visualization_figure': fig, # The Plotly figure object (can be None if generation failed)
        'gap_analysis_data': detailed_analysis,
        'report_content': report
    }

if __name__ == "__main__":
    logger.info("Run `docker-compose up` to trigger RAG system initialization and graph analysis.")