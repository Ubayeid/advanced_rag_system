# knowledge_graph.py
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
from dotenv import load_dotenv

# Neo4j specific import
from neo4j import GraphDatabase, basic_auth, exceptions as neo4j_exceptions

logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

class KnowledgeGraphVisualizer:
    """Enhanced visualizer for the Neo4j-backed knowledge graph"""

    def __init__(self, kg_manager):
        self.kg_manager = kg_manager
        try:
            self.nlp = spacy.load(os.getenv("SPACY_MODEL", "en_core_web_sm"))
            self.nlp.max_length = 2_000_000
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found for KnowledgeGraphVisualizer. Visualization might be limited. Please install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self.driver = None
        if self.kg_manager and self.kg_manager.driver:
            self.driver = self.kg_manager.driver
        else:
            try:
                self.driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USERNAME, NEO4J_PASSWORD))
                self.driver.verify_connectivity()
                logger.info("KnowledgeGraphVisualizer connected to Neo4j database directly.")
            except Exception as e:
                logger.error(f"KnowledgeGraphVisualizer failed to connect to Neo4j directly: {e}")
                self.driver = None

    def _fetch_graph_data_from_neo4j(self):
        """Fetches nodes and relationships from Neo4j to build a NetworkX graph for visualization."""
        if not self.driver:
            logger.warning("No Neo4j driver available for visualization data fetch.")
            return nx.MultiDiGraph()

        temp_graph = nx.MultiDiGraph()
        try:
            with self.driver.session() as session:
                # Fetch all nodes with their properties
                nodes_result = session.run("""
                    MATCH (n) 
                    RETURN 
                        elementId(n) AS id, 
                        LABELS(n) AS labels, 
                        properties(n) AS properties
                """)
                for record in nodes_result:
                    node_id = str(record["id"])
                    labels = record["labels"]
                    props = record["properties"]
                    
                    primary_type = 'UNKNOWN'
                    if 'Chunk' in labels: primary_type = 'CHUNK'
                    elif 'Document' in labels: primary_type = 'Document'
                    elif 'Author' in labels: primary_type = 'Author'
                    elif 'Entity' in labels: primary_type = props.get('type', 'Entity')
                    
                    node_attrs = {**props} # Start with all properties from Neo4j
                    node_attrs['id'] = node_id # Ensure 'id' is explicitly part of attrs
                    node_attrs['type'] = primary_type # Ensure primary_type is set

                    node_attrs['name'] = props.get('name', node_id) # Use name property or node_id if no name

                    # Handle 'content' property to avoid duplication if it exists in props
                    # Check if 'content' already exists and is not empty. If it is, use it.
                    # Otherwise, use 'title' or an empty string.
                    if 'content' not in node_attrs or node_attrs.get('content') is None or node_attrs.get('content') == '':
                        node_attrs['content'] = props.get('title', '') # Use title as fallback for content

                    temp_graph.add_node(node_id, **node_attrs)
                
                # Fetch all relationships with their properties
                rels_result = session.run("""
                    MATCH (s)-[r]->(t) 
                    RETURN 
                        elementId(s) AS source_id, 
                        elementId(t) AS target_id, 
                        type(r) AS type, 
                        properties(r) AS properties
                """)
                for record in rels_result:
                    source_id = str(record["source_id"])
                    target_id = str(record["target_id"])
                    rel_type = record["type"]
                    rel_props = record["properties"]
                    
                    if source_id in temp_graph and target_id in temp_graph:
                        temp_graph.add_edge(source_id, target_id, 
                                            key=str(uuid.uuid4()), # Unique key for MultiDiGraph
                                            relation_type=rel_type,
                                            **rel_props
                                           )
                    else:
                        logger.warning(f"Skipping edge {source_id}-{rel_type}->{target_id}: one or both nodes not found in fetched graph data.")

            logger.info(f"Fetched {len(temp_graph.nodes())} nodes and {len(temp_graph.edges())} edges from Neo4j for visualization.")
        except Exception as e:
            logger.error(f"Error fetching graph data from Neo4j for visualization: {e}")
            return nx.MultiDiGraph()
        
        return temp_graph

    def visualize_graph_static(self, figsize=(15, 10), save_path=None):
        """Create a static visualization using matplotlib"""
        graph_to_visualize = self._fetch_graph_data_from_neo4j()

        if len(graph_to_visualize.nodes()) == 0:
            logger.warning("No nodes in the graph to visualize for static plot.")
            return

        plt.figure(figsize=figsize)

        pos = nx.spring_layout(graph_to_visualize, k=0.5, iterations=50, seed=42)

        node_types = defaultdict(list)
        for n in graph_to_visualize.nodes():
            node_type = graph_to_visualize.nodes[n].get('type', 'UNKNOWN')
            node_types[node_type].append(n)

        node_color_map = {
            'CHUNK': 'lightcoral', 'PERSON': 'blue', 'ORG': 'green', 'GPE': 'purple',
            'STUDY': 'orange', 'WATER_BODY': 'cyan', 'HYDRO_MEASUREMENT': 'yellow',
            'POLLUTANT': 'red', 'HYDRO_EVENT': 'magenta', 'HYDRO_INFRASTRUCTURE': 'brown',
            'HYDRO_MODEL': 'lime', 'MISC': 'gray', 'DATE': 'darkblue', 'MONEY': 'darkgreen',
            'DATASET': 'teal', 'Document': 'lightgreen', 'Author': 'pink', 'UNKNOWN': 'lightgray',
            'Entity': 'lightblue'
        }

        for node_type, nodes_list in node_types.items():
            color = node_color_map.get(node_type, node_color_map['UNKNOWN'])
            node_size = 300 if node_type == 'CHUNK' else 500 if node_type in ['Document', 'Author'] else 400
            nx.draw_networkx_nodes(graph_to_visualize, pos, nodelist=nodes_list, node_color=color, node_size=node_size, alpha=0.8, edgecolors='black')

        edge_labels = nx.get_edge_attributes(graph_to_visualize, 'relation_type')
        nx.draw_networkx_edges(graph_to_visualize, pos, alpha=0.3, width=1, arrowsize=10)
        nx.draw_networkx_edge_labels(graph_to_visualize, pos, edge_labels=edge_labels, font_size=7, alpha=0.7)

        node_labels = {}
        for node_id in graph_to_visualize.nodes():
            node_info = graph_to_visualize.nodes[node_id]
            node_type = node_info.get('type')
            if node_type == 'CHUNK':
                node_labels[node_id] = f"Chunk:{str(node_id)[:4]}..."
            elif node_type == 'Document':
                node_labels[node_id] = node_info.get('title', str(node_id)[:8] + '...')
            elif node_type == 'Author':
                node_labels[node_id] = node_info.get('name', str(node_id)[:8] + '...')
            else:
                node_labels[node_id] = node_info.get('name', str(node_id)[:8] + '...')
        
        nx.draw_networkx_labels(graph_to_visualize, pos, labels=node_labels, font_size=7, font_color='black')

        legend_handles = []
        for node_type, color in node_color_map.items():
            if node_type != 'UNKNOWN':
                legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=node_type,
                                                 markerfacecolor=color, markersize=10))
        plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.title("Knowledge Graph Visualization", size=16)
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

        graph_to_visualize = self._fetch_graph_data_from_neo4j()

        if len(graph_to_visualize.nodes()) == 0:
            logger.warning("No nodes in the graph to visualize for interactive plot.")
            return None

        pos = nx.spring_layout(graph_to_visualize, k=0.5, iterations=50, seed=42)

        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        node_type_map = {
            'CHUNK': 'lightcoral', 'PERSON': 'blue', 'ORG': 'green', 'GPE': 'purple',
            'STUDY': 'orange', 'WATER_BODY': 'cyan', 'HYDRO_MEASUREMENT': 'yellow',
            'POLLUTANT': 'red', 'HYDRO_EVENT': 'magenta', 'HYDRO_INFRASTRUCTURE': 'brown',
            'HYDRO_MODEL': 'lime', 'MISC': 'gray', 'DATE': 'darkblue', 'MONEY': 'darkgreen',
            'DATASET': 'teal', 'Document': 'lightgreen', 'Author': 'pink', 'Entity': 'lightblue',
            'DEFAULT': 'lightgray'
        }

        for node_id in graph_to_visualize.nodes():
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)

            node_info = graph_to_visualize.nodes[node_id]
            node_type = node_info.get('type', 'DEFAULT')
            
            display_name = ""
            hover_content = f"<b>ID:</b> {node_id}<br><b>Type:</b> {node_type}<br>"

            if node_type == 'CHUNK':
                display_name = f"Chunk:{str(node_id)[:4]}..."
                hover_content += (
                    f"<b>Document:</b> {node_info.get('document_id', 'N/A')}<br>"
                    f"<b>Page:</b> {node_info.get('page_number', 'N/A')}<br>"
                    f"<b>Section:</b> {node_info.get('section_title', 'N/A')}<br>"
                    f"<b>Content:</b> {node_info.get('content', 'N/A')}"
                )
            elif node_type == 'Document':
                display_name = node_info.get('title', f"Doc:{str(node_id)[:4]}...")
                hover_content += (
                    f"<b>Title:</b> {node_info.get('title', 'N/A')}<br>"
                    f"<b>Filepath:</b> {node_info.get('filepath', 'N/A')}<br>"
                    f"<b>Authors:</b> {', '.join(node_info.get('authors', []))}"
                )
            elif node_type == 'Author':
                display_name = node_info.get('name', f"Author:{str(node_id)[:4]}...")
                hover_content += f"<b>Name:</b> {node_info.get('name', 'N/A')}<br>"
            else:
                display_name = node_info.get('name', f"{node_type}:{str(node_id)[:4]}...")
                hover_content += (
                    f"<b>Name:</b> {node_info.get('name', 'N/A')}<br>"
                    f"<b>Entity Type:</b> {node_info.get('entity_type', 'N/A')}<br>"
                    f"<b>Confidence:</b> {node_info.get('confidence', 'N/A')}"
                )

            incoming_edges = graph_to_visualize.in_edges(node_id, data=True)
            outgoing_edges = graph_to_visualize.out_edges(node_id, data=True)

            if incoming_edges or outgoing_edges:
                hover_content += "<br><b>Relations:</b>"
                for u, v, data in incoming_edges:
                    source_node_display = graph_to_visualize.nodes[u].get('name', str(u)[:8] + '...') if graph_to_visualize.nodes[u].get('type') not in ['CHUNK', 'Document', 'Author'] else f"{graph_to_visualize.nodes[u].get('type')}:{str(u)[:4]}..."
                    hover_content += f"<br> &larr; {data.get('relation_type', 'related to')} {source_node_display}"
                for u, v, data in outgoing_edges:
                    target_node_display = graph_to_visualize.nodes[v].get('name', str(v)[:8] + '...') if graph_to_visualize.nodes[v].get('type') not in ['CHUNK', 'Document', 'Author'] else f"{graph_to_visualize.nodes[v].get('type')}:{str(v)[:4]}..."
                    hover_content += f"<br> &rarr; {data.get('relation_type', 'related to')} {target_node_display}"

            node_text.append(hover_content)

            connections = graph_to_visualize.degree(node_id)
            node_size.append(10 + connections * 3)
            node_color.append(node_type_map.get(node_type, node_type_map['DEFAULT']))

        node_trace = go.Scatter(
            x=node_x, y=node_y, text=node_text, mode='markers',
            hoverinfo='text',
            marker=dict(size=node_size, color=node_color, line_width=1, line_color='DarkSlateGrey')
        )

        edge_x = []
        edge_y = []
        edge_annotations = []

        for edge in graph_to_visualize.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2

            text_x_offset = (x1 - x0) * 0.1
            text_y_offset = (y1 - y0) * 0.1

            edge_length = np.sqrt((x1-x0)**2 + (y1-y0)**2)
            if edge_length > 0:
                angle_rad = np.arctan2(y1 - y0, x1 - x0)
                angle_deg = np.degrees(angle_rad)
                if angle_deg > 90 or angle_deg < -90:
                    angle_deg += 180

                edge_annotations.append(
                    go.layout.Annotation(
                        x=(mid_x + text_x_offset),
                        y=(mid_y + text_y_offset),
                        xref='x', yref='y',
                        ax=x0, ay=y0,
                        axref='x', ayref='y',
                        showarrow=True,
                        arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor='gray',
                        opacity=0.7,
                        text=edge[2].get('relation_type', 'rel'),
                        font=dict(size=8, color='darkslategray'),
                        textangle=angle_deg,
                        xanchor="center", yanchor="bottom",
                    )
                )

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(width=0.8, color='gray'),
            hoverinfo='none',
            opacity=0.7
        )

        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
                            title=dict(text="Interactive Knowledge Graph", font=dict(size=16)),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=edge_annotations + [
                                dict(
                                    text="Node size = number of connections",
                                    showarrow=False,
                                    xref="paper", yref="paper",
                                    x=0.005, y=-0.002,
                                    xanchor="left", yanchor="bottom",
                                    font=dict(color="black", size=10)
                                ),
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

        try:
            output_html_file = os.path.join(os.getenv("STORAGE_PATH", "./storage"), "knowledge_graph_interactive.html")
            fig.write_html(output_html_file)
        except Exception as e:
            logger.error(f"Error saving interactive graph to HTML: {e}")
            print(f"Error saving interactive graph to HTML: {e}")
            return None

        return fig

class KnowledgeGapAnalyzer:
    """Comprehensive knowledge gap analysis"""

    def __init__(self, kg_manager):
        self.kg_manager = kg_manager

    def _fetch_analysis_data_from_neo4j(self):
        """Fetches necessary data from Neo4j for gap analysis."""
        if not self.kg_manager.driver:
            logger.warning("No Neo4j driver available for analysis data fetch.")
            return None

        data = {}
        try:
            with self.kg_manager.driver.session() as session:
                # Total nodes and edges
                count_result = session.run("MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN count(DISTINCT n) AS total_nodes, count(DISTINCT r) AS total_edges")
                counts = count_result.single()
                data['total_nodes'] = counts['total_nodes']
                data['total_edges'] = counts['total_edges']

                # Isolated nodes (no relationships at all)
                isolated_nodes_result = session.run("MATCH (n) WHERE NOT (n)--() RETURN n.name AS name, LABELS(n)[0] AS type, elementId(n) AS id LIMIT 10")
                data['isolated_nodes_list'] = [r.data() for r in isolated_nodes_result]
                data['isolated_nodes_count'] = session.run("MATCH (n) WHERE NOT (n)--() RETURN count(n) AS count").single()['count']

                temp_graph = nx.MultiDiGraph()
                nodes_result = session.run("MATCH (n) RETURN elementId(n) AS id, LABELS(n) AS labels, properties(n) AS properties")
                for r in nodes_result:
                    node_id = str(r["id"])
                    labels = r["labels"]
                    props = r["properties"]

                    primary_type = 'UNKNOWN'
                    if 'Chunk' in labels: primary_type = 'CHUNK'
                    elif 'Document' in labels: primary_type = 'Document'
                    elif 'Author' in labels: primary_type = 'Author'
                    elif 'Entity' in labels: primary_type = props.get('type', 'Entity')
                    
                    node_attrs = {**props}
                    node_attrs['id'] = node_id
                    node_attrs['type'] = primary_type
                    node_attrs['name'] = props.get('name', node_id)
                    if 'content' not in node_attrs or node_attrs.get('content') is None or node_attrs.get('content') == '':
                        node_attrs['content'] = props.get('title', '')
                    temp_graph.add_node(node_id, **node_attrs)

                rels_result = session.run("MATCH (s)-[r]->(t) RETURN elementId(s) AS s_id, elementId(t) AS t_id, type(r) AS type")
                for r in rels_result: 
                    temp_graph.add_edge(str(r["s_id"]), str(r["t_id"]), relation_type=r["type"])
                
                components = list(nx.weakly_connected_components(temp_graph))
                data['connected_components'] = len(components)
                data['largest_component_size'] = max(len(c) for c in components) if components else 0
                data['graph_density'] = nx.density(temp_graph) if temp_graph.number_of_nodes() > 1 else 0
                data['connectivity_ratio'] = data['largest_component_size'] / data['total_nodes'] if data['total_nodes'] > 0 else 0

                # Node degrees
                degrees_result = session.run("MATCH (n) RETURN COUNT { (n)--() } AS degree")
                degrees = [r["degree"] for r in degrees_result]
                if degrees:
                    data['degree_distribution'] = {
                        'min': min(degrees), 'max': max(degrees), 'mean': np.mean(degrees),
                        'median': np.median(degrees), 'std': np.std(degrees)
                    }
                    data['average_degree'] = np.mean(degrees)
                else:
                    data['degree_distribution'] = {}
                    data['average_degree'] = 0

                # Entity type distribution
                entity_type_dist_result = session.run("MATCH (e:Entity) RETURN e.type AS type, count(e) AS count")
                data['entity_type_distribution'] = {r["type"]: r["count"] for r in entity_type_dist_result}

                # Sparse entities (entities with few connections)
                sparse_entities_result = session.run("""
                    MATCH (e:Entity)
                    WHERE COUNT { (e)--() } < 2
                    RETURN elementId(e) AS entity_id, e.name AS name, e.type AS type, COUNT { (e)--() } AS connections
                    LIMIT 20
                """)
                data['sparse_entities'] = [r.data() for r in sparse_entities_result]

                # Cross-document entities
                cross_doc_entities_result = session.run("""
                    MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document)
                    WITH e, COLLECT(DISTINCT elementId(d)) AS doc_ids
                    WHERE size(doc_ids) > 1
                    RETURN elementId(e) AS entity_id, e.name AS name, size(doc_ids) AS doc_count
                    LIMIT 10
                """)
                data['cross_document_entity_examples'] = [r.data() for r in cross_doc_entities_result]
                data['cross_document_entities_count'] = session.run("""
                    MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document)
                    WITH e, COLLECT(DISTINCT elementId(d)) AS doc_ids
                    WHERE size(doc_ids) > 1
                    RETURN count(DISTINCT e) AS count
                """).single()['count']

                # Sparse documents (documents with few entities)
                sparse_docs_result = session.run("""
                    MATCH (d:Document)
                    OPTIONAL MATCH (d)<-[:MENTIONED_IN]-(e:Entity)
                    WITH d, COLLECT(DISTINCT elementId(e)) AS entity_ids
                    WHERE size(entity_ids) < 3
                    RETURN elementId(d) AS doc_id, d.title AS title, size(entity_ids) AS entity_count
                    LIMIT 10
                """)
                data['sparse_document_examples'] = [r.data() for r in sparse_docs_result]
                data['sparse_documents_count'] = session.run("""
                    MATCH (d:Document)
                    OPTIONAL MATCH (d)<-[:MENTIONED_IN]-(e:Entity)
                    WITH d, COLLECT(DISTINCT elementId(e)) AS entity_ids
                    WHERE size(entity_ids) < 3
                    RETURN count(DISTINCT d) AS count
                """).single()['count']
                
                # Average entities per document
                avg_entities_per_doc_result = session.run("""
                    MATCH (d:Document)
                    OPTIONAL MATCH (d)<-[:MENTIONED_IN]-(e:Entity)
                    WITH d, COLLECT(DISTINCT elementId(e)) AS entity_ids
                    RETURN avg(size(entity_ids)) AS avg_count
                """).single()
                data['average_entities_per_document'] = avg_entities_per_doc_result['avg_count'] if avg_entities_per_doc_result else 0

        except Exception as e:
            logger.error(f"Error fetching analysis data from Neo4j: {e}")
            return {'error': f"Failed to fetch analysis data: {e}"}
        
        return data

    def analyze_gaps(self) -> Dict[str, Any]:
        """Comprehensive gap analysis"""
        analysis_data = self._fetch_analysis_data_from_neo4j()
        if 'error' in analysis_data:
            return analysis_data

        analysis = {
            'structural_gaps': {
                'total_nodes': analysis_data.get('total_nodes', 0),
                'total_edges': analysis_data.get('total_edges', 0),
                'isolated_nodes': analysis_data.get('isolated_nodes_count', 0),
                'isolated_node_list': [f"{n['name']} ({n['type']})" for n in analysis_data.get('isolated_nodes_list', [])],
                'connected_components': analysis_data.get('connected_components', 0),
                'largest_component_size': analysis_data.get('largest_component_size', 0),
                'graph_density': analysis_data.get('graph_density', 0),
                'connectivity_ratio': analysis_data.get('connectivity_ratio', 0)
            },
            'content_gaps': {
                'total_entities': sum(analysis_data.get('entity_type_distribution', {}).values()),
                'total_documents': len(self.kg_manager.all_document_metadata),
                'entity_type_distribution': analysis_data.get('entity_type_distribution', {}),
                'sparse_entities': analysis_data.get('sparse_entities', []),
                'entities_per_document': analysis_data.get('average_entities_per_document', 0)
            },
            'connectivity_gaps': {
                'average_degree': analysis_data.get('average_degree', 0),
                'degree_distribution': analysis_data.get('degree_distribution', {}),
                'bridges': 0,
                'articulation_points': 0,
                'critical_nodes_examples': []
            },
            'entity_coverage': {
                'cross_document_entities_count': analysis_data.get('cross_document_entities_count', 0),
                'cross_document_entity_examples': [e['name'] for e in analysis_data.get('cross_document_entity_examples', [])],
                'sparse_documents_count': analysis_data.get('sparse_documents_count', 0),
                'sparse_document_examples': [d['doc_id'] for d in analysis_data.get('sparse_document_examples', [])],
                'average_entities_per_document': analysis_data.get('average_entities_per_document', 0)
            },
            'recommendations': []
        }

        if analysis_data.get('total_nodes', 0) > 0:
            try:
                temp_graph_for_connectivity = self._fetch_graph_data_for_connectivity_analysis()
                if temp_graph_for_connectivity.number_of_nodes() > 0:
                    undirected_graph = temp_graph_for_connectivity.to_undirected()
                    if nx.is_connected(undirected_graph):
                        analysis['connectivity_gaps']['bridges'] = len(list(nx.bridges(undirected_graph)))
                        analysis['connectivity_gaps']['articulation_points'] = len(list(nx.articulation_points(undirected_graph)))
                        analysis['connectivity_gaps']['critical_nodes_examples'] = [
                            temp_graph_for_connectivity.nodes[n].get('name', str(n)[:20] + '...') 
                            for n in list(nx.articulation_points(undirected_graph))[:10]
                        ]
                    else:
                        logger.info("Graph is not connected for full bridge/articulation analysis. Calculating within components.")
                        total_bridges = 0
                        total_articulation_points = set()
                        for component_nodes in nx.connected_components(undirected_graph):
                            subgraph = undirected_graph.subgraph(component_nodes)
                            total_bridges += len(list(nx.bridges(subgraph)))
                            total_articulation_points.update(list(nx.articulation_points(subgraph)))
                        analysis['connectivity_gaps']['bridges'] = total_bridges
                        analysis['connectivity_gaps']['articulation_points'] = len(total_articulation_points)
                        analysis['connectivity_gaps']['critical_nodes_examples'] = [
                            temp_graph_for_connectivity.nodes[n].get('name', str(n)[:20] + '...') 
                            for n in list(total_articulation_points)[:10]
                        ]

            except Exception as e:
                logger.error(f"Error during NetworkX-based connectivity analysis: {e}")

        analysis['recommendations'] = self._generate_recommendations(analysis)

        return analysis

    def _fetch_graph_data_for_connectivity_analysis(self):
        """Fetches nodes and relationships from Neo4j to build a NetworkX graph for connectivity analysis."""
        if not self.kg_manager.driver:
            logger.warning("No Neo4j driver available for connectivity graph data fetch.")
            return nx.MultiDiGraph()

        temp_graph = nx.MultiDiGraph()
        try:
            with self.kg_manager.driver.session() as session:
                nodes_result = session.run("MATCH (n) RETURN elementId(n) AS id, LABELS(n) AS labels, properties(n) AS properties")
                for r in nodes_result: 
                    node_id = str(r["id"])
                    labels = r["labels"]
                    props = r["properties"]

                    primary_type = 'UNKNOWN'
                    if 'Chunk' in labels: primary_type = 'CHUNK'
                    elif 'Document' in labels: primary_type = 'Document'
                    elif 'Author' in labels: primary_type = 'Author'
                    elif 'Entity' in labels: primary_type = props.get('type', 'Entity')
                    
                    node_attrs = {**props}
                    node_attrs['id'] = node_id
                    node_attrs['type'] = primary_type
                    node_attrs['name'] = props.get('name', node_id)
                    if 'content' not in node_attrs or node_attrs.get('content') is None or node_attrs.get('content') == '':
                        node_attrs['content'] = props.get('title', '')
                    temp_graph.add_node(node_id, **node_attrs)
                
                rels_result = session.run("MATCH (s)-[r]->(t) RETURN elementId(s) AS s_id, elementId(t) AS t_id, type(r) AS type")
                for r in rels_result: 
                    temp_graph.add_edge(str(r["s_id"]), str(r["t_id"]), relation_type=r["type"])
            
            logger.debug(f"Fetched {len(temp_graph.nodes())} nodes and {len(temp_graph.edges())} edges for connectivity analysis.")
        except Exception as e:
            logger.error(f"Error fetching graph data for connectivity analysis from Neo4j: {e}")
            return nx.MultiDiGraph()
        
        return temp_graph

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        structural = analysis.get('structural_gaps', {})
        content = analysis.get('content_gaps', {})
        connectivity = analysis.get('connectivity_gaps', {})
        coverage = analysis.get('entity_coverage', {})

        if structural.get('isolated_nodes', 0) > 0:
            recommendations.append(f"Address {structural['isolated_nodes']} isolated nodes ({', '.join(structural['isolated_node_list'][:3])}...) by adding more contextual information in new documents or by enriching existing ones. These nodes are not connected to the main graph.")

        if structural.get('connectivity_ratio', 0) < 0.7:
            recommendations.append(f"Improve overall graph connectivity (ratio: {structural.get('connectivity_ratio', 0):.2%}) by ensuring entities and chunks have more cross-references. This will help retrieve more related information.")

        if content.get('sparse_entities', []):
            recommendations.append(f"Expand context for {len(content['sparse_entities'])} sparsely connected entities (e.g., {content['sparse_entities'][0].get('name', content['sparse_entities'][0].get('id'))} with {content['sparse_entities'][0]['connections']} connections). Focus on adding documents or relations that link these entities to others.")

        if content.get('entities_per_document', 0) < 3 and content.get('total_documents', 0) > 0:
            recommendations.append(f"The average number of entities per document is low ({content.get('entities_per_document', 0):.2f}). Consider refining entity extraction or adding richer documents.")

        if connectivity.get('articulation_points', 0) > 0:
            recommendations.append(f"Strengthen connections around {connectivity['articulation_points']} critical nodes ({', '.join(connectivity['critical_nodes_examples'][:3])}...). These nodes are crucial for graph connectivity, and their removal would partition the graph.")

        if connectivity.get('bridges', 0) > 0:
            recommendations.append(f"There are {connectivity['bridges']} bridge edges. Consider adding redundant paths to connect the components linked by these bridges for improved robustness.")

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
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report)
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
- **Sparse entities** (< 2 connections): {len(content.get('sparse_entities', []))} (Examples: {', '.join([e.get('name', e.get('id')) for e in content['sparse_entities'][:3]])})
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

    visualizer = KnowledgeGraphVisualizer(rag_system.kg_manager)
    analyzer = KnowledgeGapAnalyzer(rag_system.kg_manager)

    print("Creating interactive visualization (saved to knowledge_graph_interactive.html)...")
    fig = visualizer.visualize_graph_interactive()
    if fig:
        logger.info("Interactive visualization HTML file generated in your 'storage' directory.")

    print("\nGenerating comprehensive gap analysis report...")
    report_save_path = os.path.join(os.getenv("STORAGE_PATH", "./storage"), "knowledge_gap_report.md")
    report = analyzer.create_gap_report(report_save_path)
    print("Knowledge gap report saved to knowledge_gap_report.md")

    detailed_analysis = analyzer.analyze_gaps()

    return {
        'visualization_figure': fig,
        'gap_analysis_data': detailed_analysis,
        'report_content': report
    }

if __name__ == "__main__":
    logger.info("Run `docker-compose up` to trigger RAG system initialization and graph analysis.")