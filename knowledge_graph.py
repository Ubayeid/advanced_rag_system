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
import uuid
import re

# Neo4j specific import
from neo4j import GraphDatabase, basic_auth, exceptions as neo4j_exceptions

logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Research domain keywords for hypothesis generation
RESEARCH_DOMAINS = {
    'hydrology': ['water', 'flow', 'river', 'basin', 'precipitation', 'evaporation', 'groundwater', 'surface water', 'flood', 'drought'],
    'climate': ['temperature', 'climate', 'weather', 'atmospheric', 'greenhouse', 'emission', 'carbon', 'warming'],
    'environmental': ['ecosystem', 'biodiversity', 'pollution', 'conservation', 'sustainability', 'environmental'],
    'geospatial': ['spatial', 'geographic', 'location', 'mapping', 'gis', 'remote sensing', 'satellite'],
    'modeling': ['model', 'simulation', 'prediction', 'forecast', 'algorithm', 'computational'],
    'data_science': ['machine learning', 'ai', 'data analysis', 'statistics', 'big data', 'analytics']
}

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

class ResearchHypothesisGenerator:
    """Generates research hypotheses and directions based on knowledge gaps"""
    
    def __init__(self, kg_manager):
        self.kg_manager = kg_manager
        
    def _identify_research_domains(self, entities: List[Dict]) -> Dict[str, float]:
        """Identify which research domains are most relevant based on entity content"""
        domain_scores = defaultdict(float)
        
        for entity in entities:
            entity_name = entity.get('name', '').lower()
            entity_type = entity.get('type', '').lower()
            
            for domain, keywords in RESEARCH_DOMAINS.items():
                for keyword in keywords:
                    if keyword in entity_name or keyword in entity_type:
                        domain_scores[domain] += 1.0
                        
        # Normalize scores
        total = sum(domain_scores.values())
        if total > 0:
            domain_scores = {k: v/total for k, v in domain_scores.items()}
            
        return dict(domain_scores)
    
    def _generate_hypotheses(self, gaps: Dict[str, Any], entities: List[Dict]) -> List[Dict[str, Any]]:
        """Generate specific research hypotheses based on identified gaps"""
        hypotheses = []
        
        # Identify research domains
        domains = self._identify_research_domains(entities)
        primary_domain = max(domains.items(), key=lambda x: x[1])[0] if domains else 'general'
        
        # Hypothesis 1: Isolated Entities Research
        isolated_nodes = gaps.get('structural_gaps', {}).get('isolated_nodes', 0)
        if isolated_nodes > 0:
            isolated_examples = gaps.get('structural_gaps', {}).get('isolated_node_list', [])
            hypotheses.append({
                'type': 'exploratory',
                'priority': 'high' if isolated_nodes > 5 else 'medium',
                'title': f'Investigate {primary_domain.title()} Context for Isolated Entities',
                'hypothesis': f'Isolated entities ({isolated_examples[:3]}) represent unexplored {primary_domain} concepts that could reveal new relationships when studied in broader context.',
                'research_question': f'What are the underlying {primary_domain} mechanisms connecting these isolated entities to the broader knowledge network?',
                'methodology': [
                    'Literature review of isolated entities',
                    'Cross-document analysis for missing connections',
                    'Expert interviews in relevant domain',
                    'Data collection to bridge knowledge gaps'
                ],
                'expected_outcomes': [
                    'New entity relationships discovered',
                    'Enhanced knowledge graph connectivity',
                    'Identification of research opportunities'
                ],
                'timeline': '3-6 months',
                'resources_needed': ['Domain experts', 'Additional literature', 'Data collection tools']
            })
        
        # Hypothesis 2: Connectivity Gaps
        connectivity_ratio = gaps.get('structural_gaps', {}).get('connectivity_ratio', 0)
        if connectivity_ratio < 0.7:
            hypotheses.append({
                'type': 'analytical',
                'priority': 'high',
                'title': f'Improve {primary_domain.title()} Knowledge Integration',
                'hypothesis': f'Low connectivity ratio ({connectivity_ratio:.1%}) indicates fragmented {primary_domain} knowledge that requires systematic integration.',
                'research_question': f'How can we systematically connect fragmented {primary_domain} knowledge to create a more cohesive understanding?',
                'methodology': [
                    'Network analysis of knowledge gaps',
                    'Systematic literature review',
                    'Meta-analysis of existing studies',
                    'Integration framework development'
                ],
                'expected_outcomes': [
                    'Improved knowledge graph connectivity',
                    'Integration framework for domain knowledge',
                    'Identification of key bridging concepts'
                ],
                'timeline': '6-12 months',
                'resources_needed': ['Network analysis tools', 'Literature databases', 'Collaboration platforms']
            })
        
        # Hypothesis 3: Entity Coverage Gaps
        sparse_entities = gaps.get('content_gaps', {}).get('sparse_entities', [])
        if sparse_entities:
            hypotheses.append({
                'type': 'descriptive',
                'priority': 'medium',
                'title': f'Expand {primary_domain.title()} Entity Coverage',
                'hypothesis': f'Sparsely connected entities represent understudied {primary_domain} concepts that could benefit from expanded research coverage.',
                'research_question': f'What additional information and relationships are needed to fully understand these sparsely connected {primary_domain} entities?',
                'methodology': [
                    'Entity relationship mapping',
                    'Gap analysis in existing literature',
                    'Expert consultation',
                    'Targeted data collection'
                ],
                'expected_outcomes': [
                    'Enhanced entity relationships',
                    'Improved knowledge coverage',
                    'New research directions identified'
                ],
                'timeline': '4-8 months',
                'resources_needed': ['Domain experts', 'Literature databases', 'Data collection tools']
            })
        
        # Hypothesis 4: Cross-Document Integration
        cross_doc_entities = gaps.get('entity_coverage', {}).get('cross_document_entities_count', 0)
        if cross_doc_entities < 10:
            hypotheses.append({
                'type': 'integrative',
                'priority': 'high',
                'title': f'Develop {primary_domain.title()} Cross-Document Integration Framework',
                'hypothesis': f'Limited cross-document entity sharing ({cross_doc_entities} entities) indicates need for better integration across {primary_domain} literature.',
                'research_question': f'How can we systematically integrate {primary_domain} knowledge across multiple documents and sources?',
                'methodology': [
                    'Cross-document entity analysis',
                    'Integration framework development',
                    'Standardization of entity representation',
                    'Collaborative knowledge building'
                ],
                'expected_outcomes': [
                    'Improved cross-document integration',
                    'Standardized entity representation',
                    'Enhanced knowledge discovery'
                ],
                'timeline': '8-12 months',
                'resources_needed': ['Integration platforms', 'Standardization tools', 'Collaboration networks']
            })
        
        return hypotheses
    
    def _generate_research_directions(self, gaps: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific research directions based on gaps"""
        directions = []
        
        # Direction 1: Data Collection Priorities
        isolated_nodes = gaps.get('structural_gaps', {}).get('isolated_nodes', 0)
        if isolated_nodes > 0:
            directions.append({
                'category': 'Data Collection',
                'priority': 'Immediate',
                'action': f'Collect additional data for {isolated_nodes} isolated entities',
                'specific_tasks': [
                    'Identify data sources for isolated entities',
                    'Design data collection protocols',
                    'Implement systematic data gathering',
                    'Validate and integrate new data'
                ],
                'success_metrics': [
                    'Reduction in isolated nodes by 50%',
                    'Improved entity connectivity',
                    'Enhanced knowledge coverage'
                ],
                'timeline': '2-4 months',
                'resources': ['Data collection tools', 'Domain experts', 'Validation frameworks']
            })
        
        # Direction 2: Literature Review Priorities
        connectivity_ratio = gaps.get('structural_gaps', {}).get('connectivity_ratio', 0)
        if connectivity_ratio < 0.8:
            directions.append({
                'category': 'Literature Review',
                'priority': 'High',
                'action': 'Conduct systematic literature review to identify missing connections',
                'specific_tasks': [
                    'Identify key literature gaps',
                    'Systematic review of related works',
                    'Cross-reference analysis',
                    'Integration of findings'
                ],
                'success_metrics': [
                    'Improved connectivity ratio to >80%',
                    'Identification of key bridging literature',
                    'Enhanced understanding of relationships'
                ],
                'timeline': '3-6 months',
                'resources': ['Literature databases', 'Review tools', 'Expert consultation']
            })
        
        # Direction 3: Methodological Development
        sparse_entities = gaps.get('content_gaps', {}).get('sparse_entities', [])
        if len(sparse_entities) > 5:
            directions.append({
                'category': 'Methodological Development',
                'priority': 'Medium',
                'action': 'Develop methods for better entity relationship discovery',
                'specific_tasks': [
                    'Analyze current relationship discovery methods',
                    'Develop improved algorithms',
                    'Test and validate new approaches',
                    'Implement systematic relationship discovery'
                ],
                'success_metrics': [
                    'Improved entity relationship coverage',
                    'Reduced sparse entities by 30%',
                    'Enhanced knowledge discovery capabilities'
                ],
                'timeline': '6-9 months',
                'resources': ['Algorithm development tools', 'Testing frameworks', 'Validation datasets']
            })
        
        return directions

class EnhancedKnowledgeGapAnalyzer:
    """Enhanced knowledge gap analysis with research hypothesis generation"""

    def __init__(self, kg_manager):
        self.kg_manager = kg_manager
        self.hypothesis_generator = ResearchHypothesisGenerator(kg_manager)

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

                # Entity type distribution
                entity_types_result = session.run("MATCH (e:Entity) RETURN e.original_entity_type AS type, count(e) AS count")
                data['entity_type_distribution'] = {r['type']: r['count'] for r in entity_types_result}

                # FIXED: Use COUNT {} instead of size() for sparse entities
                sparse_entities_result = session.run("""
                    MATCH (e:Entity)
                    WITH e, COUNT { (e)--() } AS connections
                    WHERE connections < 3
                    RETURN e.name AS name, e.id AS id, connections
                    ORDER BY connections ASC
                    LIMIT 10
                """)
                data['sparse_entities'] = [r.data() for r in sparse_entities_result]

                # Cross-document entities
                cross_doc_result = session.run("""
                    MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document)
                    WITH e, count(DISTINCT d) AS doc_count
                    WHERE doc_count > 1
                    RETURN e.name AS name, doc_count
                    ORDER BY doc_count DESC
                    LIMIT 10
                """)
                data['cross_document_entity_examples'] = [r.data() for r in cross_doc_result]
                data['cross_document_entities_count'] = session.run("""
                    MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document)
                    WITH e, count(DISTINCT d) AS doc_count
                    WHERE doc_count > 1
                    RETURN count(e) AS count
                """).single()['count']

                # Sparse documents
                sparse_docs_result = session.run("""
                    MATCH (d:Document)
                    OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(d)
                    WITH d, count(e) AS entity_count
                    WHERE entity_count < 3
                    RETURN d.id AS doc_id, entity_count
                    ORDER BY entity_count ASC
                    LIMIT 10
                """)
                data['sparse_document_examples'] = [r.data() for r in sparse_docs_result]
                data['sparse_documents_count'] = session.run("""
                    MATCH (d:Document)
                    OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(d)
                    WITH d, count(e) AS entity_count
                    WHERE entity_count < 3
                    RETURN count(d) AS count
                """).single()['count']

                # Average entities per document
                avg_entities_result = session.run("""
                    MATCH (d:Document)
                    OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(d)
                    WITH d, count(e) AS entity_count
                    RETURN avg(entity_count) AS avg_entities
                """)
                data['average_entities_per_document'] = avg_entities_result.single()['avg_entities'] or 0

                # FIXED: Use COUNT {} instead of size() for node degrees
                degrees_result = session.run("MATCH (n) RETURN COUNT { (n)--() } AS degree")
                degrees = [r["degree"] for r in degrees_result]
                if degrees:
                    data['degree_distribution'] = {
                        'min': min(degrees), 'max': max(degrees), 'mean': np.mean(degrees),
                        'median': np.median(degrees), 'std': np.std(degrees)
                    }
                    data['average_degree'] = np.mean(degrees)

                # Build NetworkX graph for connectivity analysis
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

        except Exception as e:
            logger.error(f"Error fetching analysis data from Neo4j: {e}")
            return {'error': str(e)}

        return data

    def analyze_gaps(self) -> Dict[str, Any]:
        """Comprehensive gap analysis with research hypotheses"""
        analysis_data = self._fetch_analysis_data_from_neo4j()
        if 'error' in analysis_data:
            return analysis_data

        # Initialize analysis dictionary first
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
            'research_hypotheses': [],
            'research_directions': [],
            'recommendations': []
        }

        # Get all entities for hypothesis generation
        entities = []
        if self.kg_manager.driver:
            try:
                with self.kg_manager.driver.session() as session:
                    entities_result = session.run("MATCH (e:Entity) RETURN e.name AS name, e.original_entity_type AS type, e.id AS id")
                    entities = [r.data() for r in entities_result]
            except Exception as e:
                logger.error(f"Error fetching entities for hypothesis generation: {e}")

        # Generate research hypotheses and directions
        try:
            analysis['research_hypotheses'] = self.hypothesis_generator._generate_hypotheses(analysis, entities)
            analysis['research_directions'] = self.hypothesis_generator._generate_research_directions(analysis)
        except Exception as e:
            logger.error(f"Error generating research hypotheses: {e}")
            analysis['research_hypotheses'] = []
            analysis['research_directions'] = []

        # Add connectivity analysis
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

        # Generate recommendations
        try:
            analysis['recommendations'] = self._generate_recommendations(analysis)
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            analysis['recommendations'] = ["Error generating recommendations"]

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
            recommendations.append(f"**PRIORITY 1**: Address {structural['isolated_nodes']} isolated nodes by conducting targeted research on these entities. These represent unexplored research opportunities.")

        if structural.get('connectivity_ratio', 0) < 0.7:
            recommendations.append(f"**PRIORITY 2**: Improve knowledge integration by focusing on cross-document relationships. Current connectivity ratio of {structural.get('connectivity_ratio', 0):.1%} indicates fragmented knowledge.")

        if content.get('sparse_entities', []):
            recommendations.append(f"**PRIORITY 3**: Expand research on {len(content['sparse_entities'])} sparsely connected entities. These represent understudied areas with high research potential.")

        if connectivity.get('articulation_points', 0) > 0:
            recommendations.append(f"**CRITICAL**: Focus on {connectivity['articulation_points']} critical nodes that are essential for knowledge connectivity. These are high-impact research targets.")

        if coverage.get('cross_document_entities_count', 0) < 5:
            recommendations.append(f"**INTEGRATION NEEDED**: Only {coverage.get('cross_document_entities_count', 0)} entities appear across multiple documents. Develop systematic integration approaches.")

        if not recommendations:
            recommendations.append("Knowledge graph structure is well-connected and comprehensive. Focus on deepening existing relationships and exploring emerging connections.")

        return recommendations

    def create_enhanced_gap_report(self, save_path=None) -> str:
        """Create an enhanced gap analysis report with research hypotheses"""
        analysis = self.analyze_gaps()

        report = f"""
# Enhanced Knowledge Gap Analysis & Research Directions Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
{self._create_enhanced_executive_summary(analysis)}

---

## Research Hypotheses & Opportunities
{self._format_research_hypotheses(analysis.get('research_hypotheses', []))}

---

## Specific Research Directions
{self._format_research_directions(analysis.get('research_directions', []))}

---

## Knowledge Gap Analysis
{self._format_enhanced_structural_analysis(analysis['structural_gaps'])}
{self._format_enhanced_content_analysis(analysis['content_gaps'])}
{self._format_enhanced_connectivity_analysis(analysis['connectivity_gaps'])}
{self._format_enhanced_coverage_analysis(analysis['entity_coverage'])}

---

## Actionable Recommendations
{self._format_enhanced_recommendations(analysis['recommendations'])}

---

## Implementation Roadmap
{self._create_implementation_roadmap(analysis)}

---

## Success Metrics & Evaluation
{self._create_success_metrics(analysis)}
"""

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report)
            except Exception as e:
                logger.error(f"Error saving enhanced knowledge gap report to {save_path}: {e}")

        return report

    def _create_enhanced_executive_summary(self, analysis):
        structural = analysis.get('structural_gaps', {})
        hypotheses = analysis.get('research_hypotheses', [])
        
        return f"""
## Knowledge Graph Overview
- **Total Knowledge Nodes**: {structural.get('total_nodes', 0)}
- **Total Relationships**: {structural.get('total_edges', 0)}
- **Knowledge Connectivity**: {structural.get('connectivity_ratio', 0):.1%}
- **Research Opportunities Identified**: {len(hypotheses)}

## Key Research Insights
This analysis reveals **{len(hypotheses)} specific research opportunities** that can advance knowledge in your domain. The knowledge graph shows {structural.get('isolated_nodes', 0)} isolated concepts that represent unexplored research areas, and a connectivity ratio of {structural.get('connectivity_ratio', 0):.1%} indicating {self._get_connectivity_description(structural.get('connectivity_ratio', 0))}.

## Immediate Action Items
1. **High-Priority Research**: {len([h for h in hypotheses if h.get('priority') == 'high'])} hypotheses identified
2. **Data Collection Needs**: {structural.get('isolated_nodes', 0)} entities require additional data
3. **Integration Opportunities**: {analysis.get('entity_coverage', {}).get('cross_document_entities_count', 0)} cross-document entities need systematic integration
"""

    def _get_connectivity_description(self, ratio):
        if ratio >= 0.9:
            return "excellent knowledge integration"
        elif ratio >= 0.7:
            return "good knowledge integration with room for improvement"
        elif ratio >= 0.5:
            return "moderate knowledge fragmentation requiring attention"
        else:
            return "significant knowledge fragmentation requiring immediate action"

    def _format_research_hypotheses(self, hypotheses):
        if not hypotheses:
            return "No specific research hypotheses identified at this time."
        
        formatted = []
        for i, hypothesis in enumerate(hypotheses, 1):
            formatted.append(f"""
### Hypothesis {i}: {hypothesis['title']}
**Priority**: {hypothesis['priority'].upper()}
**Type**: {hypothesis['type'].title()}

**Hypothesis Statement**: {hypothesis['hypothesis']}

**Research Question**: {hypothesis['research_question']}

**Proposed Methodology**:
{chr(10).join([f"- {method}" for method in hypothesis['methodology']])}

**Expected Outcomes**:
{chr(10).join([f"- {outcome}" for outcome in hypothesis['expected_outcomes']])}

**Timeline**: {hypothesis['timeline']}

**Resources Required**:
{chr(10).join([f"- {resource}" for resource in hypothesis['resources_needed']])}
""")
        
        return '\n'.join(formatted)

    def _format_research_directions(self, directions):
        if not directions:
            return "No specific research directions identified at this time."
        
        formatted = []
        for direction in directions:
            formatted.append(f"""
### {direction['category']}: {direction['action']}
**Priority**: {direction['priority']}

**Specific Tasks**:
{chr(10).join([f"- {task}" for task in direction['specific_tasks']])}

**Success Metrics**:
{chr(10).join([f"- {metric}" for metric in direction['success_metrics']])}

**Timeline**: {direction['timeline']}

**Required Resources**:
{chr(10).join([f"- {resource}" for resource in direction['resources']])}
""")
        
        return '\n'.join(formatted)

    def _format_enhanced_structural_analysis(self, structural):
        return f"""
## Structural Knowledge Gaps

### Network Overview
- **Total Knowledge Nodes**: {structural.get('total_nodes', 0)}
- **Total Relationships**: {structural.get('total_edges', 0)}
- **Knowledge Connectivity**: {structural.get('connectivity_ratio', 0):.1%}
- **Graph Density**: {structural.get('graph_density', 0):.4f}

### Critical Issues
- **Isolated Knowledge Nodes**: {structural.get('isolated_nodes', 0)} entities lack connections
  - Examples: {', '.join(structural.get('isolated_node_list', [])[:5])}
  - **Research Impact**: These represent unexplored research opportunities
- **Connected Components**: {structural.get('connected_components', 0)} separate knowledge clusters
  - **Largest Component**: {structural.get('largest_component_size', 0)} nodes
  - **Research Opportunity**: Integration needed between components
"""

    def _format_enhanced_content_analysis(self, content):
        return f"""
## Content Knowledge Gaps

### Entity Coverage
- **Total Entities**: {content.get('total_entities', 0)}
- **Entity Types**: {content.get('entity_type_distribution', {})}
- **Average Entities per Document**: {content.get('entities_per_document', 0):.2f}

### Research Opportunities
- **Sparsely Connected Entities**: {len(content.get('sparse_entities', []))} entities with limited relationships
  - **Research Value**: These represent understudied concepts
  - **Action Required**: Expand research coverage for these entities
"""

    def _format_enhanced_connectivity_analysis(self, connectivity):
        return f"""
## Knowledge Connectivity Analysis

### Network Structure
- **Average Node Degree**: {connectivity.get('average_degree', 0):.2f}
- **Critical Nodes**: {connectivity.get('articulation_points', 0)} nodes are essential for connectivity
- **Bridge Connections**: {connectivity.get('bridges', 0)} critical relationships

### Research Implications
- **Critical Nodes**: {', '.join(connectivity.get('critical_nodes_examples', [])[:5])}
  - **Research Priority**: These nodes are high-impact research targets
  - **Risk Assessment**: Loss of these nodes would fragment knowledge
"""

    def _format_enhanced_coverage_analysis(self, coverage):
        return f"""
## Cross-Document Knowledge Integration

### Integration Status
- **Cross-Document Entities**: {coverage.get('cross_document_entities_count', 0)} entities appear in multiple documents
- **Sparse Documents**: {coverage.get('sparse_documents_count', 0)} documents have limited entity coverage

### Research Needs
- **Integration Gaps**: Limited cross-document entity sharing indicates fragmentation
- **Systematic Integration**: Need for frameworks to connect knowledge across documents
"""

    def _format_enhanced_recommendations(self, recommendations):
        return '\n'.join([f"### {rec}" for rec in recommendations])

    def _create_implementation_roadmap(self, analysis):
        hypotheses = analysis.get('research_hypotheses', [])
        directions = analysis.get('research_directions', [])
        
        roadmap = """
## Implementation Roadmap

### Phase 1: Immediate Actions (0-3 months)
"""
        
        immediate_actions = []
        for direction in directions:
            if direction.get('priority') == 'Immediate':
                immediate_actions.append(f"- {direction['action']}")
        
        if immediate_actions:
            roadmap += '\n'.join(immediate_actions)
        else:
            roadmap += "- Conduct preliminary research on identified gaps\n- Establish research priorities\n- Begin data collection planning"
        
        roadmap += """

### Phase 2: Short-term Research (3-6 months)
"""
        
        short_term = []
        for hypothesis in hypotheses:
            if '3-6' in hypothesis.get('timeline', '') or '4-8' in hypothesis.get('timeline', ''):
                short_term.append(f"- {hypothesis['title']}")
        
        if short_term:
            roadmap += '\n'.join(short_term)
        else:
            roadmap += "- Implement systematic literature review\n- Begin hypothesis testing\n- Develop integration frameworks"
        
        roadmap += """

### Phase 3: Medium-term Development (6-12 months)
"""
        
        medium_term = []
        for hypothesis in hypotheses:
            if '6-12' in hypothesis.get('timeline', '') or '8-12' in hypothesis.get('timeline', ''):
                medium_term.append(f"- {hypothesis['title']}")
        
        if medium_term:
            roadmap += '\n'.join(medium_term)
        else:
            roadmap += "- Complete major research initiatives\n- Implement integration frameworks\n- Evaluate and refine approaches"
        
        roadmap += """

### Phase 4: Long-term Integration (12+ months)
- Establish sustainable knowledge integration processes
- Develop ongoing gap monitoring systems
- Create collaborative research networks
- Implement continuous improvement frameworks
"""
        
        return roadmap

    def _create_success_metrics(self, analysis):
        return """
## Success Metrics & Evaluation Framework

### Quantitative Metrics
- **Knowledge Connectivity**: Target >80% connectivity ratio
- **Isolated Node Reduction**: Reduce isolated nodes by 50%
- **Cross-Document Integration**: Increase cross-document entities by 100%
- **Entity Relationship Density**: Improve average node degree by 25%

### Qualitative Metrics
- **Research Impact**: Number of new research directions identified
- **Knowledge Integration**: Improved understanding of relationships
- **Collaboration Enhancement**: Increased cross-disciplinary connections
- **Innovation Potential**: New hypotheses and research opportunities

### Evaluation Timeline
- **Monthly**: Progress tracking on immediate actions
- **Quarterly**: Assessment of short-term research progress
- **Semi-annually**: Comprehensive gap analysis update
- **Annually**: Full knowledge graph reassessment

### Success Indicators
- Reduced knowledge fragmentation
- Increased research collaboration
- Enhanced understanding of domain relationships
- Improved knowledge discovery capabilities
- Sustainable knowledge integration processes
"""

# Update the main analysis function to use the enhanced analyzer
def analyze_rag_knowledge_graph(rag_system):
    """Analyze the knowledge graph from a RAG system and generate an enhanced gap report."""
    
    # Use the enhanced analyzer
    analyzer = EnhancedKnowledgeGapAnalyzer(rag_system.kg_manager)

    logger.info("Generating enhanced knowledge gap analysis with research hypotheses...")
    report_save_path = os.path.join(os.getenv("STORAGE_PATH", "./storage"), "enhanced_knowledge_gap_report.md")
    report = analyzer.create_enhanced_gap_report(report_save_path)
    logger.info("Enhanced knowledge gap report saved to enhanced_knowledge_gap_report.md")

    detailed_analysis = analyzer.analyze_gaps()

    return {
        'gap_analysis_data': detailed_analysis,
        'report_content': report
    }

if __name__ == "__main__":
    logger.info("Run `docker-compose up` to trigger RAG system initialization and enhanced graph analysis.")