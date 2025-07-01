from flask import Flask, request, jsonify, render_template
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the SimpleAdvancedRAGSystem class directly
from main import SimpleAdvancedRAGSystem, answer_query
import knowledge_graph

app = Flask(__name__, template_folder='templates')

# Initialize a logger for web_server.py if not already done globally
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
web_logger = logging.getLogger(__name__)

# This will ensure the knowledge base is built and KG analysis runs once when the Flask app starts.
global_rag_system = None # Define a global variable for the RAG system in web_server.py

@app.before_first_request
def initialize_rag_system_on_startup():
    """  Initializes the RAG system and performs KG analysis when the Flask app receives its first request. """

    global global_rag_system
    if global_rag_system is None:
        web_logger.info("Initializing RAG system on first request...")
        global_rag_system = SimpleAdvancedRAGSystem()
        global_rag_system.initialize_knowledge_base()

        web_logger.info("Generating interactive knowledge graph visualization and gap report...")
        try:
            analysis_results = knowledge_graph.analyze_rag_knowledge_graph(global_rag_system)
            web_logger.info("Knowledge graph visualization generated. Check your default web browser for interactive plot.")
            web_logger.info(f"Knowledge gap report saved to knowledge_gap_report.md and printed above.")
        except Exception as e:
            web_logger.error(f"Error generating knowledge graph analysis: {e}")
        web_logger.info("RAG system initialization and KG analysis complete.")

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    try:
        response = answer_query(user_message)
    except Exception as e:
        web_logger.error(f"Error during chat query: {e}")
        response = f"Error: {str(e)}"
    return jsonify({'answer': response})

if __name__ == '__main__':
    web_logger.info("Starting Flask web server...")
    app.run(host='0.0.0.0', port=8000, debug=True)