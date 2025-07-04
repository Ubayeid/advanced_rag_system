from flask import Flask, request, jsonify, render_template
import sys
import os
import logging
# Add the parent directory of web_server.py to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import SimpleAdvancedRAGSystem directly from main.py
from main import SimpleAdvancedRAGSystem, LLM_MODEL_NAME # Import LLM_MODEL_NAME to check for 'debug' behavior
import knowledge_graph # This is your knowledge_graph.py file

app = Flask(__name__, template_folder='templates')

# Initialize a logger for web_server.py
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
web_logger = logging.getLogger(__name__)

# Declare global_rag_system here, but initialize it within if __name__ == '__main__':
global_rag_system = None

# ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Ensure RAG system is initialized before querying
    global global_rag_system
    if global_rag_system is None:
        web_logger.error("RAG system not initialized. Cannot process query.")
        return jsonify({'error': 'RAG system is not ready. Please try again later.'}), 503

    try:
        # Call the query method on the initialized global_rag_system instance
        response_data = global_rag_system.query(user_message)
        if response_data.get('status') == 'success':
            response = response_data.get('answer', 'No answer found.')
        else:
            response = f"Error: {response_data.get('error', 'Unknown error')}"

    except Exception as e:
        web_logger.error(f"Error during chat query: {e}")
        response = f"Error: {str(e)}"
    return jsonify({'answer': response})


# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == '__main__':

    web_logger.info("Starting Flask web server...")

    if global_rag_system is None:
        web_logger.info("Initializing RAG system (main process/worker level)...")
        global_rag_system = SimpleAdvancedRAGSystem()
        global_rag_system.initialize_knowledge_base()
        web_logger.info("RAG system initialization complete.")
    else:
        web_logger.info("RAG system already initialized for this process.")

    app.run(host='0.0.0.0', port=8000, debug=False)