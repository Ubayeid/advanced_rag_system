from flask import Flask, request, jsonify, render_template
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import answer_query  # You may need to adjust this import based on your main.py
import knowledge_graph

app = Flask(__name__, template_folder='templates')

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
        response = f"Error: {str(e)}"
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

print("Generating interactive knowledge graph visualization and gap report...")
sys.stdout.flush()
try:
    analysis_results = knowledge_graph.analyze_rag_knowledge_graph(rag_system)
    print("Knowledge graph visualization generated. Check your default web browser for interactive plot.")
    print(f"Knowledge gap report saved to knowledge_gap_report.md and printed above.")
    sys.stdout.flush()
except Exception as e:
    logger.error(f"Error during knowledge graph visualization/analysis: {e}")
    print(f"An error occurred during graph visualization: {e}")
    sys.stdout.flush()