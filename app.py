import os
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import uuid

# Import the LangGraph chatbot logic from previous implementation
from ai_therapist import create_chat_graph, ChatState
from langchain_core.messages import HumanMessage, AIMessage

app = Flask(__name__)
CORS(app)

# Set a secret key for sessions
app.secret_key = os.urandom(24)

# Initialize the chat graph
chat_graph = create_chat_graph()

@app.route('/')
def index():
    """Render the main chat page"""
    # Ensure a unique conversation ID for each session
    if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    # Get user message from request
    user_message = request.json.get('message', '')
    
    # Retrieve or create conversation thread
    conversation_id = session.get('conversation_id', str(uuid.uuid4()))
    thread_config = {"configurable": {"thread_id": conversation_id}}
    
    try:
        # Prepare the input for the chat graph
        initial_input = {
            "messages": [HumanMessage(content=user_message)]
        }
        
        # Invoke the chat graph
        response = chat_graph.invoke(initial_input, thread_config)
        
        # Extract the AI's response
        ai_response = response['messages'][-1].content if response['messages'] else "I'm not sure how to respond."
        
        return jsonify({
            'status': 'success', 
            'message': ai_response
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)