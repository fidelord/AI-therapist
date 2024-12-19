import os
from typing import Annotated, TypedDict, List, Union, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env

# Now you can access the key
openai_api_key = os.getenv("OPENAI_API_KEY")
def messages_reducer(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """Reducer for combining messages"""
    return left + right

def user_info_reducer(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer for combining user info"""
    return {**left, **right}

def summary_reducer(left: str, right: str) -> str:
    """Reducer for combining conversation summaries"""
    return right  # Always take the most recent summary


# Define the state with explicit reducers
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], messages_reducer]
    user_info: Annotated[Dict[str, Any], user_info_reducer]
    conversation_summary: Annotated[str, summary_reducer]



def create_model():
    try:
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=openai_api_key
            
        )
    except Exception as e:
        print(f"Error creating model: {e}")
        raise

def process_user_input(state: ChatState) -> ChatState:
    # Extract user info from the last message if available
    if state['messages']:
        last_message = state['messages'][-1]
        if not state['user_info'] and hasattr(last_message, 'content'):
            try:
                user_info = {'name': last_message.content.split()[0]}
            except:
                user_info = {'name': 'User'}
            return {
                **state,
                'user_info': user_info,
                'messages': []  # Clear messages to prepare for response
            }
    return state

def generate_response(state: ChatState) -> ChatState:
    model = create_model()
    
    # Prepare context
    context = f"User Info: {state['user_info']}\n"
    context += f"Conversation Summary: {state['conversation_summary']}"
    
    # System prompt
    system_prompt = f"""You are a compassionate and the best AI therapist in the world. 
    Context: {context}
    Provide a supportive and insightful response. make sure to show empathy for your client"""
    
    try:
        # Generate response
        response = model.invoke(
            state['messages'] + [HumanMessage(content=system_prompt)]
        )
        
        # Validate response
        if response and hasattr(response, 'content') and response.content:
            # Return updated state with new message
            return {
                **state,
                'messages': [response]
            }
        else:
            print("Received empty response from OpenAI API.")
            error_response = AIMessage(content="I'm having trouble processing that. Could you rephrase?")
            return {
                **state,
                'messages': [error_response]
            }
    
    except Exception as e:
        print(f"Error in generate_response: {e}")
        error_response = AIMessage(content="I'm having trouble processing that. Could you rephrase?")
        return {
            **state,
            'messages': [error_response]
        }

def update_summary(state: ChatState) -> ChatState:
    model = create_model()
    
    if not state['messages']:
        return state
    
    try:
        # Summarize recent conversation
        summary_request = f"Provide a brief, empathetic summary of the last interaction: {state['messages'][-1].content}"
        
        summary_response = model.invoke([
            HumanMessage(content=summary_request)
        ])
        
        # Validate response
        if summary_response and hasattr(summary_response, 'content') and summary_response.content:
            # Return updated state with new summary
            return {
                **state,
                'conversation_summary': summary_response.content
            }
        else:
            print("Received empty summary response from OpenAI API.")
            return state
    
    except Exception as e:
        print(f"Error in update_summary: {e}")
        return state

def create_chat_graph():
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("process_input", process_user_input)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("update_summary", update_summary)
    
    # Define workflow
    workflow.set_entry_point("process_input")
    workflow.add_edge("process_input", "generate_response")
    workflow.add_edge("generate_response", "update_summary")
    workflow.add_edge("update_summary", END)
    
    # Compile the graph
    return workflow.compile(checkpointer=MemorySaver())
# Main chatbot runner
def run_chatbot():
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    # Create the chat graph
    chat_graph = create_chat_graph()
    
    # Conversation thread configuration
    initial_thread = {"configurable": {"thread_id": "unique_conversation_id"}}
    
    # Example conversation
    try:
        # First interaction
        response = chat_graph.invoke(
            {"messages": [HumanMessage(content="Hi, I'm feeling anxious lately. Can we talk?")]}, 
            initial_thread

        )

        messages = response.get('messages', [])
        print(response['messages'][-1].content)

        # Check if the list is not empty
        if messages:
            # Access the last message in the list
            last_message = messages[-1]
            
            # Ensure the last message has a 'content' attribute
            if hasattr(last_message, 'content'):
                print("First Response:", last_message.content)
            else:
                print("The last message does not have a 'content' attribute.")
        else:
            print("The 'messages' list is empty.")
        #print("First Response:", response.messages[-1].content)
        
        # Follow-up interaction
        response = chat_graph.invoke(
            {"messages": [HumanMessage(content="I'm struggling with work stress")]}, 
            initial_thread
        )
        print("Second Response:", response['messages'][-1].content)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

# Run the chatbot
if __name__ == "__main__":
    run_chatbot()