"""
Streamlit Chatbot Application for Insurance Q&A
User-friendly interface with chat history and source citations
"""

import os
import sys
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.vector_store import VectorStore
from src.retriever import RAGRetriever
from src.llm_handler import LLMHandler, ChatbotResponse


# Page configuration
st.set_page_config(
    page_title="OMINIMO Insurance Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .user-message {
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        border-left: 4px solid #4caf50;
    }
    .source-box {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .confidence-high {
        color: #4caf50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    /* Hide Streamlit branding and deploy button */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_chatbot():
    """Initialize chatbot components (cached for performance)"""
    load_dotenv()
    
    try:
        # Use relative path from project root
        vector_store = VectorStore(persist_directory="vector_db")
        
        if vector_store.collection.count() == 0:
            return None, "Vector store is empty. Please run setup first."
        
        retriever = RAGRetriever(vector_store, top_k=5)
        llm_handler = LLMHandler()
        
        return (vector_store, retriever, llm_handler), None
        
    except Exception as e:
        return None, f"Error initializing chatbot: {str(e)}"


def display_message(role: str, content: str, response: ChatbotResponse = None):
    """Display a chat message with styling"""
    css_class = "user-message" if role == "user" else "bot-message"
    
    with st.container():
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <strong>{role.upper()}</strong><br/>
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        # Display sources and confidence for bot messages
        if role == "assistant" and response:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if response.sources:
                    with st.expander("Sources", expanded=False):
                        for i, source in enumerate(response.sources, 1):
                            st.markdown(f"**{i}.** {source}")
            
            with col2:
                confidence_class = f"confidence-{response.confidence}"
                st.markdown(f"""
                <div style="text-align: right;">
                    Confidence: <span class="{confidence_class}">{response.confidence.upper()}</span>
                </div>
                """, unsafe_allow_html=True)


def display_sidebar(vector_store):
    """Display sidebar with information and controls"""
    with st.sidebar:
        st.markdown("### OMINIMO Insurance Assistant")
        st.markdown("---")
        
        # System status
        st.markdown("#### System Status")
        
        if vector_store:
            stats = vector_store.get_stats()
            st.success("System Ready")
            
            with st.expander("Knowledge Base Stats", expanded=False):
                st.metric("Total Documents", stats['total_chunks'])
                
                st.markdown("**Sources:**")
                for source, count in stats['sources'].items():
                    st.markdown(f"- {source}: {count} chunks")
        else:
            st.error("System Not Ready")
        
        st.markdown("---")
        
        # Frequently asked questions
        st.markdown("#### Frequently Asked Questions")
        sample_questions = [
            "What does MTPL insurance cover?",
            "How do I file a claim?",
            "What is the deductible amount?",
            "What are the payment terms?",
            "What damages are excluded from coverage?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{hash(question)}", use_container_width=True):
                st.session_state.suggested_question = question
        
        st.markdown("---")
        
        # Settings
        with st.expander("Settings", expanded=False):
            st.session_state.show_retrieval_info = st.checkbox(
                "Show retrieval details", 
                value=st.session_state.get('show_retrieval_info', False)
            )
            
            st.session_state.show_timing = st.checkbox(
                "Show response timing", 
                value=st.session_state.get('show_timing', False)
            )
        
        st.markdown("---")
        
        # Clear conversation
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.8rem; color: #666;">
        <strong>About:</strong><br/>
        RAG-powered chatbot for car insurance queries.
        Built for OMINIMO assessment.
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application"""
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'suggested_question' not in st.session_state:
        st.session_state.suggested_question = None
    
    # Initialize chatbot
    init_result, error = initialize_chatbot()
    
    if error:
        st.error(f"{error}")
        st.info("Please run the setup script to build the vector database.")
        st.code("python src/vector_store.py", language="bash")
        return
    
    vector_store, retriever, llm_handler = init_result
    
    # Display sidebar
    display_sidebar(vector_store)
    
    # Main content
    st.markdown('<div class="main-header">OMINIMO Insurance Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask me anything about your car insurance policy</div>', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(
            message["role"], 
            message["content"],
            message.get("response")
        )
    
    # Chat input
    query = st.chat_input("Type your question here...")
    
    # Handle suggested question from sidebar
    if st.session_state.suggested_question:
        query = st.session_state.suggested_question
        st.session_state.suggested_question = None
    
    if query:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })
        
        # Display user message
        display_message("user", query)
        
        # Generate response
        with st.spinner("Thinking..."):
            start_time = time.time()
            
            # Retrieve relevant documents
            retrieval_results = retriever.retrieve(query)
            retrieval_time = time.time() - start_time
            
            # Generate answer
            response = llm_handler.generate_answer(query, retrieval_results)
            total_time = time.time() - start_time
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "response": response
        })
        
        # Display assistant message
        display_message("assistant", response.answer, response)
        
        # Show timing if enabled
        if st.session_state.get('show_timing', False):
            st.caption(f"Retrieval: {retrieval_time:.2f}s | Total: {total_time:.2f}s")
        
        # Show retrieval info if enabled
        if st.session_state.get('show_retrieval_info', False):
            with st.expander("Retrieval Details", expanded=False):
                for i, result in enumerate(retrieval_results, 1):
                    st.markdown(f"""
                    **Result {i}:** {result.source} (Page {result.page})
                    - Section: {result.section}
                    - Relevance: {result.relevance_score:.3f}
                    - Text: {result.text[:200]}...
                    """)
        
        # Generate follow-up questions
        if response.is_in_scope:
            followups = llm_handler.generate_followup_questions(query, response.answer)
            if followups:
                st.markdown("**You might also want to ask:**")
                cols = st.columns(min(len(followups), 3))
                for i, fq in enumerate(followups):
                    with cols[i % 3]:
                        if st.button(fq, key=f"followup_{i}", use_container_width=True):
                            st.session_state.suggested_question = fq
                            st.rerun()
        
        st.rerun()


if __name__ == "__main__":
    main()
