#!/usr/bin/env python3
"""Quick test of the Groq-powered insurance chatbot"""

import sys
sys.path.append('src')

from vector_store import VectorStore
from retriever import RAGRetriever
from llm_handler import LLMHandler

print("="*60)
print("OMINIMO Insurance Chatbot - Quick Test")
print("="*60)

# Initialize components
print("\n1. Loading vector store...")
vector_store = VectorStore(persist_directory="vector_db")
print(f"   ✓ Loaded {vector_store.collection.count()} document chunks")

print("\n2. Initializing retriever...")
retriever = RAGRetriever(vector_store, top_k=5)
print("   ✓ Retriever ready")

print("\n3. Initializing Groq LLM handler...")
llm_handler = LLMHandler()
print("   ✓ LLM handler ready (using Llama 3.1 70B)")

# Test queries
test_queries = [
    "What is the exact coverage amount? How much money does the insurance cover?",
    "What are the payment terms?",
]

print("\n" + "="*60)
print("TESTING CHATBOT:")
print("="*60)

for i, query in enumerate(test_queries, 1):
    print(f"\n\nTest {i}: {query}")
    print("-" * 60)
    
    # Retrieve context
    print("   -> Retrieving relevant context...")
    results = retriever.retrieve(query)
    print(f"   -> Found {len(results)} relevant chunks")
    
    # Show retrieval details
    print("\n   RETRIEVED CHUNKS:")
    for j, doc in enumerate(results, 1):
        score = doc.relevance_score
        source = doc.source
        page = doc.page
        print(f"      [{j}] Score: {score:.3f} | {source} Page {page}")
        print(f"          Content: {doc.text[:150]}...")
    
    # Generate answer
    print("\n   -> Generating answer with Groq...")
    response = llm_handler.generate_answer(query, results)
    
    print(f"\n   [OK] In Scope: {response.is_in_scope}")
    print(f"   [OK] Confidence: {response.confidence}")
    print(f"\n   ANSWER:\n   {response.answer}\n")
    
    if response.sources:
        print(f"   Sources:")
        for source in response.sources:
            print(f"      - {source}")

print("\n" + "="*60)
print("Test completed successfully!")
print("="*60)
print("\nTo launch full Streamlit UI:")
print("   streamlit run app.py")
