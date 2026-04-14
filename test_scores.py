#!/usr/bin/env python3
"""Check retrieval scores for common questions"""

import sys
sys.path.append('src')

from vector_store import VectorStore
from retriever import RAGRetriever

print("Loading vector store...")
vector_store = VectorStore(persist_directory="vector_db")
retriever = RAGRetriever(vector_store, top_k=5)

test_queries = [
    "What does MTPL insurance cover?",
    "How do I file a claim?",
    "What is the deductible amount?",
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    results = retriever.retrieve(query)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.relevance_score:.4f}")
        print(f"   Source: {result.source}, Page {result.page}")
        print(f"   Text preview: {result.text[:100]}...")
    
    scores = [r.relevance_score for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\n   → Average retrieval score: {avg_score:.4f}")
