import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from vector_store import VectorStore


@dataclass
class RetrievalResult:
    text: str
    source: str
    page: int
    section: str
    relevance_score: float
    chunk_id: str
    
    def format_citation(self) -> str:
        """Format as a citation string"""
        return f"[{self.source}, Page {self.page}, Section: {self.section}]"


class RAGRetriever:
    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        load_dotenv()
        self.vector_store = vector_store
        self.top_k = top_k
        self.retrieval_factor = 2  
    
    def _calculate_relevance_score(self, distance: float) -> float:
        
        import math
        normalized = (distance - 15) / 3 
        score = 1 / (1 + math.exp(normalized))
        return score
    
    def _keyword_boost(self, query: str, text: str, source: str = "") -> float:
        
        query_lower = query.lower()
        text_lower = text.lower()
        source_lower = source.lower()
        
        important_keywords = [
            'coverage', 'claim', 'premium', 'deductible', 'liability',
            'policy', 'insurance', 'damage', 'accident', 'mtpl',
            'terms', 'conditions', 'regulations', 'cover', 'fedezet',
            'biztosítás', 'kár'
        ]
        
        answer_indicators = [
            'million', 'euros', 'eur', 'huf', 'indemnify', 'obliged to',
            'limit of', 'up to', 'maximum', 'injured parties'
        ]
        
        boost = 1.0
        query_words = set(query_lower.split())
        
       
        amount_queries = ['how much', 'amount', 'coverage', 'limit', 'maximum', 'money', 'cost', 'price', 'pay']
        if any(aq in query_lower for aq in amount_queries):
            if any(indicator in text_lower for indicator in answer_indicators):
                boost += 3.0  
        
        if 'mtpl' in query_lower and 'mtpl product' in source_lower:
            boost += 1.5
        
        if query_lower in text_lower:
            boost += 0.3
        
        for keyword in important_keywords:
            if keyword in query_words and keyword in text_lower:
                boost += 0.15
        
        return boost
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        seen_texts = set()
        unique_results = []
        
        for result in results:
            fingerprint = result['text'][:100].strip()
            
            if fingerprint not in seen_texts:
                seen_texts.add(fingerprint)
                unique_results.append(result)
        
        return unique_results
    
    def retrieve(self, 
                 query: str, 
                 filter_source: Optional[str] = None,
                 apply_reranking: bool = True) -> List[RetrievalResult]:
        
        query_lower = query.lower()
        expanded_query = query
        
        coverage_amount_indicators = ['coverage amount', 'coverage limit', 'how much coverage',
                                      'how much money', 'insurance cover', 'maximum coverage',
                                      'coverage sum', 'insurance amount', 'coverage value']
        
        if any(indicator in query_lower for indicator in coverage_amount_indicators):
            expanded_query = f"{query} insurance limit million euros indemnify injured parties property damage personal injury"
        
        mtpl_mentioned = 'mtpl' in query_lower or 'motor third party' in query_lower or 'kgfb' in query_lower
        
        num_results = self.top_k * self.retrieval_factor if apply_reranking else self.top_k
        
        filter_dict = {"source": filter_source} if filter_source else None
        
        raw_results = self.vector_store.search(
            query=expanded_query,  
            filter_dict=filter_dict
        )
        
        mtpl_results = []
        if mtpl_mentioned and not filter_source:
            mtpl_results = self.vector_store.search(
                query=expanded_query,  
                top_k=3,  
                filter_dict={"source": "MTPL Product Information"}
            )
        
        if mtpl_results:
            for mtpl_res in mtpl_results:
                mtpl_res['distance'] = mtpl_res['distance'] * 0.5  
            raw_results = mtpl_results + raw_results
        
        if not raw_results:
            return []
        
        scored_results = []
        for result in raw_results:
            base_score = self._calculate_relevance_score(result['distance'])
            
            if apply_reranking:
                boost = self._keyword_boost(query, result['text'], result['metadata']['source'])
                final_score = base_score * boost
            else:
                final_score = base_score
            
            result['final_score'] = final_score
            scored_results.append(result)
        
        scored_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        scored_results = self._deduplicate_results(scored_results)
        
        top_results = scored_results[:self.top_k]
        
        retrieval_results = []
        for result in top_results:
            retrieval_result = RetrievalResult(
                text=result['text'],
                source=result['metadata']['source'],
                page=result['metadata']['page'],
                section=result['metadata']['section'],
                relevance_score=result['final_score'],
                chunk_id=result['id']
            )
            retrieval_results.append(retrieval_result)
        
        return retrieval_results
    
    def retrieve_with_context_expansion(self, query: str) -> List[RetrievalResult]:
        initial_results = self.retrieve(query, apply_reranking=True)
        
        expanded_results = []
        seen_ids = set()
        
        for result in initial_results[:3]: 
            if result.chunk_id not in seen_ids:
                expanded_results.append(result)
                seen_ids.add(result.chunk_id)
            
        for result in initial_results[3:]:
            if result.chunk_id not in seen_ids:
                expanded_results.append(result)
                seen_ids.add(result.chunk_id)
        
        return expanded_results[:self.top_k]
    
    def format_context_for_llm(self, results: List[RetrievalResult]) -> str:
        
        if not results:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_part = f"""
[Source {i}: {result.source}, Page {result.page}, Section: {result.section}]
{result.text}
"""
            context_parts.append(context_part.strip())
        
        return "\n\n".join(context_parts)


if __name__ == "__main__":
    print("Initializing retriever...")
    
    vector_store = VectorStore()
    
    if vector_store.collection.count() == 0:
        print("Vector store is empty. Please run vector_store.py first to build the index.")
    else:
        retriever = RAGRetriever(vector_store, top_k=5)
        
        # Test queries
        test_queries = [
            "What does MTPL insurance cover?",
            "How do I file a claim?",
            "What are the policy terms and conditions?",
            "What is the premium calculation based on?"
        ]
        
        print("\n" + "="*60)
        print("TESTING RETRIEVAL:")
        print("="*60)
        
        for query in test_queries:
            print(f"\n\nQuery: {query}")
            print("-" * 60)
            
            results = retriever.retrieve(query)
            
            for i, result in enumerate(results, 1):
                print(f"\n[{i}] {result.format_citation()}")
                print(f"Relevance: {result.relevance_score:.3f}")
                print(f"Text: {result.text[:150]}...")
            
            print("\n" + "="*60)
            print("FORMATTED CONTEXT FOR LLM:")
            print("="*60)
            context = retriever.format_context_for_llm(results[:2])
            print(context[:500] + "...")
