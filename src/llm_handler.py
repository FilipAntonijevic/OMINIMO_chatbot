import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from groq import Groq
from dotenv import load_dotenv
from retriever import RetrievalResult


@dataclass
class ChatbotResponse:
    answer: str
    sources: List[str]
    confidence: str 
    is_in_scope: bool
    reasoning: Optional[str] = None


class LLMHandler:
    def __init__(self, 
                 model: str = "llama-3.3-70b-versatile",
                 temperature: float = 0.1,
                 max_tokens: int = 1000):
       
        load_dotenv()
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        self.system_prompt = """You are an expert insurance assistant specializing in car insurance policies, specifically MTPL (Motor Third Party Liability) insurance.

Your role is to:
1. Answer questions ONLY based on the provided context from official insurance documents
2. Provide accurate, clear, and professional responses
3. If the context doesn't contain enough information to answer the question fully, say so explicitly
4. Never make up information or provide answers not supported by the context
5. Use simple language that customers can understand while maintaining professionalism
6. ALWAYS respond in ENGLISH, even if the source documents are in Hungarian - translate the key information to English

When answering:
- Start with a direct answer to the question
- Support your answer with specific details from the context
- Be concise but thorough
- Translate Hungarian terms to English (e.g., "biztosítás" → "insurance", "kár" → "damage", "fedezet" → "coverage")
- DO NOT mention source citations in your answer (e.g., do NOT write "According to [Source 1]" or "as stated in [Source 2]")
- Write naturally without referencing sources - the sources will be displayed separately to the user

If the question is outside the scope of the provided insurance documents (e.g., about cooking, sports, etc.), politely explain that you can only answer questions about car insurance policies."""
    
    def _check_scope(self, query: str) -> Tuple[bool, str]:
        query_lower = query.lower().strip()
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
                     'zdravo', 'cao', 'hej', 'pozdrav']
        farewells = ['bye', 'goodbye', 'see you', 'thank you', 'thanks']
        simple_queries = ['how are you', 'what can you do', 'help', 'who are you', 
                          'what is this', 'explain', 'tell me about yourself']
        
        if any(greeting in query_lower for greeting in greetings):
            return True, "greeting"
        
        if any(farewell in query_lower for farewell in farewells):
            return True, "farewell"
        
        if any(simple in query_lower for simple in simple_queries):
            return True, "conversational"
        
        scope_check_prompt = f"""Determine if the following question is related to car insurance, MTPL insurance, insurance policies, claims, coverage, premiums, payment terms, or any general insurance topics.

Question: "{query}"

IMPORTANT: Even if the question is phrased differently or uses different word order, if it asks about insurance topics (like "terms of payment", "payment terms", "how to pay", etc.), answer YES.

Respond with ONLY "YES" if the question is insurance-related in ANY way, or "NO" if it's completely unrelated (e.g., cooking, sports, weather, general knowledge).
Then on a new line, briefly explain your reasoning in one sentence."""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",  
                messages=[{"role": "user", "content": scope_check_prompt}],
                temperature=0,
                max_tokens=100
            )
            
            answer = response.choices[0].message.content.strip()
            lines = answer.split('\n', 1)
            
            is_in_scope = lines[0].strip().upper() == "YES"
            reasoning = lines[1].strip() if len(lines) > 1 else ""
            
            return is_in_scope, reasoning
            
        except Exception as e:
            print(f"Error in scope check: {e}")
            return True, "Scope check failed, assuming in-scope"
    
    def _assess_confidence(self, 
                          query: str, 
                          context: str, 
                          answer: str,
                          retrieval_scores: List[float]) -> str:
        
        avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0
        
        uncertainty_phrases = [
            "i don't have", "not mentioned", "unclear", "cannot find",
            "insufficient information", "not specified", "not clear",
            "i apologize", "outside my area", "cannot answer"
        ]
        has_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)
        
        is_substantial = len(answer.split()) > 20
        
        if avg_retrieval_score > 0.4 and not has_uncertainty and is_substantial:
            return "high"
        elif avg_retrieval_score > 0.20 and is_substantial:
            return "medium"
        else:
            return "low"
    
    def generate_answer(self,
                       query: str,
                       retrieval_results: List[RetrievalResult],
                       include_reasoning: bool = False) -> ChatbotResponse:
        
        is_in_scope, scope_reasoning = self._check_scope(query)
        
        if scope_reasoning == "greeting":
            return ChatbotResponse(
                answer="Hello! I'm your OMINIMO Insurance Assistant. I can help you with questions about car insurance, MTPL insurance, policy coverage, claims, and related insurance topics. How can I assist you today?",
                sources=[],
                confidence="high",
                is_in_scope=True,
                reasoning=scope_reasoning if include_reasoning else None
            )
        
        if scope_reasoning == "farewell":
            return ChatbotResponse(
                answer="Thank you for using OMINIMO Insurance Assistant. If you have any more questions about your insurance, feel free to ask anytime. Have a great day!",
                sources=[],
                confidence="high",
                is_in_scope=True,
                reasoning=scope_reasoning if include_reasoning else None
            )
        
        if scope_reasoning == "conversational":
            return ChatbotResponse(
                answer="I'm your OMINIMO Insurance Assistant. I can help you with questions about car insurance policies, MTPL insurance, coverage details, claims procedures, payment terms, and other insurance-related topics. Feel free to ask me anything about your insurance!",
                sources=[],
                confidence="high",
                is_in_scope=True,
                reasoning=scope_reasoning if include_reasoning else None
            )
        
        if not is_in_scope:
            return ChatbotResponse(
                answer="I can only answer questions about OMINIMO insurance products and services. This includes car insurance (MTPL), policy coverage, claims procedures, payment terms, and other insurance-related topics. Could you please rephrase your question to focus on insurance matters?",
                sources=[],
                confidence="high",
                is_in_scope=False,
                reasoning=scope_reasoning if include_reasoning else None
            )
        
        if not retrieval_results:
            return ChatbotResponse(
                answer="I don't have enough information in the available documents to answer your question. Please try rephrasing your question or contact customer support for more specific assistance.",
                sources=[],
                confidence="low",
                is_in_scope=True,
                reasoning="No relevant context found" if include_reasoning else None
            )
        
        context_parts = []
        sources_list = []
        
        for i, result in enumerate(retrieval_results, 1):
            context_part = f"[Source {i}: {result.source}, Page {result.page}]\n{result.text}"
            context_parts.append(context_part)
            sources_list.append(result.format_citation())
        
        context = "\n\n".join(context_parts)
        
        user_prompt = f"""Context from insurance documents (may be in Hungarian):

{context}

Question: {query}

Please provide a clear and accurate answer IN ENGLISH based solely on the context above. Translate any Hungarian information to English. Write naturally without mentioning source numbers - just provide the information directly."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            
            retrieval_scores = [r.relevance_score for r in retrieval_results]
            confidence = self._assess_confidence(query, context, answer, retrieval_scores)
            
            seen_sources = set()
            unique_sources = []
            for source in sources_list:
                if source not in seen_sources:
                    seen_sources.add(source)
                    unique_sources.append(source)
            
            return ChatbotResponse(
                answer=answer,
                sources=unique_sources,
                confidence=confidence,
                is_in_scope=True,
                reasoning=f"Used {len(retrieval_results)} sources" if include_reasoning else None
            )
            
        except Exception as e:
            error_message = str(e)
            print(f"Error generating answer: {e}")
            
            if "rate_limit" in error_message.lower() or "429" in error_message:
                return ChatbotResponse(
                    answer="Token limit reached. The daily API quota has been exceeded. Please try again later or contact support.",
                    sources=[],
                    confidence="low",
                    is_in_scope=True,
                    reasoning=f"Rate limit error: {str(e)}" if include_reasoning else None
                )
            else:
                return ChatbotResponse(
                    answer="I apologize, but I encountered an error generating a response. Please try again.",
                    sources=[],
                    confidence="low",
                    is_in_scope=True,
                    reasoning=f"Error: {str(e)}" if include_reasoning else None
                )
    
    def generate_followup_questions(self, 
                                   query: str, 
                                   answer: str,
                                   num_questions: int = 3) -> List[str]:
       
        prompt = f"""Based on this insurance Q&A, suggest {num_questions} relevant follow-up questions that a customer might want to ask:

Question: {query}
Answer: {answer}

Provide ONLY the questions, one per line, without numbering or explanations."""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            questions = response.choices[0].message.content.strip().split('\n')
            return [q.strip('- 0123456789.') for q in questions if q.strip()][:num_questions]
            
        except Exception as e:
            print(f"Error generating follow-up questions: {e}")
            return []


if __name__ == "__main__":
    from vector_store import VectorStore
    from retriever import RAGRetriever
    
    print("Initializing components...")
    vector_store = VectorStore()
    
    if vector_store.collection.count() == 0:
        print("Vector store is empty. Please run vector_store.py first.")
    else:
        retriever = RAGRetriever(vector_store, top_k=5)
        llm_handler = LLMHandler()
        
        test_queries = [
            "What does MTPL insurance cover?",
            "How do I file a claim?",
            "What is the best recipe for chocolate cake?", 
            "What are the payment terms?"
        ]
        
        print("\n" + "="*60)
        print("TESTING LLM HANDLER:")
        print("="*60)
        
        for query in test_queries:
            print(f"\n\nQuery: {query}")
            print("-" * 60)
            
            results = retriever.retrieve(query)
            
            response = llm_handler.generate_answer(query, results, include_reasoning=True)
            
            print(f"\nIn Scope: {response.is_in_scope}")
            print(f"Confidence: {response.confidence}")
            print(f"\nAnswer:\n{response.answer}")
            
            if response.sources:
                print(f"\nSources:")
                for source in response.sources:
                    print(f"  - {source}")
            
            if response.reasoning:
                print(f"\nReasoning: {response.reasoning}")
            
            if response.is_in_scope:
                followups = llm_handler.generate_followup_questions(query, response.answer)
                if followups:
                    print(f"\nSuggested follow-up questions:")
                    for fq in followups:
                        print(f"  • {fq}")
