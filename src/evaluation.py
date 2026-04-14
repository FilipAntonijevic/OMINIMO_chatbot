import os
import time
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from dotenv import load_dotenv
from vector_store import VectorStore
from retriever import RAGRetriever
from llm_handler import LLMHandler


@dataclass
class TestCase:
    question: str
    expected_topics: List[str]  
    expected_source: str 
    category: str 


@dataclass
class EvaluationResult:
    question: str
    answer: str
    sources: List[str]
    confidence: str
    retrieval_time: float
    generation_time: float
    total_time: float
    relevance_score: float  # 0-1
    completeness_score: float  # 0-1
    source_accuracy: bool
    passed: bool
    notes: str = ""


class ChatbotEvaluator:
    
    def __init__(self, vector_store: VectorStore, retriever: RAGRetriever, llm_handler: LLMHandler):
        self.vector_store = vector_store
        self.retriever = retriever
        self.llm_handler = llm_handler
        
        # Define test cases
        self.test_cases = self._create_test_cases()
    
    def _create_test_cases(self) -> List[TestCase]:
        return [
            TestCase(
                question="What does MTPL insurance cover?",
                expected_topics=["third party liability", "bodily injury", "property damage"],
                expected_source="MTPL Product Information",
                category="Coverage"
            ),
            TestCase(
                question="What damages are excluded from MTPL coverage?",
                expected_topics=["exclusions", "not covered"],
                expected_source="MTPL Product Information",
                category="Coverage"
            ),
            
            TestCase(
                question="How do I file a claim?",
                expected_topics=["claim procedure", "notification", "documentation"],
                expected_source="User Regulations",
                category="Claims"
            ),
            TestCase(
                question="What documents do I need to submit with a claim?",
                expected_topics=["documents", "evidence", "police report"],
                expected_source="User Regulations",
                category="Claims"
            ),
            
            TestCase(
                question="What are the payment terms for the premium?",
                expected_topics=["payment", "premium", "due date"],
                expected_source="Terms and Conditions",
                category="Terms"
            ),
            TestCase(
                question="Can I cancel my policy?",
                expected_topics=["cancellation", "termination", "notice period"],
                expected_source="Terms and Conditions",
                category="Terms"
            ),
            
            TestCase(
                question="Am I covered if I have an accident abroad?",
                expected_topics=["foreign", "abroad", "international", "territory"],
                expected_source="MTPL Product Information",
                category="Coverage"
            ),
            TestCase(
                question="What is the deductible amount?",
                expected_topics=["deductible", "excess", "self-participation"],
                expected_source="Terms and Conditions",
                category="Terms"
            ),
            
            TestCase(
                question="What happens if the other driver is uninsured?",
                expected_topics=["uninsured", "third party"],
                expected_source="MTPL Product Information",
                category="Coverage"
            ),
            
            TestCase(
                question="What is the best car to buy?",
                expected_topics=[],
                expected_source="",
                category="Out of Scope"
            ),
        ]
    
    def _calculate_relevance_score(self, answer: str, expected_topics: List[str]) -> float:
        if not expected_topics:
            return 1.0  
        
        answer_lower = answer.lower()
        covered_topics = sum(1 for topic in expected_topics if topic.lower() in answer_lower)
        
        return covered_topics / len(expected_topics)
    
    def _calculate_completeness_score(self, answer: str) -> float:
        
        word_count = len(answer.split())
        
        if word_count < 20:
            return 0.3
        
        if 20 <= word_count <= 200:
            return 1.0
        
        if word_count > 200:
            return 0.8
        
        return 0.5
    
    def _check_source_accuracy(self, sources: List[str], expected_source: str) -> bool:
        if not expected_source:
            return True 
        
        return any(expected_source in source for source in sources)
    
    def evaluate_test_case(self, test_case: TestCase) -> EvaluationResult:
        print(f"Testing: {test_case.question}")
        
        start_time = time.time()
        retrieval_results = self.retriever.retrieve(test_case.question)
        retrieval_time = time.time() - start_time
        
        gen_start = time.time()
        response = self.llm_handler.generate_answer(test_case.question, retrieval_results)
        generation_time = time.time() - gen_start
        
        total_time = time.time() - start_time
        
        relevance_score = self._calculate_relevance_score(response.answer, test_case.expected_topics)
        completeness_score = self._calculate_completeness_score(response.answer)
        source_accuracy = self._check_source_accuracy(response.sources, test_case.expected_source)
        
        if test_case.category == "Out of Scope":
            passed = not response.is_in_scope
        else:
            passed = (
                response.is_in_scope and
                relevance_score >= 0.5 and
                completeness_score >= 0.5 and
                source_accuracy
            )
        
        return EvaluationResult(
            question=test_case.question,
            answer=response.answer,
            sources=response.sources,
            confidence=response.confidence,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            source_accuracy=source_accuracy,
            passed=passed,
            notes=f"Category: {test_case.category}"
        )
    
    def run_evaluation(self) -> Dict:
        print("\n" + "="*60)
        print("RUNNING CHATBOT EVALUATION")
        print("="*60 + "\n")
        
        results = []
        
        for test_case in self.test_cases:
            result = self.evaluate_test_case(test_case)
            results.append(result)
            print(f"  {'✓' if result.passed else '✗'} {result.question[:50]}...")
            print(f"    Time: {result.total_time:.2f}s | Relevance: {result.relevance_score:.2f} | Complete: {result.completeness_score:.2f}")
            print()
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        
        avg_retrieval_time = sum(r.retrieval_time for r in results) / total_tests
        avg_generation_time = sum(r.generation_time for r in results) / total_tests
        avg_total_time = sum(r.total_time for r in results) / total_tests
        
        avg_relevance = sum(r.relevance_score for r in results) / total_tests
        avg_completeness = sum(r.completeness_score for r in results) / total_tests
        source_accuracy_rate = sum(1 for r in results if r.source_accuracy) / total_tests
        
        retrieval_precision = source_accuracy_rate
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": passed_tests / total_tests,
            "metrics": {
                "avg_retrieval_time": avg_retrieval_time,
                "avg_generation_time": avg_generation_time,
                "avg_total_time": avg_total_time,
                "avg_relevance_score": avg_relevance,
                "avg_completeness_score": avg_completeness,
                "source_accuracy_rate": source_accuracy_rate,
                "retrieval_precision_at_k": retrieval_precision
            },
            "results": [asdict(r) for r in results]
        }
        
        return summary
    
    def print_summary(self, summary: Dict):
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60 + "\n")
        
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']} ({summary['pass_rate']*100:.1f}%)")
        print()
        
        print("Performance Metrics:")
        metrics = summary['metrics']
        print(f"  Avg Retrieval Time: {metrics['avg_retrieval_time']:.3f}s")
        print(f"  Avg Generation Time: {metrics['avg_generation_time']:.3f}s")
        print(f"  Avg Total Time: {metrics['avg_total_time']:.3f}s")
        print()
        
        print("Quality Metrics:")
        print(f"  Avg Relevance Score: {metrics['avg_relevance_score']:.2f}")
        print(f"  Avg Completeness Score: {metrics['avg_completeness_score']:.2f}")
        print(f"  Source Accuracy Rate: {metrics['source_accuracy_rate']*100:.1f}%")
        print(f"  Retrieval Precision@K: {metrics['retrieval_precision_at_k']:.2f}")
        print()
        
        print("Results by Category:")
        results = summary['results']
        categories = set(r['notes'].split(': ')[1] for r in results)
        
        for category in sorted(categories):
            cat_results = [r for r in results if category in r['notes']]
            cat_passed = sum(1 for r in cat_results if r['passed'])
            print(f"  {category}: {cat_passed}/{len(cat_results)} passed")
    
    def save_results(self, summary: Dict, output_file: str = "evaluation_results.json"):
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def main():
    load_dotenv()
    
    print("Initializing chatbot components...")
    vector_store = VectorStore()
    
    if vector_store.collection.count() == 0:
        print("ERROR: Vector store is empty. Please run vector_store.py first.")
        return
    
    retriever = RAGRetriever(vector_store, top_k=5)
    llm_handler = LLMHandler()
    
    evaluator = ChatbotEvaluator(vector_store, retriever, llm_handler)
    
    summary = evaluator.run_evaluation()
    
    evaluator.print_summary(summary)
    
    evaluator.save_results(summary)
    
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    
    for i, result in enumerate(summary['results'], 1):
        print(f"\n[Test {i}] {result['question']}")
        print(f"Status: {'✓ PASS' if result['passed'] else '✗ FAIL'}")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {', '.join(result['sources'][:2]) if result['sources'] else 'None'}")
        print(f"Metrics: Relevance={result['relevance_score']:.2f}, Completeness={result['completeness_score']:.2f}, Time={result['total_time']:.2f}s")


if __name__ == "__main__":
    main()
