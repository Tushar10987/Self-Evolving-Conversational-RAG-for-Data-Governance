"""
A/B Experiment Engine for RAG system evaluation.
Runs experiments comparing different retrieval strategies and computes metrics.
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time
import logging
import uuid
from pathlib import Path
import random

# Import our modules
from retrieve import create_retriever, EnsembleRetriever
from hyde import create_hyde_retriever, HyDERetriever
from critic import create_critic, CriticModel

from config import experiment_config, data_config, system_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    experiment_id: str
    strategy_name: str
    query: str
    response: str
    retrieved_docs: List[str]
    retrieval_time: float
    critic_score: float
    hallucination_score: float
    sensitivity_score: float
    factuality_score: float
    user_satisfaction: float
    precision_at_5: float
    precision_at_10: float
    retriever_weights: Dict[str, float]
    timestamp: str

@dataclass
class ExperimentSummary:
    """Summary of experiment results."""
    experiment_id: str
    strategy_name: str
    total_queries: int
    avg_retrieval_time: float
    avg_critic_score: float
    avg_hallucination_score: float
    avg_sensitivity_score: float
    avg_factuality_score: float
    avg_user_satisfaction: float
    avg_precision_at_5: float
    avg_precision_at_10: float
    p95_retrieval_time: float
    retriever_weights: Dict[str, float]

class TestQueryGenerator:
    """Generates test queries for experiments."""
    
    def __init__(self):
        self.query_templates = [
            "Who owns the {table_type} table?",
            "What tables contain {data_type} information?",
            "Show me data lineage for {domain} data",
            "Which {department} tables have {attribute}?",
            "What is the {metric} of {table_type} tables?",
            "Who has access to {sensitive_data} data?",
            "Which tables are {condition}?",
            "Show me {relationship} for {table_type}",
            "What {compliance} requirements apply to {data_type}?",
            "How is {data_type} data {processed}?"
        ]
        
        self.table_types = ["customer", "transaction", "user", "product", "financial", "analytics"]
        self.data_types = ["sensitive", "personal", "financial", "transactional", "analytical", "audit"]
        self.domains = ["customer", "financial", "operational", "marketing", "sales", "hr"]
        self.departments = ["Engineering", "Finance", "Marketing", "Sales", "HR", "Operations"]
        self.attributes = ["encrypted", "masked", "restricted", "public", "archived"]
        self.metrics = ["size", "row count", "last updated", "access frequency"]
        self.sensitive_data = ["personal", "financial", "health", "confidential"]
        self.conditions = ["owned by Engineering", "containing PII", "masked", "encrypted"]
        self.relationships = ["dependencies", "lineage", "ownership", "access patterns"]
        self.compliance = ["GDPR", "CCPA", "SOX", "HIPAA", "PCI"]
        self.processed = ["masked", "encrypted", "aggregated", "filtered", "transformed"]
    
    def generate_queries(self, count: int = None) -> List[str]:
        """Generate test queries."""
        count = count or experiment_config.TEST_QUERIES_COUNT
        queries = []
        
        for i in range(count):
            template = random.choice(self.query_templates)
            
            # Fill template with random values
            query = template.format(
                table_type=random.choice(self.table_types),
                data_type=random.choice(self.data_types),
                domain=random.choice(self.domains),
                department=random.choice(self.departments),
                attribute=random.choice(self.attributes),
                metric=random.choice(self.metrics),
                sensitive_data=random.choice(self.sensitive_data),
                condition=random.choice(self.conditions),
                relationship=random.choice(self.relationships),
                compliance=random.choice(self.compliance),
                processed=random.choice(self.processed)
            )
            
            queries.append(query)
        
        return queries

class UserSatisfactionSimulator:
    """Simulates user satisfaction based on response quality."""
    
    def __init__(self):
        self.satisfaction_factors = {
            "response_length": 0.1,
            "specificity": 0.2,
            "relevance": 0.3,
            "completeness": 0.2,
            "clarity": 0.2
        }
    
    def simulate_satisfaction(self, response: str, query: str, retrieved_docs: List[str]) -> float:
        """Simulate user satisfaction score."""
        score = 0.0
        
        # Response length factor
        word_count = len(response.split())
        if 10 <= word_count <= 100:
            score += 0.1
        elif word_count > 100:
            score += 0.05
        
        # Specificity factor (check for specific names, numbers, etc.)
        specific_indicators = ["table", "user", "department", "date", "count", "size"]
        specificity_count = sum(1 for indicator in specific_indicators if indicator in response.lower())
        score += min(specificity_count * 0.02, 0.2)
        
        # Relevance factor (keyword overlap with query)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        relevance = min(overlap / max(len(query_words), 1), 1.0)
        score += relevance * 0.3
        
        # Completeness factor (based on retrieved docs)
        if retrieved_docs:
            doc_content = " ".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in retrieved_docs])
            doc_words = set(doc_content.lower().split())
            response_words = set(response.lower().split())
            completeness = len(response_words.intersection(doc_words)) / max(len(response_words), 1)
            score += completeness * 0.2
        
        # Clarity factor (check for clear structure)
        clarity_indicators = ["because", "therefore", "specifically", "for example", "in summary"]
        clarity_count = sum(1 for indicator in clarity_indicators if indicator in response.lower())
        score += min(clarity_count * 0.05, 0.2)
        
        return min(score, 1.0)

class PrecisionCalculator:
    """Calculates precision metrics for retrieval."""
    
    def __init__(self):
        self.relevance_keywords = {
            "ownership": ["owner", "owned", "responsible", "steward"],
            "lineage": ["lineage", "dependency", "feeds", "source", "target"],
            "sensitivity": ["sensitive", "masked", "encrypted", "restricted", "pii"],
            "access": ["access", "permission", "grant", "role", "user"],
            "compliance": ["gdpr", "ccpa", "sox", "hipaa", "compliance"]
        }
    
    def calculate_precision(self, query: str, retrieved_docs: List[str], k: int) -> float:
        """Calculate precision@k for retrieved documents."""
        if not retrieved_docs:
            return 0.0
        
        # Determine query type
        query_lower = query.lower()
        query_type = None
        
        for category, keywords in self.relevance_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                query_type = category
                break
        
        if not query_type:
            # Default to general relevance
            query_type = "general"
        
        # Check relevance of top-k documents
        relevant_count = 0
        top_k_docs = retrieved_docs[:k]
        
        for doc in top_k_docs:
            doc_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            doc_lower = doc_content.lower()
            
            # Check if document is relevant based on query type
            if self._is_relevant(doc_lower, query_type, query_lower):
                relevant_count += 1
        
        return relevant_count / len(top_k_docs) if top_k_docs else 0.0
    
    def _is_relevant(self, doc_content: str, query_type: str, query: str) -> bool:
        """Check if document is relevant to query."""
        # Simple keyword-based relevance
        if query_type == "general":
            # Check for query keywords in document
            query_words = set(query.split())
            doc_words = set(doc_content.split())
            overlap = len(query_words.intersection(doc_words))
            return overlap >= 2  # At least 2 words overlap
        
        # Type-specific relevance
        relevant_keywords = self.relevance_keywords.get(query_type, [])
        return any(keyword in doc_content for keyword in relevant_keywords)

class ExperimentRunner:
    """Runs A/B experiments comparing different strategies."""
    
    def __init__(self):
        self.query_generator = TestQueryGenerator()
        self.satisfaction_simulator = UserSatisfactionSimulator()
        self.precision_calculator = PrecisionCalculator()
        self.critic = create_critic()
        
        # Load test queries
        self.test_queries = self._load_or_generate_queries()
    
    def _load_or_generate_queries(self) -> List[str]:
        """Load existing test queries or generate new ones."""
        queries_file = Path(data_config.DATA_DIR) / "test_queries.json"
        
        if queries_file.exists():
            with open(queries_file, 'r') as f:
                queries = json.load(f)
            logger.info(f"Loaded {len(queries)} existing test queries")
        else:
            queries = self.query_generator.generate_queries()
            
            # Save queries for future use
            with open(queries_file, 'w') as f:
                json.dump(queries, f, indent=2)
            logger.info(f"Generated and saved {len(queries)} test queries")
        
        return queries
    
    def run_experiment(self, strategy_name: str, retriever_weights: Dict[str, float], 
                      use_hyde: bool = False) -> List[ExperimentResult]:
        """Run a complete experiment with given strategy."""
        experiment_id = str(uuid.uuid4())
        results = []
        
        # Create retriever with specified weights
        base_retriever = create_retriever(retriever_weights)
        
        # Add HyDE if requested
        if use_hyde:
            retriever = create_hyde_retriever(base_retriever)
        else:
            retriever = base_retriever
        
        logger.info(f"Running experiment {experiment_id} with strategy: {strategy_name}")
        logger.info(f"Retriever weights: {retriever_weights}")
        logger.info(f"Using HyDE: {use_hyde}")
        
        for i, query in enumerate(self.test_queries):
            logger.info(f"Processing query {i+1}/{len(self.test_queries)}: {query}")
            
            try:
                # Retrieve documents
                start_time = time.time()
                
                if use_hyde:
                    retrieval_result = retriever.retrieve_with_hyde(query)
                    retrieved_docs = retrieval_result["combined_documents"]
                    retrieval_time = retrieval_result["total_time"]
                else:
                    retrieval_result = retriever.retrieve(query)
                    retrieved_docs = retrieval_result.documents
                    retrieval_time = retrieval_result.total_time
                
                # Generate response (simplified for demo)
                response = self._generate_response(query, retrieved_docs)
                
                # Evaluate with critic
                critic_score = self.critic.evaluate_response(response, retrieved_docs)
                
                # Calculate precision
                precision_at_5 = self.precision_calculator.calculate_precision(query, retrieved_docs, 5)
                precision_at_10 = self.precision_calculator.calculate_precision(query, retrieved_docs, 10)
                
                # Simulate user satisfaction
                user_satisfaction = self.satisfaction_simulator.simulate_satisfaction(
                    response, query, retrieved_docs
                )
                
                # Create result
                result = ExperimentResult(
                    experiment_id=experiment_id,
                    strategy_name=strategy_name,
                    query=query,
                    response=response,
                    retrieved_docs=[doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in retrieved_docs],
                    retrieval_time=retrieval_time,
                    critic_score=critic_score.overall_score,
                    hallucination_score=critic_score.hallucination_score,
                    sensitivity_score=critic_score.sensitivity_score,
                    factuality_score=critic_score.factuality_score,
                    user_satisfaction=user_satisfaction,
                    precision_at_5=precision_at_5,
                    precision_at_10=precision_at_10,
                    retriever_weights=retriever_weights,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                continue
        
        logger.info(f"Experiment {experiment_id} completed with {len(results)} results")
        return results
    
    def _generate_response(self, query: str, retrieved_docs: List) -> str:
        """Generate a response based on retrieved documents."""
        if not retrieved_docs:
            return "I don't have enough information to answer that question."
        
        # Extract document content
        doc_contents = []
        for doc in retrieved_docs:
            if hasattr(doc, 'page_content'):
                doc_contents.append(doc.page_content)
            else:
                doc_contents.append(str(doc))
        
        # Simple response generation based on document content
        response_parts = []
        
        # Add relevant information from documents
        for content in doc_contents[:3]:  # Use top 3 documents
            # Extract key information
            if "owner" in content.lower() and "owner" in query.lower():
                response_parts.append(content)
            elif "lineage" in content.lower() and "lineage" in query.lower():
                response_parts.append(content)
            elif "table" in content.lower() and "table" in query.lower():
                response_parts.append(content)
        
        if response_parts:
            return " ".join(response_parts[:2])  # Combine top 2 relevant parts
        else:
            return f"Based on the available information: {doc_contents[0][:200]}..."
    
    def run_ab_experiment(self) -> Dict[str, Any]:
        """Run A/B experiment comparing baseline vs tuned strategies."""
        logger.info("Starting A/B experiment...")
        
        # Run baseline experiment
        baseline_results = self.run_experiment(
            strategy_name="baseline",
            retriever_weights=experiment_config.BASELINE_WEIGHTS,
            use_hyde=False
        )
        
        # Run tuned experiment
        tuned_results = self.run_experiment(
            strategy_name="tuned",
            retriever_weights=experiment_config.TUNED_WEIGHTS,
            use_hyde=True
        )
        
        # Generate summaries
        baseline_summary = self._create_summary(baseline_results)
        tuned_summary = self._create_summary(tuned_results)
        
        # Save results
        self._save_experiment_results(baseline_results + tuned_results)
        
        return {
            "baseline": baseline_summary,
            "tuned": tuned_summary,
            "improvement": self._calculate_improvement(baseline_summary, tuned_summary)
        }
    
    def _create_summary(self, results: List[ExperimentResult]) -> ExperimentSummary:
        """Create summary statistics from experiment results."""
        if not results:
            return None
        
        return ExperimentSummary(
            experiment_id=results[0].experiment_id,
            strategy_name=results[0].strategy_name,
            total_queries=len(results),
            avg_retrieval_time=np.mean([r.retrieval_time for r in results]),
            avg_critic_score=np.mean([r.critic_score for r in results]),
            avg_hallucination_score=np.mean([r.hallucination_score for r in results]),
            avg_sensitivity_score=np.mean([r.sensitivity_score for r in results]),
            avg_factuality_score=np.mean([r.factuality_score for r in results]),
            avg_user_satisfaction=np.mean([r.user_satisfaction for r in results]),
            avg_precision_at_5=np.mean([r.precision_at_5 for r in results]),
            avg_precision_at_10=np.mean([r.precision_at_10 for r in results]),
            p95_retrieval_time=np.percentile([r.retrieval_time for r in results], 95),
            retriever_weights=results[0].retriever_weights
        )
    
    def _calculate_improvement(self, baseline: ExperimentSummary, tuned: ExperimentSummary) -> Dict[str, float]:
        """Calculate improvement percentages."""
        if not baseline or not tuned:
            return {}
        
        improvements = {}
        metrics = [
            "avg_critic_score", "avg_factuality_score", "avg_user_satisfaction",
            "avg_precision_at_5", "avg_precision_at_10"
        ]
        
        for metric in metrics:
            baseline_val = getattr(baseline, metric)
            tuned_val = getattr(tuned, metric)
            
            if baseline_val > 0:
                improvement = ((tuned_val - baseline_val) / baseline_val) * 100
                improvements[f"{metric}_improvement_pct"] = improvement
        
        return improvements
    
    def _save_experiment_results(self, results: List[ExperimentResult]):
        """Save experiment results to file."""
        # Convert to DataFrame
        df_data = []
        for result in results:
            df_data.append({
                "experiment_id": result.experiment_id,
                "strategy_name": result.strategy_name,
                "query": result.query,
                "response": result.response,
                "retrieval_time": result.retrieval_time,
                "critic_score": result.critic_score,
                "hallucination_score": result.hallucination_score,
                "sensitivity_score": result.sensitivity_score,
                "factuality_score": result.factuality_score,
                "user_satisfaction": result.user_satisfaction,
                "precision_at_5": result.precision_at_5,
                "precision_at_10": result.precision_at_10,
                "retriever_weights": json.dumps(result.retriever_weights),
                "timestamp": result.timestamp
            })
        
        df = pd.DataFrame(df_data)
        
        # Save to CSV
        results_file = Path(data_config.ARTIFACTS_DIR) / data_config.EXPERIMENT_RESULTS_FILE
        df.to_csv(results_file, index=False)
        logger.info(f"Saved experiment results to {results_file}")

def main():
    """Run the A/B experiment."""
    runner = ExperimentRunner()
    results = runner.run_ab_experiment()
    
    print("\n=== A/B Experiment Results ===")
    print(f"Baseline Strategy: {results['baseline'].strategy_name}")
    print(f"  - Avg Critic Score: {results['baseline'].avg_critic_score:.3f}")
    print(f"  - Avg Precision@5: {results['baseline'].avg_precision_at_5:.3f}")
    print(f"  - Avg User Satisfaction: {results['baseline'].avg_user_satisfaction:.3f}")
    print(f"  - P95 Retrieval Time: {results['baseline'].p95_retrieval_time:.3f}s")
    
    print(f"\nTuned Strategy: {results['tuned'].strategy_name}")
    print(f"  - Avg Critic Score: {results['tuned'].avg_critic_score:.3f}")
    print(f"  - Avg Precision@5: {results['tuned'].avg_precision_at_5:.3f}")
    print(f"  - Avg User Satisfaction: {results['tuned'].avg_user_satisfaction:.3f}")
    print(f"  - P95 Retrieval Time: {results['tuned'].p95_retrieval_time:.3f}s")
    
    print(f"\nImprovements:")
    for metric, improvement in results['improvement'].items():
        print(f"  - {metric}: {improvement:+.1f}%")

if __name__ == "__main__":
    main() 