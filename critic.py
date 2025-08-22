"""
Critic model for hallucination detection and sensitivity scoring.
Evaluates the quality and safety of generated responses.
"""

import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time
import logging

# Text processing
from sentence_transformers import SentenceTransformer
import torch

from config import critic_config, llm_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CriticScore:
    """Score from the critic model."""
    hallucination_score: float
    sensitivity_score: float
    factuality_score: float
    overall_score: float
    reasoning: str
    evaluation_time: float

class HallucinationDetector:
    """Detects potential hallucinations in responses."""
    
    def __init__(self):
        self.sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP address
            r'\b\d{10,11}\b',  # Phone numbers
        ]
        
        self.hallucination_indicators = [
            "i don't have access to",
            "i cannot provide",
            "i don't know",
            "i'm not sure",
            "i don't have information",
            "no data available",
            "information not found",
            "unable to determine",
            "cannot verify",
            "no records show"
        ]
        
        self.confidence_indicators = [
            "definitely",
            "certainly",
            "absolutely",
            "without a doubt",
            "clearly",
            "obviously",
            "evidently",
            "undoubtedly"
        ]
    
    def detect_hallucination(self, response: str, retrieved_docs: List[str]) -> Tuple[float, str]:
        """Detect potential hallucinations in the response."""
        score = 0.0
        reasoning = []
        
        # Check for sensitive information exposure
        sensitive_count = 0
        for pattern in self.sensitive_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            sensitive_count += len(matches)
        
        if sensitive_count > 0:
            score += 0.3
            reasoning.append(f"Found {sensitive_count} potential sensitive data patterns")
        
        # Check for hallucination indicators
        response_lower = response.lower()
        hallucination_count = 0
        for indicator in self.hallucination_indicators:
            if indicator in response_lower:
                hallucination_count += 1
        
        if hallucination_count > 0:
            score += 0.2
            reasoning.append(f"Found {hallucination_count} uncertainty indicators")
        
        # Check for overconfidence
        confidence_count = 0
        for indicator in self.confidence_indicators:
            if indicator in response_lower:
                confidence_count += 1
        
        if confidence_count > 2:
            score += 0.2
            reasoning.append(f"High confidence indicators ({confidence_count}) without strong evidence")
        
        # Check for factual consistency with retrieved documents
        if retrieved_docs:
            consistency_score = self._check_factual_consistency(response, retrieved_docs)
            score += (1.0 - consistency_score) * 0.3
            reasoning.append(f"Factual consistency score: {consistency_score:.2f}")
        
        # Check for vague or generic responses
        if len(response.split()) < 10:
            score += 0.1
            reasoning.append("Response is too short/vague")
        
        # Normalize score to 0-1 range
        score = min(score, 1.0)
        
        return score, "; ".join(reasoning) if reasoning else "No hallucination indicators detected"
    
    def _check_factual_consistency(self, response: str, retrieved_docs: List[str]) -> float:
        """Check if response is consistent with retrieved documents."""
        if not retrieved_docs:
            return 0.0
        
        # Simple keyword overlap check
        response_words = set(response.lower().split())
        doc_words = set()
        
        for doc in retrieved_docs:
            doc_words.update(doc.lower().split())
        
        # Calculate overlap
        overlap = len(response_words.intersection(doc_words))
        total_unique = len(response_words.union(doc_words))
        
        if total_unique == 0:
            return 0.0
        
        return overlap / total_unique

class SensitivityDetector:
    """Detects sensitive information in responses."""
    
    def __init__(self):
        self.sensitive_keywords = [
            "password", "secret", "private", "confidential", "restricted",
            "ssn", "social security", "credit card", "bank account",
            "medical", "health", "diagnosis", "treatment",
            "salary", "compensation", "bonus", "income",
            "address", "phone", "email", "personal",
            "pii", "phi", "pci", "gdpr", "ccpa"
        ]
        
        self.data_governance_sensitive = [
            "masked", "encrypted", "redacted", "anonymized",
            "sensitive", "restricted access", "need to know",
            "compliance", "audit", "violation", "breach"
        ]
    
    def detect_sensitivity(self, response: str) -> Tuple[float, str]:
        """Detect sensitive information in the response."""
        score = 0.0
        reasoning = []
        
        response_lower = response.lower()
        
        # Check for sensitive keywords
        sensitive_count = 0
        found_keywords = []
        
        for keyword in self.sensitive_keywords:
            if keyword in response_lower:
                sensitive_count += 1
                found_keywords.append(keyword)
        
        if sensitive_count > 0:
            score += min(sensitive_count * 0.1, 0.5)
            reasoning.append(f"Found {sensitive_count} sensitive keywords: {', '.join(found_keywords)}")
        
        # Check for data governance sensitive terms
        governance_count = 0
        for term in self.data_governance_sensitive:
            if term in response_lower:
                governance_count += 1
        
        if governance_count > 0:
            score += min(governance_count * 0.05, 0.3)
            reasoning.append(f"Found {governance_count} data governance sensitive terms")
        
        # Check for specific data patterns
        patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
        
        pattern_count = 0
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                pattern_count += len(matches)
                reasoning.append(f"Found {len(matches)} {pattern_name} patterns")
        
        if pattern_count > 0:
            score += min(pattern_count * 0.15, 0.4)
        
        # Check for potential data exposure
        if "original value" in response_lower or "unmasked" in response_lower:
            score += 0.3
            reasoning.append("Potential exposure of original/unmasked data")
        
        # Normalize score
        score = min(score, 1.0)
        
        return score, "; ".join(reasoning) if reasoning else "No sensitive information detected"

class FactualityChecker:
    """Checks factual accuracy of responses."""
    
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def check_factuality(self, response: str, retrieved_docs: List[str]) -> Tuple[float, str]:
        """Check factual accuracy against retrieved documents."""
        if not retrieved_docs:
            return 0.0, "No retrieved documents to check against"
        
        try:
            # Encode response and documents
            response_embedding = self.encoder.encode([response])
            doc_embeddings = self.encoder.encode(retrieved_docs)
            
            # Calculate cosine similarities
            similarities = []
            for doc_embedding in doc_embeddings:
                similarity = np.dot(response_embedding[0], doc_embedding) / (
                    np.linalg.norm(response_embedding[0]) * np.linalg.norm(doc_embedding)
                )
                similarities.append(similarity)
            
            # Use max similarity as factuality score
            max_similarity = max(similarities)
            
            reasoning = f"Max similarity with retrieved docs: {max_similarity:.3f}"
            
            return max_similarity, reasoning
            
        except Exception as e:
            logger.error(f"Error in factuality check: {e}")
            return 0.0, f"Error in factuality check: {str(e)}"

class CriticModel:
    """Main critic model that combines all evaluation components."""
    
    def __init__(self):
        self.hallucination_detector = HallucinationDetector()
        self.sensitivity_detector = SensitivityDetector()
        self.factuality_checker = FactualityChecker()
        
    def evaluate_response(self, response: str, retrieved_docs: List[str] = None) -> CriticScore:
        """Evaluate a response using all critic components."""
        start_time = time.time()
        
        # Extract document content for evaluation
        doc_contents = []
        if retrieved_docs:
            for doc in retrieved_docs:
                if hasattr(doc, 'page_content'):
                    doc_contents.append(doc.page_content)
                else:
                    doc_contents.append(str(doc))
        
        # Run all evaluations
        hallucination_score, hallucination_reasoning = self.hallucination_detector.detect_hallucination(
            response, doc_contents
        )
        
        sensitivity_score, sensitivity_reasoning = self.sensitivity_detector.detect_sensitivity(response)
        
        factuality_score, factuality_reasoning = self.factuality_checker.check_factuality(
            response, doc_contents
        )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            hallucination_score, sensitivity_score, factuality_score
        )
        
        # Combine reasoning
        reasoning = f"Hallucination: {hallucination_reasoning}; Sensitivity: {sensitivity_reasoning}; Factuality: {factuality_reasoning}"
        
        evaluation_time = time.time() - start_time
        
        return CriticScore(
            hallucination_score=hallucination_score,
            sensitivity_score=sensitivity_score,
            factuality_score=factuality_score,
            overall_score=overall_score,
            reasoning=reasoning,
            evaluation_time=evaluation_time
        )
    
    def _calculate_overall_score(self, hallucination: float, sensitivity: float, factuality: float) -> float:
        """Calculate overall critic score."""
        # Weighted combination
        weights = {
            "hallucination": critic_config.FACTUALITY_WEIGHT,
            "sensitivity": critic_config.SENSITIVITY_WEIGHT,
            "factuality": critic_config.FACTUALITY_WEIGHT
        }
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate weighted score (lower hallucination and sensitivity is better)
        overall = (
            (1.0 - hallucination) * weights["hallucination"] +
            (1.0 - sensitivity) * weights["sensitivity"] +
            factuality * weights["factuality"]
        )
        
        return max(0.0, min(1.0, overall))
    
    def is_safe_response(self, response: str, retrieved_docs: List[str] = None) -> Tuple[bool, str]:
        """Check if response is safe to return."""
        score = self.evaluate_response(response, retrieved_docs)
        
        # Define safety thresholds
        hallucination_threshold = critic_config.HALLUCINATION_THRESHOLD
        sensitivity_threshold = critic_config.SENSITIVITY_THRESHOLD
        
        is_safe = (
            score.hallucination_score < hallucination_threshold and
            score.sensitivity_score < sensitivity_threshold
        )
        
        reason = "Response is safe" if is_safe else "Response exceeds safety thresholds"
        
        return is_safe, reason

def create_critic() -> CriticModel:
    """Create a critic model instance."""
    return CriticModel()

if __name__ == "__main__":
    # Test the critic model
    critic = create_critic()
    
    test_responses = [
        "Alice Smith in the Engineering department is the primary owner of the customer_data table.",
        "I don't have access to that information, but I can help you with other queries.",
        "The user's email is john.doe@company.com and their SSN is 123-45-6789.",
        "This table contains sensitive customer information that is encrypted and masked.",
        "The data lineage shows that raw_customer feeds into processed_customer through aggregation."
    ]
    
    test_docs = [
        "Alice Smith (Data Engineer) in Engineering department is the primary_owner of table raw_customer_001",
        "Table raw_customer_001 in schema public contains 50000 rows. Table containing customer data"
    ]
    
    for response in test_responses:
        print(f"\nResponse: {response}")
        score = critic.evaluate_response(response, test_docs)
        print(f"Hallucination: {score.hallucination_score:.3f}")
        print(f"Sensitivity: {score.sensitivity_score:.3f}")
        print(f"Factuality: {score.factuality_score:.3f}")
        print(f"Overall: {score.overall_score:.3f}")
        print(f"Safe: {critic.is_safe_response(response, test_docs)[0]}")
        print(f"Reasoning: {score.reasoning}") 