"""
Unit tests for the critic module.
"""

import pytest
from unittest.mock import Mock, patch

# Import modules to test
from critic import (
    HallucinationDetector, 
    SensitivityDetector, 
    FactualityChecker, 
    CriticModel, 
    create_critic
)

class TestHallucinationDetector:
    """Test the hallucination detector."""
    
    def test_init(self):
        """Test detector initialization."""
        detector = HallucinationDetector()
        assert len(detector.sensitive_patterns) > 0
        assert len(detector.hallucination_indicators) > 0
        assert len(detector.confidence_indicators) > 0
    
    def test_detect_hallucination_safe_response(self):
        """Test detection with safe response."""
        detector = HallucinationDetector()
        response = "Alice Smith owns the customer_data table."
        docs = ["Alice Smith is the owner of customer_data table"]
        
        score, reasoning = detector.detect_hallucination(response, docs)
        
        assert 0 <= score <= 1
        assert isinstance(reasoning, str)
    
    def test_detect_hallucination_risky_response(self):
        """Test detection with risky response."""
        detector = HallucinationDetector()
        response = "I don't have access to that information, but I can help you with other queries."
        docs = []
        
        score, reasoning = detector.detect_hallucination(response, docs)
        
        assert score > 0  # Should detect uncertainty indicators
        assert "uncertainty" in reasoning.lower()
    
    def test_detect_hallucination_sensitive_data(self):
        """Test detection with sensitive data."""
        detector = HallucinationDetector()
        response = "The user's email is john.doe@company.com and their SSN is 123-45-6789."
        docs = []
        
        score, reasoning = detector.detect_hallucination(response, docs)
        
        assert score > 0  # Should detect sensitive patterns
        assert "sensitive" in reasoning.lower()
    
    def test_check_factual_consistency(self):
        """Test factual consistency check."""
        detector = HallucinationDetector()
        response = "Alice Smith owns the customer_data table"
        docs = ["Alice Smith is the owner of customer_data table"]
        
        consistency = detector._check_factual_consistency(response, docs)
        
        assert 0 <= consistency <= 1
        assert consistency > 0  # Should have some overlap
    
    def test_check_factual_consistency_no_docs(self):
        """Test factual consistency with no documents."""
        detector = HallucinationDetector()
        response = "Some response"
        docs = []
        
        consistency = detector._check_factual_consistency(response, docs)
        
        assert consistency == 0.0

class TestSensitivityDetector:
    """Test the sensitivity detector."""
    
    def test_init(self):
        """Test detector initialization."""
        detector = SensitivityDetector()
        assert len(detector.sensitive_keywords) > 0
        assert len(detector.data_governance_sensitive) > 0
    
    def test_detect_sensitivity_safe_response(self):
        """Test detection with safe response."""
        detector = SensitivityDetector()
        response = "Alice Smith owns the customer_data table."
        
        score, reasoning = detector.detect_sensitivity(response)
        
        assert 0 <= score <= 1
        assert isinstance(reasoning, str)
    
    def test_detect_sensitivity_sensitive_keywords(self):
        """Test detection with sensitive keywords."""
        detector = SensitivityDetector()
        response = "This table contains sensitive customer information that is encrypted."
        
        score, reasoning = detector.detect_sensitivity(response)
        
        assert score > 0  # Should detect sensitive keywords
        assert "sensitive" in reasoning.lower()
    
    def test_detect_sensitivity_data_patterns(self):
        """Test detection with data patterns."""
        detector = SensitivityDetector()
        response = "The user's email is john.doe@company.com"
        
        score, reasoning = detector.detect_sensitivity(response)
        
        assert score > 0  # Should detect email pattern
        assert "email" in reasoning.lower()
    
    def test_detect_sensitivity_governance_terms(self):
        """Test detection with governance terms."""
        detector = SensitivityDetector()
        response = "This data is masked and requires restricted access."
        
        score, reasoning = detector.detect_sensitivity(response)
        
        assert score > 0  # Should detect governance terms
        assert "governance" in reasoning.lower()

class TestFactualityChecker:
    """Test the factuality checker."""
    
    def test_init(self):
        """Test checker initialization."""
        checker = FactualityChecker()
        assert checker.encoder is not None
    
    @patch('critic.SentenceTransformer')
    def test_check_factuality(self, mock_encoder):
        """Test factuality checking."""
        # Mock encoder
        mock_encoder.return_value.encode.return_value = [[0.1, 0.2, 0.3]]
        
        checker = FactualityChecker()
        response = "Alice Smith owns the customer_data table"
        docs = ["Alice Smith is the owner of customer_data table"]
        
        score, reasoning = checker.check_factuality(response, docs)
        
        assert 0 <= score <= 1
        assert isinstance(reasoning, str)
    
    def test_check_factuality_no_docs(self):
        """Test factuality checking with no documents."""
        checker = FactualityChecker()
        response = "Some response"
        docs = []
        
        score, reasoning = checker.check_factuality(response, docs)
        
        assert score == 0.0
        assert "No retrieved documents" in reasoning

class TestCriticModel:
    """Test the main critic model."""
    
    def test_init(self):
        """Test model initialization."""
        critic = CriticModel()
        assert critic.hallucination_detector is not None
        assert critic.sensitivity_detector is not None
        assert critic.factuality_checker is not None
    
    def test_evaluate_response(self):
        """Test response evaluation."""
        critic = CriticModel()
        response = "Alice Smith owns the customer_data table"
        docs = ["Alice Smith is the owner of customer_data table"]
        
        score = critic.evaluate_response(response, docs)
        
        assert score.hallucination_score >= 0
        assert score.sensitivity_score >= 0
        assert score.factuality_score >= 0
        assert score.overall_score >= 0
        assert score.overall_score <= 1
        assert score.evaluation_time > 0
        assert isinstance(score.reasoning, str)
    
    def test_evaluate_response_no_docs(self):
        """Test response evaluation with no documents."""
        critic = CriticModel()
        response = "Some response"
        docs = []
        
        score = critic.evaluate_response(response, docs)
        
        assert score.hallucination_score >= 0
        assert score.sensitivity_score >= 0
        assert score.factuality_score >= 0
        assert score.overall_score >= 0
    
    def test_calculate_overall_score(self):
        """Test overall score calculation."""
        critic = CriticModel()
        
        # Test with good scores
        overall = critic._calculate_overall_score(0.1, 0.1, 0.9)
        assert 0 <= overall <= 1
        
        # Test with bad scores
        overall = critic._calculate_overall_score(0.9, 0.9, 0.1)
        assert 0 <= overall <= 1
    
    def test_is_safe_response(self):
        """Test safety check."""
        critic = CriticModel()
        response = "Alice Smith owns the customer_data table"
        docs = ["Alice Smith is the owner of customer_data table"]
        
        is_safe, reason = critic.is_safe_response(response, docs)
        
        assert isinstance(is_safe, bool)
        assert isinstance(reason, str)
    
    def test_is_safe_response_unsafe(self):
        """Test safety check with unsafe response."""
        critic = CriticModel()
        response = "The user's email is john.doe@company.com and their SSN is 123-45-6789."
        docs = []
        
        is_safe, reason = critic.is_safe_response(response, docs)
        
        # Should be unsafe due to sensitive data
        assert not is_safe
        assert "thresholds" in reason.lower()

class TestCreateFunctions:
    """Test creation functions."""
    
    def test_create_critic(self):
        """Test creating critic model."""
        critic = create_critic()
        assert isinstance(critic, CriticModel)

if __name__ == "__main__":
    pytest.main([__file__]) 