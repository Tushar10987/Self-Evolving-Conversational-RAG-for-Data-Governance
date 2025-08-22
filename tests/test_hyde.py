"""
Unit tests for the HyDE module.
"""

import pytest
from unittest.mock import Mock, patch

# Import modules to test
from hyde import HyDEGenerator, HyDERetriever, create_hyde_generator, create_hyde_retriever

class TestHyDEGenerator:
    """Test the HyDE generator."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = HyDEGenerator()
        assert generator.model_name is not None
        assert generator.device in ["cuda", "cpu"]
    
    @patch('hyde.AutoTokenizer.from_pretrained')
    @patch('hyde.AutoModelForCausalLM.from_pretrained')
    def test_load_model(self, mock_model, mock_tokenizer):
        """Test model loading."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        
        generator = HyDEGenerator()
        generator._load_model()
        
        assert generator.tokenizer is not None
        assert generator.model is not None
    
    def test_create_hypothesis_prompt(self):
        """Test hypothesis prompt creation."""
        generator = HyDEGenerator()
        query = "Who owns the customer_data table?"
        
        prompt = generator._create_hypothesis_prompt(query)
        
        assert isinstance(prompt, str)
        assert query in prompt
        assert len(prompt) > len(query)
    
    @patch('hyde.HyDEGenerator._generate_with_local_model')
    def test_generate_hypotheses(self, mock_generate):
        """Test hypothesis generation."""
        mock_generate.return_value = "Alice Smith is the owner of the customer_data table."
        
        generator = HyDEGenerator()
        query = "Who owns the customer_data table?"
        
        result = generator.generate_hypotheses(query, num_hypotheses=2)
        
        assert len(result.hypotheses) == 2
        assert result.generation_time > 0
        assert result.model_used is not None
    
    def test_enhance_query(self):
        """Test query enhancement."""
        generator = HyDEGenerator()
        query = "Who owns the customer_data table?"
        
        # Test with hypotheses enabled
        enhancement = generator.enhance_query(query, use_hypotheses=True)
        
        assert enhancement["original_query"] == query
        assert len(enhancement["enhanced_queries"]) >= 1
        assert "enhancement_weight" in enhancement
        
        # Test with hypotheses disabled
        enhancement = generator.enhance_query(query, use_hypotheses=False)
        
        assert enhancement["original_query"] == query
        assert len(enhancement["enhanced_queries"]) == 1
        assert enhancement["enhancement_weight"] == 0.0

class TestHyDERetriever:
    """Test the HyDE retriever."""
    
    def test_init(self):
        """Test retriever initialization."""
        base_retriever = Mock()
        hyde_retriever = HyDERetriever(base_retriever)
        
        assert hyde_retriever.base_retriever == base_retriever
        assert hyde_retriever.hyde_generator is not None
    
    @patch('hyde.HyDEGenerator.enhance_query')
    def test_retrieve_with_hyde(self, mock_enhance):
        """Test HyDE-enhanced retrieval."""
        # Mock enhancement result
        mock_enhance.return_value = {
            "original_query": "test query",
            "enhanced_queries": ["test query", "test query enhanced"],
            "hypotheses": ["enhanced hypothesis"],
            "enhancement_weight": 0.2
        }
        
        # Mock base retriever
        base_retriever = Mock()
        base_retriever.retrieve.return_value = Mock(
            documents=[Mock(page_content="test doc")],
            scores=[0.8],
            total_time=0.1
        )
        
        hyde_retriever = HyDERetriever(base_retriever)
        
        result = hyde_retriever.retrieve_with_hyde("test query")
        
        assert "query" in result
        assert "enhanced_queries" in result
        assert "hypotheses" in result
        assert "combined_documents" in result
        assert result["total_time"] > 0
    
    def test_combine_results(self):
        """Test result combination."""
        hyde_retriever = HyDERetriever(Mock())
        
        # Mock original result
        original_result = Mock()
        original_result.documents = [Mock(page_content="doc1")]
        original_result.scores = [0.8]
        
        # Mock enhanced results
        enhanced_result = Mock()
        enhanced_result.documents = [Mock(page_content="doc2")]
        enhanced_result.scores = [0.6]
        
        combined = hyde_retriever._combine_results(original_result, [enhanced_result])
        
        assert len(combined) > 0
        assert all(hasattr(doc, 'page_content') for doc in combined)

class TestCreateFunctions:
    """Test creation functions."""
    
    def test_create_hyde_generator(self):
        """Test creating HyDE generator."""
        generator = create_hyde_generator()
        assert isinstance(generator, HyDEGenerator)
    
    def test_create_hyde_retriever(self):
        """Test creating HyDE retriever."""
        base_retriever = Mock()
        hyde_retriever = create_hyde_retriever(base_retriever)
        assert isinstance(hyde_retriever, HyDERetriever)

if __name__ == "__main__":
    pytest.main([__file__]) 