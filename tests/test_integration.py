"""
Integration tests for the complete RAG pipeline.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock

# Import modules to test
from retrieve import create_retriever
from hyde import create_hyde_generator
from critic import create_critic
from experiment import ExperimentRunner

class TestRAGPipeline:
    """Test the complete RAG pipeline integration."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for testing."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir()
        
        # Create sample documents
        documents = [
            {
                "id": "doc1",
                "content": "Alice Smith (Data Engineer) in Engineering department is the primary_owner of table customer_data",
                "metadata": {
                    "type": "ownership",
                    "owner_name": "Alice Smith",
                    "owner_department": "Engineering",
                    "table_name": "customer_data"
                }
            },
            {
                "id": "doc2", 
                "content": "Table customer_data in schema public contains 50000 rows. Table containing customer information",
                "metadata": {
                    "type": "table",
                    "table_name": "customer_data",
                    "row_count": 50000
                }
            },
            {
                "id": "doc3",
                "content": "Table raw_customer feeds into customer_data through aggregation transformation",
                "metadata": {
                    "type": "lineage",
                    "source_table": "raw_customer",
                    "target_table": "customer_data",
                    "transformation_type": "aggregation"
                }
            }
        ]
        
        with open(data_dir / "documents.json", 'w') as f:
            json.dump(documents, f)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @patch('config.data_config.DATA_DIR')
    def test_retriever_integration(self, mock_data_dir, temp_data_dir):
        """Test retriever integration with sample data."""
        mock_data_dir.__str__ = lambda: str(Path(temp_data_dir) / "data")
        
        # Create retriever
        retriever = create_retriever()
        
        # Test retrieval
        result = retriever.retrieve("Who owns the customer_data table?")
        
        assert result.documents is not None
        assert len(result.documents) > 0
        assert result.scores is not None
        assert result.total_time > 0
    
    def test_hyde_integration(self):
        """Test HyDE integration."""
        # Create HyDE generator
        hyde_gen = create_hyde_generator()
        
        # Test hypothesis generation
        query = "Who owns the customer_data table?"
        enhancement = hyde_gen.enhance_query(query)
        
        assert enhancement["original_query"] == query
        assert len(enhancement["enhanced_queries"]) > 1
        assert "hypotheses" in enhancement
    
    def test_critic_integration(self):
        """Test critic integration."""
        # Create critic
        critic = create_critic()
        
        # Test response evaluation
        response = "Alice Smith in the Engineering department is the primary owner of the customer_data table."
        retrieved_docs = ["Alice Smith (Data Engineer) in Engineering department is the primary_owner of table customer_data"]
        
        score = critic.evaluate_response(response, retrieved_docs)
        
        assert score.hallucination_score >= 0
        assert score.sensitivity_score >= 0
        assert score.factuality_score >= 0
        assert score.overall_score >= 0
        assert score.overall_score <= 1
    
    @patch('config.data_config.DATA_DIR')
    def test_experiment_integration(self, mock_data_dir, temp_data_dir):
        """Test experiment integration."""
        mock_data_dir.__str__ = lambda: str(Path(temp_data_dir) / "data")
        
        # Create experiment runner
        runner = ExperimentRunner()
        
        # Test with a small number of queries
        with patch.object(runner, 'test_queries', ["Who owns the customer_data table?"]):
            results = runner.run_experiment("test", {"dense": 0.4, "sparse": 0.3, "metadata": 0.3})
            
            assert len(results) > 0
            assert all(hasattr(result, 'experiment_id') for result in results)
            assert all(hasattr(result, 'critic_score') for result in results)

class TestEndToEndPipeline:
    """Test end-to-end pipeline functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return {
            "query": "Who owns the customer_data table?",
            "expected_response": "Alice Smith",
            "documents": [
                "Alice Smith (Data Engineer) in Engineering department is the primary_owner of table customer_data",
                "Table customer_data in schema public contains 50000 rows"
            ]
        }
    
    def test_full_pipeline(self, sample_data):
        """Test the complete pipeline from query to response."""
        # Create components
        retriever = create_retriever()
        critic = create_critic()
        
        # Mock document loading
        with patch('retrieve.load_documents') as mock_load:
            mock_load.return_value = [
                {
                    "content": sample_data["documents"][0],
                    "metadata": {"type": "ownership"}
                },
                {
                    "content": sample_data["documents"][1], 
                    "metadata": {"type": "table"}
                }
            ]
            
            # Test retrieval
            result = retriever.retrieve(sample_data["query"])
            
            # Test response generation
            response = "Alice Smith in the Engineering department is the primary owner of the customer_data table."
            
            # Test critic evaluation
            score = critic.evaluate_response(response, sample_data["documents"])
            
            # Assertions
            assert result.documents is not None
            assert len(result.documents) > 0
            assert score.overall_score > 0
            assert "Alice Smith" in response

class TestErrorHandling:
    """Test error handling in the pipeline."""
    
    def test_retriever_no_documents(self):
        """Test retriever behavior with no documents."""
        retriever = create_retriever()
        
        # Mock empty document loading
        with patch('retrieve.load_documents', return_value=[]):
            result = retriever.retrieve("test query")
            
            assert result.documents == []
            assert result.scores == []
    
    def test_critic_empty_response(self):
        """Test critic behavior with empty response."""
        critic = create_critic()
        
        score = critic.evaluate_response("", [])
        
        assert score.hallucination_score >= 0
        assert score.sensitivity_score >= 0
        assert score.factuality_score >= 0
    
    def test_experiment_invalid_weights(self):
        """Test experiment with invalid weights."""
        runner = ExperimentRunner()
        
        # Test with weights that don't sum to 1
        invalid_weights = {"dense": 0.5, "sparse": 0.5, "metadata": 0.5}
        
        with patch.object(runner, 'test_queries', ["test query"]):
            results = runner.run_experiment("test", invalid_weights)
            
            # Should still work but may have warnings
            assert len(results) > 0

class TestPerformance:
    """Test performance characteristics."""
    
    def test_retrieval_performance(self):
        """Test retrieval performance with multiple queries."""
        retriever = create_retriever()
        
        queries = [
            "Who owns the customer_data table?",
            "What tables contain sensitive information?",
            "Show me data lineage for financial data"
        ]
        
        total_time = 0
        for query in queries:
            result = retriever.retrieve(query)
            total_time += result.total_time
        
        # Should complete in reasonable time
        assert total_time < 10.0  # 10 seconds for 3 queries
    
    def test_critic_performance(self):
        """Test critic performance."""
        critic = create_critic()
        
        response = "Alice Smith in the Engineering department is the primary owner of the customer_data table."
        docs = ["Alice Smith (Data Engineer) in Engineering department is the primary_owner of table customer_data"]
        
        import time
        start_time = time.time()
        
        for _ in range(10):
            score = critic.evaluate_response(response, docs)
        
        total_time = time.time() - start_time
        
        # Should complete 10 evaluations in reasonable time
        assert total_time < 5.0  # 5 seconds for 10 evaluations

class TestConfiguration:
    """Test configuration handling."""
    
    def test_retriever_weights_configuration(self):
        """Test different retriever weight configurations."""
        weights_configs = [
            {"dense": 0.4, "sparse": 0.3, "metadata": 0.3},
            {"dense": 0.5, "sparse": 0.25, "metadata": 0.25},
            {"dense": 0.6, "sparse": 0.2, "metadata": 0.2}
        ]
        
        for weights in weights_configs:
            retriever = create_retriever(weights)
            assert retriever.weights == weights
    
    def test_critic_thresholds(self):
        """Test critic threshold configurations."""
        critic = create_critic()
        
        # Test with different response types
        safe_response = "Alice Smith owns the customer_data table."
        risky_response = "The user's email is john.doe@company.com and their SSN is 123-45-6789."
        
        safe_score = critic.evaluate_response(safe_response, [])
        risky_score = critic.evaluate_response(risky_response, [])
        
        # Risky response should have higher sensitivity score
        assert risky_score.sensitivity_score > safe_score.sensitivity_score

if __name__ == "__main__":
    pytest.main([__file__]) 