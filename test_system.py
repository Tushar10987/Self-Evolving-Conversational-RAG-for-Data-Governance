#!/usr/bin/env python3
"""
Simple test script to verify the RAG system components work.
"""

import os
import sys
import json
from pathlib import Path

def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    try:
        from config import validate_config
        result = validate_config()
        print(f"✓ Configuration validation: {result}")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_data_generation():
    """Test data generation."""
    print("Testing data generation...")
    try:
        # Create data directory
        os.makedirs("data", exist_ok=True)
        
        # Create simple test data
        test_documents = [
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
            }
        ]
        
        with open("data/documents.json", "w") as f:
            json.dump(test_documents, f, indent=2)
        
        print("✓ Test data created")
        return True
    except Exception as e:
        print(f"✗ Data generation error: {e}")
        return False

def test_retriever():
    """Test retriever functionality."""
    print("Testing retriever...")
    try:
        from retrieve import create_retriever
        
        # Create retriever with test data
        retriever = create_retriever()
        
        # Test retrieval
        result = retriever.retrieve("Who owns the customer_data table?")
        
        print(f"✓ Retriever created and tested")
        print(f"  - Retrieved {len(result.documents)} documents")
        print(f"  - Retrieval time: {result.total_time:.3f}s")
        return True
    except Exception as e:
        print(f"✗ Retriever error: {e}")
        return False

def test_critic():
    """Test critic functionality."""
    print("Testing critic...")
    try:
        from critic import create_critic
        
        critic = create_critic()
        
        # Test evaluation
        response = "Alice Smith owns the customer_data table."
        docs = ["Alice Smith (Data Engineer) in Engineering department is the primary_owner of table customer_data"]
        
        score = critic.evaluate_response(response, docs)
        
        print(f"✓ Critic created and tested")
        print(f"  - Overall score: {score.overall_score:.3f}")
        print(f"  - Hallucination: {score.hallucination_score:.3f}")
        print(f"  - Sensitivity: {score.sensitivity_score:.3f}")
        return True
    except Exception as e:
        print(f"✗ Critic error: {e}")
        return False

def test_experiment():
    """Test experiment functionality."""
    print("Testing experiment...")
    try:
        from experiment import ExperimentRunner
        
        # Create experiment runner
        runner = ExperimentRunner()
        
        # Test with minimal queries
        test_queries = ["Who owns the customer_data table?"]
        
        # Mock the test queries
        runner.test_queries = test_queries
        
        # Run experiment
        results = runner.run_experiment("test", {"dense": 0.4, "sparse": 0.3, "metadata": 0.3})
        
        print(f"✓ Experiment runner created and tested")
        print(f"  - Generated {len(results)} results")
        return True
    except Exception as e:
        print(f"✗ Experiment error: {e}")
        return False

def test_server():
    """Test server functionality."""
    print("Testing server...")
    try:
        from server.main import app
        
        # Test app creation
        assert app is not None
        print("✓ FastAPI app created successfully")
        return True
    except Exception as e:
        print(f"✗ Server error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Self-Evolving Conversational RAG System")
    print("=" * 50)
    
    tests = [
        test_config,
        test_data_generation,
        test_retriever,
        test_critic,
        test_experiment,
        test_server
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 