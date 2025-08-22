"""
Unit tests for the retrieval system.
"""

import pytest
import json
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

# Import modules to test
from retrieve import DenseRetriever, SparseRetriever, MetadataRetriever, EnsembleRetriever, create_retriever

class TestDenseRetriever:
    """Test the dense retriever."""
    
    def test_init(self):
        """Test retriever initialization."""
        retriever = DenseRetriever()
        assert retriever.model_name is not None
        assert retriever.documents == []
    
    def test_add_documents(self):
        """Test adding documents."""
        retriever = DenseRetriever()
        documents = [
            {"content": "Test document 1", "metadata": {"type": "test"}},
            {"content": "Test document 2", "metadata": {"type": "test"}}
        ]
        
        retriever.add_documents(documents)
        assert len(retriever.documents) == 2
    
    @patch('retrieve.SentenceTransformer')
    def test_retrieve(self, mock_encoder):
        """Test document retrieval."""
        # Mock the encoder
        mock_encoder.return_value.encode.return_value = [[0.1, 0.2, 0.3]]
        
        retriever = DenseRetriever()
        documents = [
            {"content": "Test document 1", "metadata": {"type": "test"}},
            {"content": "Test document 2", "metadata": {"type": "test"}}
        ]
        retriever.add_documents(documents)
        
        # Mock FAISS search
        with patch('retrieve.faiss.IndexFlatIP') as mock_faiss:
            mock_faiss.return_value.search.return_value = ([[0.8, 0.6]], [[0, 1]])
            retriever.vectorstore = Mock()
            retriever.vectorstore.index = mock_faiss.return_value
            
            result = retriever.retrieve("test query")
            assert result.retriever_type == "dense"
            assert len(result.documents) > 0

class TestSparseRetriever:
    """Test the sparse retriever."""
    
    def test_init(self):
        """Test retriever initialization."""
        retriever = SparseRetriever()
        assert retriever.bm25 is None
        assert retriever.documents == []
    
    def test_add_documents(self):
        """Test adding documents."""
        retriever = SparseRetriever()
        documents = [
            {"content": "Test document 1", "metadata": {"type": "test"}},
            {"content": "Test document 2", "metadata": {"type": "test"}}
        ]
        
        retriever.add_documents(documents)
        assert len(retriever.documents) == 2
        assert retriever.bm25 is not None
    
    def test_retrieve(self):
        """Test document retrieval."""
        retriever = SparseRetriever()
        documents = [
            {"content": "Test document 1", "metadata": {"type": "test"}},
            {"content": "Test document 2", "metadata": {"type": "test"}}
        ]
        retriever.add_documents(documents)
        
        result = retriever.retrieve("test")
        assert result.retriever_type == "sparse"
        assert isinstance(result.scores, list)

class TestMetadataRetriever:
    """Test the metadata retriever."""
    
    def test_init(self):
        """Test retriever initialization."""
        retriever = MetadataRetriever()
        assert retriever.documents == []
        assert retriever.metadata_index == {}
    
    def test_add_documents(self):
        """Test adding documents."""
        retriever = MetadataRetriever()
        documents = [
            {"content": "Test document 1", "metadata": {"type": "table", "table_id": "table_001"}},
            {"content": "Test document 2", "metadata": {"type": "column", "table_id": "table_001"}}
        ]
        
        retriever.add_documents(documents)
        assert len(retriever.documents) == 2
        assert "table" in retriever.metadata_index
        assert "table_001" in retriever.metadata_index
    
    def test_retrieve(self):
        """Test document retrieval."""
        retriever = MetadataRetriever()
        documents = [
            {"content": "Test document 1", "metadata": {"type": "table", "table_name": "customer_data"}},
            {"content": "Test document 2", "metadata": {"type": "column", "table_name": "user_data"}}
        ]
        retriever.add_documents(documents)
        
        result = retriever.retrieve("customer_data table")
        assert result.retriever_type == "metadata"
        assert isinstance(result.scores, list)

class TestEnsembleRetriever:
    """Test the ensemble retriever."""
    
    def test_init(self):
        """Test retriever initialization."""
        weights = {"dense": 0.4, "sparse": 0.3, "metadata": 0.3}
        retriever = EnsembleRetriever(weights)
        assert retriever.weights == weights
        assert retriever.documents == []
    
    def test_add_documents(self):
        """Test adding documents to ensemble."""
        retriever = EnsembleRetriever()
        documents = [
            {"content": "Test document 1", "metadata": {"type": "test"}},
            {"content": "Test document 2", "metadata": {"type": "test"}}
        ]
        
        retriever.add_documents(documents)
        assert len(retriever.documents) == 2
    
    def test_update_weights(self):
        """Test updating retriever weights."""
        retriever = EnsembleRetriever()
        new_weights = {"dense": 0.5, "sparse": 0.25, "metadata": 0.25}
        
        retriever.update_weights(new_weights)
        assert retriever.weights == new_weights
    
    @patch('retrieve.DenseRetriever.retrieve')
    @patch('retrieve.SparseRetriever.retrieve')
    @patch('retrieve.MetadataRetriever.retrieve')
    def test_retrieve(self, mock_metadata, mock_sparse, mock_dense):
        """Test ensemble retrieval."""
        # Mock individual retriever results
        mock_dense.return_value = Mock(
            documents=[Mock(page_content="doc1")],
            scores=[0.8],
            retriever_type="dense",
            retrieval_time=0.1
        )
        mock_sparse.return_value = Mock(
            documents=[Mock(page_content="doc2")],
            scores=[0.6],
            retriever_type="sparse",
            retrieval_time=0.1
        )
        mock_metadata.return_value = Mock(
            documents=[Mock(page_content="doc3")],
            scores=[0.7],
            retriever_type="metadata",
            retrieval_time=0.1
        )
        
        retriever = EnsembleRetriever()
        documents = [
            {"content": "Test document 1", "metadata": {"type": "test"}},
            {"content": "Test document 2", "metadata": {"type": "test"}}
        ]
        retriever.add_documents(documents)
        
        result = retriever.retrieve("test query")
        assert result.documents is not None
        assert result.scores is not None
        assert result.total_time > 0

class TestCreateRetriever:
    """Test the create_retriever function."""
    
    @patch('retrieve.load_documents')
    def test_create_retriever(self, mock_load_docs):
        """Test creating a retriever."""
        mock_load_docs.return_value = [
            {"content": "Test document", "metadata": {"type": "test"}}
        ]
        
        retriever = create_retriever()
        assert isinstance(retriever, EnsembleRetriever)
    
    @patch('retrieve.load_documents')
    def test_create_retriever_with_weights(self, mock_load_docs):
        """Test creating a retriever with custom weights."""
        mock_load_docs.return_value = [
            {"content": "Test document", "metadata": {"type": "test"}}
        ]
        
        weights = {"dense": 0.5, "sparse": 0.25, "metadata": 0.25}
        retriever = create_retriever(weights)
        assert retriever.weights == weights

class TestLoadDocuments:
    """Test document loading functionality."""
    
    @patch('retrieve.Path')
    def test_load_documents_file_exists(self, mock_path):
        """Test loading documents when file exists."""
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_path.return_value.__truediv__.return_value = mock_file
        
        with patch('builtins.open', mock_open(read_data='[{"content": "test", "metadata": {}}]')):
            from retrieve import load_documents
            documents = load_documents()
            assert len(documents) == 1
    
    @patch('retrieve.Path')
    def test_load_documents_file_not_exists(self, mock_path):
        """Test loading documents when file doesn't exist."""
        mock_file = Mock()
        mock_file.exists.return_value = False
        mock_path.return_value.__truediv__.return_value = mock_file
        
        from retrieve import load_documents
        documents = load_documents()
        assert documents == []

if __name__ == "__main__":
    pytest.main([__file__]) 