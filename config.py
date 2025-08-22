"""
Configuration settings for the self-evolving conversational RAG system.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class LLMConfig:
    """LLM configuration settings."""
    # Local model settings
    LOCAL_MODEL: str = "microsoft/DialoGPT-medium"  # Lightweight for POC
    LOCAL_MODEL_MAX_LENGTH: int = 512
    LOCAL_MODEL_TEMPERATURE: float = 0.7
    
    # Remote API settings (optional)
    REMOTE_API_KEY: str = ""
    REMOTE_ENDPOINT: str = ""
    REMOTE_MODEL: str = "gpt-3.5-turbo"
    
    # Use local model by default
    USE_LOCAL: bool = True

@dataclass
class RetrieverConfig:
    """Retriever configuration settings."""
    # Dense retriever
    DENSE_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DENSE_TOP_K: int = 10
    DENSE_WEIGHT: float = 0.4
    
    # Sparse retriever (BM25)
    SPARSE_TOP_K: int = 10
    SPARSE_WEIGHT: float = 0.3
    
    # Metadata retriever
    METADATA_TOP_K: int = 5
    METADATA_WEIGHT: float = 0.3
    
    # Ensemble settings
    ENSEMBLE_TOP_K: int = 15
    MIN_SCORE_THRESHOLD: float = 0.1

@dataclass
class HyDEConfig:
    """HyDE (Hypothesis-driven Document Enhancement) settings."""
    ENABLED: bool = True
    HYPOTHESIS_COUNT: int = 3
    HYPOTHESIS_LENGTH: int = 100
    ENHANCEMENT_WEIGHT: float = 0.2

@dataclass
class CriticConfig:
    """Critic model configuration."""
    ENABLED: bool = True
    HALLUCINATION_THRESHOLD: float = 0.7
    SENSITIVITY_THRESHOLD: float = 0.8
    FACTUALITY_WEIGHT: float = 0.6
    SENSITIVITY_WEIGHT: float = 0.4

@dataclass
class ExperimentConfig:
    """A/B experiment configuration."""
    BASELINE_WEIGHTS: Dict[str, float] = None
    TUNED_WEIGHTS: Dict[str, float] = None
    TEST_QUERIES_COUNT: int = 20
    EVALUATION_METRICS: list = None
    
    def __post_init__(self):
        if self.BASELINE_WEIGHTS is None:
            self.BASELINE_WEIGHTS = {
                "dense": 0.4,
                "sparse": 0.3,
                "metadata": 0.3
            }
        if self.TUNED_WEIGHTS is None:
            self.TUNED_WEIGHTS = {
                "dense": 0.5,
                "sparse": 0.25,
                "metadata": 0.25
            }
        if self.EVALUATION_METRICS is None:
            self.EVALUATION_METRICS = ["precision@5", "precision@10", "factuality", "satisfaction"]

@dataclass
class ServerConfig:
    """FastAPI server configuration."""
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    LOG_LEVEL: str = "info"
    CORS_ORIGINS: list = None
    
    def __post_init__(self):
        if self.CORS_ORIGINS is None:
            self.CORS_ORIGINS = ["*"]

@dataclass
class DataConfig:
    """Data generation and storage configuration."""
    SYNTHETIC_DATA_SIZE: int = 1000
    DATA_DIR: str = "data"
    ARTIFACTS_DIR: str = "artifacts"
    TELEMETRY_FILE: str = "telemetry.ndjson"
    EXPERIMENT_RESULTS_FILE: str = "experiment_results.csv"
    
    # Synthetic data generation
    TABLES_COUNT: int = 50
    COLUMNS_PER_TABLE: int = 10
    USERS_COUNT: int = 20
    DEPARTMENTS_COUNT: int = 8

@dataclass
class SystemConfig:
    """System-wide configuration."""
    # Load from environment variables
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Performance settings
    BATCH_SIZE: int = 32
    MAX_CONCURRENT_REQUESTS: int = 10
    
    # Timeouts
    REQUEST_TIMEOUT: int = 30
    RETRIEVAL_TIMEOUT: int = 10
    
    # File paths
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR: str = "cache"
    
    def __post_init__(self):
        # Create necessary directories
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(DataConfig.DATA_DIR, exist_ok=True)
        os.makedirs(DataConfig.ARTIFACTS_DIR, exist_ok=True)

# Global configuration instances
llm_config = LLMConfig()
retriever_config = RetrieverConfig()
hyde_config = HyDEConfig()
critic_config = CriticConfig()
experiment_config = ExperimentConfig()
server_config = ServerConfig()
data_config = DataConfig()
system_config = SystemConfig()

# Configuration validation
def validate_config() -> bool:
    """Validate configuration settings."""
    try:
        # Check retriever weights sum to 1.0
        total_weight = (
            retriever_config.DENSE_WEIGHT + 
            retriever_config.SPARSE_WEIGHT + 
            retriever_config.METADATA_WEIGHT
        )
        if abs(total_weight - 1.0) > 0.01:
            print(f"Warning: Retriever weights sum to {total_weight}, should be 1.0")
        
        # Check directories exist
        for directory in [data_config.DATA_DIR, data_config.ARTIFACTS_DIR, system_config.CACHE_DIR]:
            os.makedirs(directory, exist_ok=True)
        
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

# Initialize configuration
if __name__ == "__main__":
    validate_config() 