"""
FastAPI server for the self-evolving conversational RAG system.
Provides chat endpoints and telemetry logging.
"""

import json
import time
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from retrieve import create_retriever
from hyde import create_hyde_retriever
from critic import create_critic
from config import server_config, data_config, system_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Self-Evolving Conversational RAG",
    description="Data governance query system with automated experiments",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=server_config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    experiment_id: Optional[str] = None
    use_hyde: bool = True
    retriever_weights: Optional[Dict[str, float]] = None

class ChatResponse(BaseModel):
    response: str
    retrieved_docs: List[str]
    retrieval_time: float
    critic_score: float
    hallucination_score: float
    sensitivity_score: float
    factuality_score: float
    is_safe: bool
    request_id: str
    experiment_id: Optional[str]

class ExperimentResult(BaseModel):
    experiment_id: str
    strategy_name: str
    total_queries: int
    avg_retrieval_time: float
    avg_critic_score: float
    avg_precision_at_5: float
    avg_user_satisfaction: float

# Global instances
retriever = None
hyde_retriever = None
critic = None
telemetry_file = None

def initialize_components():
    """Initialize all system components."""
    global retriever, hyde_retriever, critic, telemetry_file
    
    logger.info("Initializing RAG system components...")
    
    # Create retrievers
    retriever = create_retriever()
    hyde_retriever = create_hyde_retriever(retriever)
    
    # Create critic
    critic = create_critic()
    
    # Setup telemetry file
    telemetry_file = Path(data_config.ARTIFACTS_DIR) / data_config.TELEMETRY_FILE
    telemetry_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("RAG system components initialized successfully")

def log_telemetry(request_id: str, request_data: Dict[str, Any], response_data: Dict[str, Any]):
    """Log telemetry data to NDJSON file."""
    try:
        telemetry_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "request": request_data,
            "response": response_data
        }
        
        with open(telemetry_file, 'a') as f:
            f.write(json.dumps(telemetry_entry) + '\n')
            
    except Exception as e:
        logger.error(f"Failed to log telemetry: {e}")

def generate_response(query: str, retrieved_docs: List) -> str:
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

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    initialize_components()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Self-Evolving Conversational RAG API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "retriever": retriever is not None,
            "hyde_retriever": hyde_retriever is not None,
            "critic": critic is not None
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Main chat endpoint for data governance queries."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(f"Processing chat request {request_id}: {request.query}")
        
        # Use custom weights if provided
        if request.retriever_weights:
            # Create temporary retriever with custom weights
            temp_retriever = create_retriever(request.retriever_weights)
            if request.use_hyde:
                temp_retriever = create_hyde_retriever(temp_retriever)
        else:
            temp_retriever = hyde_retriever if request.use_hyde else retriever
        
        # Retrieve documents
        retrieval_start = time.time()
        
        if request.use_hyde and hasattr(temp_retriever, 'retrieve_with_hyde'):
            retrieval_result = temp_retriever.retrieve_with_hyde(request.query)
            retrieved_docs = retrieval_result["combined_documents"]
            retrieval_time = retrieval_result["total_time"]
        else:
            retrieval_result = temp_retriever.retrieve(request.query)
            retrieved_docs = retrieval_result.documents
            retrieval_time = retrieval_result.total_time
        
        # Generate response
        response = generate_response(request.query, retrieved_docs)
        
        # Evaluate with critic
        critic_score = critic.evaluate_response(response, retrieved_docs)
        is_safe, safety_reason = critic.is_safe_response(response, retrieved_docs)
        
        total_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            "response": response,
            "retrieved_docs": [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in retrieved_docs],
            "retrieval_time": retrieval_time,
            "critic_score": critic_score.overall_score,
            "hallucination_score": critic_score.hallucination_score,
            "sensitivity_score": critic_score.sensitivity_score,
            "factuality_score": critic_score.factuality_score,
            "is_safe": is_safe,
            "request_id": request_id,
            "experiment_id": request.experiment_id
        }
        
        # Log telemetry in background
        background_tasks.add_task(
            log_telemetry,
            request_id,
            {
                "query": request.query,
                "experiment_id": request.experiment_id,
                "use_hyde": request.use_hyde,
                "retriever_weights": request.retriever_weights
            },
            response_data
        )
        
        logger.info(f"Chat request {request_id} completed in {total_time:.3f}s")
        
        return ChatResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error processing chat request {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/experiments/{experiment_id}/results")
async def get_experiment_results(experiment_id: str):
    """Get results for a specific experiment."""
    try:
        results_file = Path(data_config.ARTIFACTS_DIR) / data_config.EXPERIMENT_RESULTS_FILE
        
        if not results_file.exists():
            raise HTTPException(status_code=404, detail="No experiment results found")
        
        # Read CSV and filter by experiment_id
        import pandas as pd
        df = pd.read_csv(results_file)
        
        experiment_results = df[df['experiment_id'] == experiment_id]
        
        if experiment_results.empty:
            raise HTTPException(status_code=404, detail=f"No results found for experiment {experiment_id}")
        
        # Calculate summary statistics
        summary = {
            "experiment_id": experiment_id,
            "strategy_name": experiment_results['strategy_name'].iloc[0],
            "total_queries": len(experiment_results),
            "avg_retrieval_time": experiment_results['retrieval_time'].mean(),
            "avg_critic_score": experiment_results['critic_score'].mean(),
            "avg_precision_at_5": experiment_results['precision_at_5'].mean(),
            "avg_user_satisfaction": experiment_results['user_satisfaction'].mean(),
            "p95_retrieval_time": experiment_results['retrieval_time'].quantile(0.95),
            "results": experiment_results.to_dict('records')
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error retrieving experiment results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/experiments")
async def list_experiments():
    """List all available experiments."""
    try:
        results_file = Path(data_config.ARTIFACTS_DIR) / data_config.EXPERIMENT_RESULTS_FILE
        
        if not results_file.exists():
            return {"experiments": []}
        
        import pandas as pd
        df = pd.read_csv(results_file)
        
        # Get unique experiments
        experiments = df.groupby('experiment_id').agg({
            'strategy_name': 'first',
            'timestamp': 'first',
            'query': 'count'
        }).reset_index()
        
        experiments = experiments.rename(columns={'query': 'query_count'})
        
        return {"experiments": experiments.to_dict('records')}
        
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/telemetry")
async def get_telemetry(limit: int = 100):
    """Get recent telemetry data."""
    try:
        if not telemetry_file.exists():
            return {"telemetry": []}
        
        # Read last N lines from telemetry file
        with open(telemetry_file, 'r') as f:
            lines = f.readlines()
        
        # Parse last N entries
        telemetry_entries = []
        for line in lines[-limit:]:
            try:
                entry = json.loads(line.strip())
                telemetry_entries.append(entry)
            except json.JSONDecodeError:
                continue
        
        return {"telemetry": telemetry_entries}
        
    except Exception as e:
        logger.error(f"Error retrieving telemetry: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/experiment")
async def run_experiment():
    """Run a new experiment."""
    try:
        from experiment import ExperimentRunner
        
        runner = ExperimentRunner()
        results = runner.run_ab_experiment()
        
        return {
            "message": "Experiment completed successfully",
            "baseline_experiment_id": results['baseline'].experiment_id,
            "tuned_experiment_id": results['tuned'].experiment_id,
            "improvements": results['improvement']
        }
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=server_config.HOST,
        port=server_config.PORT,
        reload=system_config.DEBUG,
        log_level=server_config.LOG_LEVEL
    ) 