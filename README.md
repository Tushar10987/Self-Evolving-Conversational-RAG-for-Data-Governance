# Self-Evolving Conversational RAG for Data Governance

A production-ready prototype implementing an end-to-end RAG pipeline for data governance queries with automated experiment tuning for retriever weights and prompt templates.

## Features

- **Multi-Modal Retrieval**: Dense (FAISS), sparse (BM25), and metadata-based retrievers with ensemble scoring
- **HyDE Integration**: Hypothesis-driven document enhancement for improved retrieval
- **Critic Model**: Hallucination detection and sensitivity scoring
- **A/B Experiment Engine**: Automated evaluation of retrieval strategies
- **Observability**: Comprehensive telemetry and dashboard
- **Production Ready**: Docker containerization and API endpoints

## Tech Stack

- **Vector Store**: FAISS for dense embeddings
- **Sparse Retriever**: BM25 implementation
- **Orchestration**: LangChain for pipeline management
- **LLM**: Configurable (local HuggingFace models or remote APIs)
- **API**: FastAPI for production endpoints
- **Data**: Synthetic governance datasets (no real PII)

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- 8GB+ RAM (for local LLM models)

### Build and Run

```bash
# Clone and build
git clone <repo-url>
cd self-evolving-conversational-rag

# Build Docker image
docker build -t rag-governance .

# Run the complete demo
./demo.sh
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python scripts/generate_data.py

# Run tests
pytest tests/

# Start the API server
python server/main.py

# Run experiments
python experiment.py
```

## Project Structure

```
├── retrieve.py          # Multi-modal retrieval system
├── hyde.py             # Hypothesis-driven enhancement
├── critic.py           # Hallucination and sensitivity critic
├── experiment.py       # A/B testing engine
├── server/             # FastAPI server
├── tests/              # Unit tests and evaluation suite
├── scripts/            # Data generation and utilities
├── dashboard/          # Static HTML dashboard
├── artifacts/          # Generated logs and results
├── Dockerfile          # Container configuration
├── requirements.txt    # Python dependencies
└── demo.sh            # Complete demo script
```

## API Endpoints

### Chat Query
```bash
POST /chat
{
  "query": "Who owns the customer_data table?",
  "experiment_id": "optional_experiment_id"
}
```

### Experiment Results
```bash
GET /experiments/{experiment_id}/results
```

## Dashboard

After running experiments, view results at `dashboard/index.html`:
- Precision@k metrics
- Critic agreement rates
- Latency percentiles
- A/B comparison charts

## Configuration

### LLM Configuration
Set in `config.py`:
- `LOCAL_MODEL`: HuggingFace model name for local inference
- `REMOTE_API_KEY`: API key for remote LLM services
- `REMOTE_ENDPOINT`: Remote API endpoint URL

### Retriever Weights
Tune in `experiment.py`:
- Dense retriever weight
- Sparse retriever weight  
- Metadata retriever weight

## Evaluation Metrics

- **Precision@k**: Top-k retrieval accuracy
- **Factuality**: Critic model agreement
- **User Satisfaction**: Simulated user feedback
- **Latency**: P95 response times

## Artifacts

The system generates:
- `.ndjson` telemetry logs per request
- `experiment_results.csv` for A/B testing
- `dashboard/` with interactive visualizations
- `artifact.tar.gz` containing all outputs

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size in `config.py`
2. **Slow Retrieval**: Adjust FAISS index parameters
3. **Poor Results**: Tune retriever weights in experiments

### Logs

Check `artifacts/` directory for:
- `telemetry.ndjson`: Request-level logs
- `experiment_*.log`: Experiment execution logs
- `server.log`: API server logs

## Contributing

1. Run tests: `pytest tests/`
2. Follow PEP 8 style guide
3. Add tests for new features
4. Update documentation

## License

MIT License - see LICENSE file for details. 