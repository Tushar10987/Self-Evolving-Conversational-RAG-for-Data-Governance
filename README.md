# Self‑Evolving Conversational RAG for Data Governance

![Build](https://img.shields.io/badge/build-github_actions-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A production‑grade prototype implementing an end‑to‑end Retrieval‑Augmented Generation (RAG) pipeline purpose‑built for data‑governance queries (ownership, lineage, schema, sensitivity). The system combines dense, sparse, and graph‑like metadata retrieval with HyDE‑style hypothesis generation, a safety/factuality critic, and an experiment runner that tunes retriever weights and prompts. It ships with observability, a minimal API, Docker, and an offline evaluation harness.

## Features

- **Multi-Modal Retrieval**: Dense (FAISS), sparse (BM25), and metadata-based retrievers with ensemble scoring
- **HyDE Integration**: Hypothesis-driven document enhancement for improved retrieval
- **Critic Model**: Hallucination detection and sensitivity scoring
- **A/B Experiment Engine**: Automated evaluation of retrieval strategies
- **Observability**: Comprehensive telemetry and dashboard
- **Production Ready**: Docker containerization and API endpoints

## Tech Stack

- **Vector Store**: FAISS for dense embeddings (SentenceTransformers mini‑LM by default)
- **Sparse Retriever**: Rank‑BM25 for classical token matching and term saturation control
- **Metadata/Graph**: Lightweight in‑memory indices over table/owner/lineage fields
- **Orchestration**: LangChain community components (retrievers, vector stores)
- **LLM**: Local HuggingFace models for HyDE and templated responses; configurable remote API
- **API**: FastAPI + Uvicorn, CORS, request‑scoped telemetry
- **Data**: Synthetic governance datasets (schemas, lineage, ownership, masking) — no PII

## Architecture Overview

The pipeline implements a multi‑stage retrieval and answer generation loop:

1. Query normalization and HyDE: generate N hypothesis answers from a small LLM; append to query for retrieval expansion.
2. Multi‑modal retrieval:
   - Dense: FAISS over SentenceTransformer embeddings, cosine similarity.
   - Sparse: BM25 over tokenized corpus (lowercase, whitespace tokenization).
   - Metadata: keyed indices over `type`, `table_id`, `owner_id`, `owner_department`, etc.
3. Ensemble scoring: weighted linear combination of per‑retriever scores with a minimum threshold; top‑k selection.
4. Answer synthesis: extractive/templated response from retrieved snippets (LLM configurable).
5. Critic: heuristic + model signals for hallucination, sensitivity exposure, and factuality.
6. Telemetry: per‑request `.ndjson` record with weights, doc ids, critic scores, latency.

The `experiment.py` module runs A/B experiments that vary retriever weights and/or HyDE prompt templates over a held‑out query set, computing Precision@k, critic‑based factuality, simulated satisfaction, and latency percentiles. Results are saved to CSV and summarized in a static dashboard.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- 8GB+ RAM (for local LLM models)

### Build and Run

```bash
# Clone and build
git clone git@github.com:Tushar10987/Self-Evolving-Conversational-RAG-for-Data-Governance.git
cd Self-Evolving-Conversational-RAG-for-Data-Governance

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

### Run with Docker
```bash
docker build -t rag-governance .
docker run --rm -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  rag-governance
# Check health
curl http://localhost:8000/health | jq
```

## Project Structure

```
├── retrieve.py           # Dense + sparse + metadata retrievers; ensemble logic
├── hyde.py               # HyDE generator: hypothesis answers for retrieval expansion
├── critic.py             # Hallucination/sensitivity/factuality critic and scores
├── experiment.py         # A/B engine; writes CSV and logs; artifact bundling
├── server/               # FastAPI app; chat + telemetry; experiment endpoints
├── tests/                # Unit + integration + offline evaluation suite
├── scripts/              # Synthetic data generator and utilities
├── dashboard/            # Static HTML dashboard (open locally)
├── artifacts/            # Telemetry + experiment outputs (created at runtime)
├── Dockerfile            # Build and run the service in a container
├── requirements.txt      # Reproducible environment
└── demo.sh               # One‑shot build + tests + experiments + artifact pack
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

Response shape:
```json
{
  "response": "...",
  "retrieved_docs": ["..."],
  "retrieval_time": 0.123,
  "critic_score": 0.82,
  "hallucination_score": 0.1,
  "sensitivity_score": 0.2,
  "factuality_score": 0.85,
  "is_safe": true,
  "request_id": "uuid",
  "experiment_id": "..."
}
```

### Experiment Results
```bash
GET /experiments/{experiment_id}/results
```

Also available:
```bash
POST /experiment   # runs a baseline vs tuned A/B and returns experiment ids
GET  /experiments  # lists experiments with counts and timestamps
GET  /telemetry    # returns recent telemetry ndjson entries
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

- **Precision@k**: Fraction of retrieved docs in top‑k that match ground‑truth labels
- **Factuality**: Critic‑estimated correctness vs. retrieved evidence
- **User Satisfaction**: Heuristic proxy combining answer coverage and brevity
- **Latency**: End‑to‑end time; P50/P95 breakdown

## Artifacts

The system generates:
- `.ndjson` telemetry logs per request (weights, top‑k ids, critic scores, timings)
- `experiment_results.csv` per experiment (query‑level metrics and timestamps)
- `dashboard/` static HTML; open `dashboard/index.html` after experiments
- `artifact.tar.gz` with source, logs, data samples, and dashboard for portability

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size in `config.py`
2. **Slow Retrieval**: Adjust FAISS index parameters
3. **Poor Results**: Tune retriever weights in experiments

### Windows/HF Cache Notes
- On Windows you may see HuggingFace cache warnings about symlinks; functionality remains unaffected.
- If model downloads are rate‑limited (HTTP 429), re‑run tests or set a HF token in the environment.

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