# Self-Evolving Conversational RAG for Data Governance

A production-oriented RAG system for data-governance Q&A that ensembles dense, sparse, and metadata-aware retrievers, performs HyDE-style hypothesis expansion, and applies a critic to reduce hallucinations and flag sensitivity. It includes an offline experiment harness to tune retriever weights and prompt variants, plus observability and reproducible Docker workflows.

## Features

- **Ensembled Retrieval**: Dense (FAISS+SentenceTransformers), Sparse (BM25), and Metadata-graph retrievers with weighted late fusion.
- **HyDE Expansion**: Generates k synthetic hypotheses to expand queries and improve recall.
- **Critic Model**: Heuristic/ML critic scoring responses for hallucination risk, sensitivity exposure, and factuality.
- **Experiment Harness**: A/B runner for weights/prompt variants; logs Precision@k, critic-based factuality, simulated satisfaction, and latency.
- **Observability**: NDJSON telemetry per request; CSV/plots + static dashboard summarizing experiments.
- **Production Concerns**: Docker image, FastAPI server, generated synthetic dataset, configurable LLM backends.

## Tech Stack

- **Vector Store**: FAISS (CPU) with SentenceTransformers embeddings (default: `all-MiniLM-L6-v2`).
- **Sparse Retriever**: Rank-BM25 (Okapi) over normalized tokens.
- **Orchestration**: Minimal composition + LangChain vectorstore interface.
- **LLM**: Local HF model (configurable), or remote API via `config.py` toggles.
- **API**: FastAPI + Uvicorn.
- **Data**: Synthetic governance dataset generator (schemas, lineage, ownership). No real PII.

### Architecture Overview
- Query → optional HyDE hypotheses → retrievers (dense/sparse/metadata) → weighted fusion → top-k docs → lightweight answer synthesis → critic → response + telemetry.
- Metadata retriever indexes graph-like keys: `type`, `table_id`, `owner_id`, `owner_department`, enabling routing for governance intents (ownership, lineage, compliance).

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

# Run the complete demo (build → data → tests → experiments → server check → artifact)
./demo.sh
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data (documents.json, test_queries.json)
python scripts/generate_data.py

# Run tests (unit + integration)
pytest -q

# Start the API server
python server/main.py

# Run experiments (A/B; logs CSV + plots)
python experiment.py
```

## Project Structure

```
├── retrieve.py          # Dense(BERT)+FAISS, BM25, metadata retrievers + ensemble
├── hyde.py              # HyDE hypotheses generation and retrieval integration
├── critic.py            # Hallucination/factuality/sensitivity critic
├── experiment.py        # A/B runner, metrics, CSV outputs and plots
├── server/              # FastAPI service exposing /chat and experiment endpoints
├── scripts/             # Synthetic data generator
├── tests/               # Unit & integration tests; offline evaluation suite
├── dashboard/           # Static HTML dashboard for experiment summaries
├── artifacts/           # Telemetry (.ndjson), CSV, plots, experiment logs
├── Dockerfile           # Reproducible container
├── requirements.txt     # Dependencies (CPU-friendly)
└── demo.sh              # E2E demo/build/packaging
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

Additional endpoints:
- `GET /health` – component readiness and version
- `GET /experiments` – list aggregated experiment runs
- `GET /telemetry` – last N NDJSON entries

## Dashboard

After running experiments, view results at `dashboard/index.html`:
- Precision@k (5/10)
- Critic agreement and safety
- Latency percentiles
- A/B comparison charts

## Configuration

### LLM Configuration
Set in `config.py`:
- `LOCAL_MODEL` – HuggingFace model name for local inference
- `USE_LOCAL` – toggle local vs remote
- `REMOTE_MODEL`, `REMOTE_ENDPOINT`, `REMOTE_API_KEY` – optional remote config

### Retriever & HyDE
- `RetrieverConfig`: `DENSE_MODEL`, `*_TOP_K`, weights, `ENSEMBLE_TOP_K`, `MIN_SCORE_THRESHOLD`.
- `HyDEConfig`: enable/disable, `HYPOTHESIS_COUNT`, `HYPOTHESIS_LENGTH`, `ENHANCEMENT_WEIGHT`.

Tune weights and prompts via `experiment.py`.

## Evaluation Metrics

- **Precision@k (5/10)**: Fraction of retrieved docs matching gold per query.
- **Factuality (critic)**: Proxy grounding/safety via the critic.
- **User Satisfaction (simulated)**: Heuristic of relevance × safety.
- **Latency**: Mean and P95 for retrieval and E2E.

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