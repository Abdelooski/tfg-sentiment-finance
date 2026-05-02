# TFG — Sentiment Analysis in Financial Markets

Modular Python project that combines NLP-based sentiment analysis with financial data to study market behaviour.

## Project Structure

| Path | Purpose |
|---|---|
| `data/raw/` | Original, unmodified data (news, prices, filings) |
| `data/processed/` | Cleaned and transformed data ready for modelling |
| `notebooks/` | Exploratory analysis and experiment notebooks |
| `src/data/` | Data ingestion and loading utilities |
| `src/nlp/` | Sentiment models, tokenisation, text preprocessing |
| `src/features/` | Feature engineering (technical indicators, sentiment scores) |
| `src/models/` | Model definitions, training loops, evaluation |
| `src/utils/` | Shared helpers (logging, config, I/O) |
| `app/` | FastAPI application exposing the pipeline as an API |
| `reports/figures/` | Generated charts and visualisations |
| `reports/tables/` | Generated tables and metric summaries |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
```
