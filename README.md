# Vector Space Search Engine

A learning project for understanding bag-of-words and vector search. It is a hand-built search API over MS MARCO v1.1 passages.

## How it works

1. **Tokenize:** lowercase the text, strip emoji and punctuation, remove stopwords, then stem each token (NLTK Snowball).
2. **Index:** build a vocabulary and a sparse document-term matrix of raw term counts (SciPy CSR).
3. **Search:** vectorize the query the same way, score it against every passage with cosine similarity, and return the top-k.

On startup the app loads the index from `data/index/`. If the index isn't there, it first builds it from `data/raw/dataset.parquet`.

## Tech Stack

- Python 3.11
- FastAPI
- NumPy, SciPy
- NLTK
- pandas
- structlog
- Docker

## Setup

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements/dev.txt
```

Put the MS MARCO v1.1 data (Parquet, with a `passages.passage_text` column) at `data/raw/dataset.parquet`. The first run needs internet to download the NLTK stopwords.

Optional `.env` settings:

```env
RAW_DATA_PATH=data/raw/dataset.parquet
INDEX_PATH=data/index
DEFAULT_TOP_K=5
```

## Usage

```bash
uvicorn src.main:app --reload
# or with Docker
docker compose -f docker-compose.dev.yml up --build
```

The first startup builds the index and may take a while.

```bash
curl "http://localhost:8000/search?q=what+is+machine+learning&top_k=3"
```

```json
{
  "query": "what is machine learning",
  "top_k": 3,
  "results": [{ "score": 0.61, "document": "..." }]
}
```

API docs are at `http://localhost:8000/docs`.

## Tests

```bash
pytest
```

The tests use a small in-memory corpus, so they don't need the MS MARCO dataset.

## Limitations

- Scores use raw term counts. There is no TF-IDF or BM25.
- The whole index is loaded into memory, and every query is scored against every document.
- Only exact stemmed words match. There is no semantic search.
