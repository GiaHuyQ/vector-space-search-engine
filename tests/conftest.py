import pandas as pd
import pytest

from src.indexing import Indexer


@pytest.fixture
def corpus() -> list[str]:
    return [
        "Machine learning is a field of artificial intelligence.",
        "Deep learning uses neural networks.",
        "The cat sat on the mat.",
        "Cooking pasta requires boiling water.",
    ]


@pytest.fixture
def dataset(corpus) -> pd.DataFrame:
    # Mimics the MS MARCO v1.1 layout: each row has passages.passage_text (a list).
    return pd.DataFrame({
        "passages": [
            {"passage_text": corpus[:2]},
            {"passage_text": corpus[2:]},
        ]
    })


@pytest.fixture
def index(dataset):
    return Indexer.build(dataset)
