import pytest
from fastapi.testclient import TestClient

from src.core.settings import settings
from src.indexing import Indexer
from src.main import app


@pytest.fixture
def client(index, tmp_path, monkeypatch):
    """App with a small pre-built index, so startup skips the real dataset."""
    Indexer.save(index, tmp_path)
    monkeypatch.setattr(settings, "INDEX_PATH", tmp_path)
    with TestClient(app) as c:
        yield c


def test_search_ok(client):
    res = client.get("/search", params={"q": "neural networks", "top_k": 2})
    assert res.status_code == 200

    body = res.json()
    assert body["query"] == "neural networks"
    assert body["top_k"] == 2
    assert body["results"][0]["document"] == "Deep learning uses neural networks."
    assert "X-Request-ID" in res.headers
    assert "X-Process-Time-Ms" in res.headers


def test_search_default_top_k(client):
    res = client.get("/search", params={"q": "learning"})
    assert res.json()["top_k"] == settings.DEFAULT_TOP_K


def test_search_no_match_returns_empty_list(client):
    res = client.get("/search", params={"q": "xylophone"})
    assert res.status_code == 200
    assert res.json()["results"] == []


@pytest.mark.parametrize("params", [
    {},                              # missing q
    {"q": ""},                       # too short
    {"q": "a" * 257},                # too long
    {"q": "cat", "top_k": 0},        # below min
    {"q": "cat", "top_k": 51},       # above max
])
def test_search_invalid_params(client, params):
    assert client.get("/search", params=params).status_code == 422


def test_search_engine_missing_returns_503(client):
    del client.app.state.search_engine
    res = client.get("/search", params={"q": "cat"})
    assert res.status_code == 503


def test_startup_builds_index_from_raw_data(dataset, tmp_path, monkeypatch):
    raw = tmp_path / "dataset.parquet"
    dataset.to_parquet(raw)
    index_dir = tmp_path / "index"

    monkeypatch.setattr(settings, "RAW_DATA_PATH", raw)
    monkeypatch.setattr(settings, "INDEX_PATH", index_dir)

    with TestClient(app) as c:
        assert (index_dir / "doc_term_matrix.npz").exists()
        assert c.get("/search", params={"q": "pasta"}).json()["results"]


def test_startup_fails_without_index_or_raw_data(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RAW_DATA_PATH", tmp_path / "missing.parquet")
    monkeypatch.setattr(settings, "INDEX_PATH", tmp_path / "index")

    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass
