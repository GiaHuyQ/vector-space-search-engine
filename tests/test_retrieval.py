import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.indexing import Indexer
from src.retrieval import SearchEngine
from src.retrieval.similarity import cosine_similarity_sparse


class TestCosineSimilarity:
    def test_known_values(self):
        a = csr_matrix([[1, 0]])
        b = csr_matrix([[1, 0], [0, 1], [1, 1]])
        np.testing.assert_allclose(
            cosine_similarity_sparse(a, b), [1.0, 0.0, 1 / np.sqrt(2)]
        )

    def test_empty_query_returns_zeros(self):
        a = csr_matrix((1, 2))
        b = csr_matrix([[1, 0], [0, 1]])
        assert cosine_similarity_sparse(a, b).tolist() == [0.0, 0.0]

    def test_empty_document_scores_zero(self):
        a = csr_matrix([[1, 0]])
        b = csr_matrix([[1, 0], [0, 0]])
        with np.errstate(invalid="ignore"):
            scores = cosine_similarity_sparse(a, b)
        assert np.isfinite(scores).all()


@pytest.fixture
def engine(index):
    return SearchEngine(index.corpus, index.vocabulary, index.doc_term_matrix)


class TestSearchEngine:
    def test_best_match_first(self, engine):
        results = engine.search("neural networks", top_k=3)
        assert results[0][1] == "Deep learning uses neural networks."

    def test_results_sorted_descending(self, engine):
        scores = [s for s, _ in engine.search("learning", top_k=4)]
        assert scores == sorted(scores, reverse=True)

    def test_only_positive_scores_returned(self, engine):
        results = engine.search("learning", top_k=4)
        assert len(results) == 2
        assert all(score > 0 for score, _ in results)

    def test_respects_top_k(self, engine):
        assert len(engine.search("learning", top_k=1)) == 1

    def test_top_k_larger_than_corpus(self, engine):
        assert len(engine.search("cat", top_k=100)) == 1

    def test_out_of_vocabulary_returns_empty(self, engine):
        assert engine.search("quantum xylophone", top_k=5) == []

    def test_stopword_only_query_returns_empty(self, engine):
        assert engine.search("the is of", top_k=5) == []

    def test_works_with_loaded_index(self, index, tmp_path):
        Indexer.save(index, tmp_path)
        loaded = Indexer.load(tmp_path)
        engine = SearchEngine(loaded.corpus, loaded.vocabulary, loaded.doc_term_matrix)
        assert engine.search("pasta", top_k=1)[0][1] == "Cooking pasta requires boiling water."
