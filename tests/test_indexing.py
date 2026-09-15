import pandas as pd
import pytest

from src.indexing import Indexer
from src.indexing.corpus import create_corpus
from src.indexing.doc_term_matrix import create_doc_term_matrix_sparse
from src.indexing.vocabulary import create_vocabulary


def test_create_corpus_flattens_passages(dataset, corpus):
    assert create_corpus(dataset) == corpus


def test_create_corpus_missing_key_raises():
    bad = pd.DataFrame({"passages": [{"text": ["a"]}]})
    with pytest.raises(KeyError):
        create_corpus(bad)


def test_create_vocabulary_is_sorted_and_unique():
    vocab = create_vocabulary(["cat dog", "dog cat bird"])
    assert vocab == ["bird", "cat", "dog"]


def test_doc_term_matrix_counts():
    corpus = ["apple banana apple", "banana orange"]
    vocab = ["appl", "banana", "orang"]  # stemmed forms
    matrix = create_doc_term_matrix_sparse(corpus, vocab)
    assert matrix.shape == (2, 3)
    assert matrix.toarray().tolist() == [[2, 1, 0], [0, 1, 1]]


@pytest.mark.parametrize("corpus, vocab", [([], ["a"]), (["a"], [])])
def test_doc_term_matrix_empty_input_raises(corpus, vocab):
    with pytest.raises(ValueError):
        create_doc_term_matrix_sparse(corpus, vocab)


def test_build_shapes_match(index, corpus):
    assert index.corpus == corpus
    assert index.doc_term_matrix.shape == (len(corpus), len(index.vocabulary))


def test_save_and_load_roundtrip(index, tmp_path):
    Indexer.save(index, tmp_path / "idx")
    loaded = Indexer.load(tmp_path / "idx")

    assert loaded.corpus == index.corpus
    assert loaded.vocabulary == index.vocabulary
    assert (loaded.doc_term_matrix != index.doc_term_matrix).nnz == 0


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Indexer.load(tmp_path)
