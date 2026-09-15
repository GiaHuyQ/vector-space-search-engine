from src.preprocessing.tokenizer import tokenize
from src.preprocessing.vectorizer import vectorize_sparse


class TestTokenize:
    def test_empty_text(self):
        assert tokenize("") == []

    def test_lowercases_and_stems(self):
        assert tokenize("Running RUNS") == ["run", "run"]

    def test_removes_stopwords(self):
        assert tokenize("this is the cat") == ["cat"]

    def test_removes_punctuation_and_emoji(self):
        assert tokenize("Hello, world!! 🚀😊") == ["hello", "world"]

    def test_only_stopwords_gives_empty(self):
        assert tokenize("the and of") == []


class TestVectorizeSparse:
    vocab_index = {"cat": 0, "dog": 1, "run": 2}

    def test_counts_known_tokens(self):
        vec = vectorize_sparse("cat cat running", self.vocab_index, 3)
        assert vec.shape == (1, 3)
        assert vec.toarray().tolist() == [[2, 0, 1]]

    def test_ignores_unknown_tokens(self):
        vec = vectorize_sparse("cat bird", self.vocab_index, 3)
        assert vec.toarray().tolist() == [[1, 0, 0]]

    def test_out_of_vocabulary_is_empty(self):
        vec = vectorize_sparse("bird fish", self.vocab_index, 3)
        assert vec.shape == (1, 3)
        assert vec.nnz == 0
