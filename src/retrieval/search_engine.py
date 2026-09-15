import structlog
import numpy as np
from scipy.sparse import csr_matrix

from src.preprocessing.vectorizer import vectorize_sparse
from .similarity import cosine_similarity_sparse

logger = structlog.get_logger()

class SearchEngine:
    def __init__(
        self,
        corpus: list[str],
        vocabulary: list[str],
        doc_term_matrix: csr_matrix,
    ):
        """
        Initialize search engine with pre-build index.
        """
        self.corpus = corpus
        self.vocabulary = vocabulary
        self.vocab_index = {w:i for i, w in enumerate(vocabulary)}
        self.vocab_size = len(vocabulary)
        self.doc_term_matrix = doc_term_matrix

    def _vectorize_query(self, query: str) -> csr_matrix:
        return vectorize_sparse(query, self.vocab_index, self.vocab_size)
    
    def _score(self, query_vec: csr_matrix, doc_matrix: csr_matrix) -> np.ndarray:
        return cosine_similarity_sparse(query_vec, doc_matrix)
    
    def search(self, query: str, top_k: int) -> list[tuple[float, str]]:
        """
        Search top-k most relevant documents.
        Return: [(score, document_text)]
        """
        logger.debug("search_started", query_text=query)
        query_vec = self._vectorize_query(query)

        if query_vec.nnz == 0:
            logger.warning(
                "query_out_of_vocabulary",
                query_text=query
            )
            return []  

        scores = self._score(query_vec, self.doc_term_matrix)
        
        k = min(top_k, len(scores))

        top_indices = np.argpartition(scores, - k)[- k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = [
            (float(scores[i]), self.corpus[i]) 
            for i in top_indices if scores[i] > 0
        ]

        logger.debug(
            "search_completed",
            results_found=len(results)
        )
        return results
