import structlog
import numpy as np
from collections import Counter
from scipy.sparse import lil_matrix, csr_matrix

from src.preprocessing.tokenizer import tokenize

logger = structlog.get_logger()

def create_doc_term_matrix_sparse(
        corpus: list[str], 
        vocabulary: list[str]
    ) -> csr_matrix:

    logger.info(
        "create_sparse_matrix_started",
        corpus_size=len(corpus),
        vocabulary_size=len(vocabulary)
    )

    if not corpus or not vocabulary:
        logger.error(
            "sparse_matrix_creation_failed",
            reason="Empty corpus or vocabulary",
            corpus_size=len(corpus),
            vocabulary_size=len(vocabulary),
        )
        raise ValueError("Corpus and vocabulary must not be empty")
    
    vocab_index = {w: i for i, w in enumerate(vocabulary)}
    matrix = lil_matrix((len(corpus), len(vocabulary)), dtype=np.int32)

    for doc_id, doc in enumerate(corpus):
        tokens = tokenize(doc)
        for token, count in Counter(tokens).items():
            if token in vocab_index:
                matrix[doc_id, vocab_index[token]] = count

    logger.debug("converting_lil_to_csr_matrix")
    final_matrix = csr_matrix(matrix)

    logger.info(
        "create_sparse_matrix_completed",
        matrix_shape=final_matrix.shape,
        non_zero_elements=final_matrix.nnz,
    )

    return final_matrix

if __name__ == "__main__":
    corpus = ["apple banana apple", "banana orange"]
    vocabulary = sorted(set(
        token for doc in corpus for token in tokenize(doc)
    ))

    m = create_doc_term_matrix_sparse(corpus, vocabulary)
    print(m.toarray())