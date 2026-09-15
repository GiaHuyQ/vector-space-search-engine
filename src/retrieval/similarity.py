import numpy as np
from scipy.sparse import csr_matrix, linalg

def cosine_similarity_sparse(a: csr_matrix, b: csr_matrix) -> np.ndarray:
    # Return 0 if either vector has no non-zero elements
    if a.nnz == 0 or b.nnz == 0:
        return np.zeros(b.shape[0])
    
    dot_product = a.dot(b.T).toarray().flatten()

    a_norm = linalg.norm(a)
    b_norm = linalg.norm(b, axis=1)
    denominator = a_norm * b_norm

    # Divide only where denominator != 0; leave the rest as 0
    scores = np.zeros_like(dot_product, dtype=float)
    np.divide(dot_product, denominator, out=scores, where=denominator != 0)
    return scores

if __name__ == '__main__':
    from scipy.sparse import csr_matrix

    vec_a = csr_matrix([1, 2, 1])
    vec_b = csr_matrix([[1, 1, 1], [1, 0, 1], [2, 0, 1]])

    print(cosine_similarity_sparse(vec_a, vec_b))