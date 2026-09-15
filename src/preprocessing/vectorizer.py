from collections import Counter

from scipy.sparse import csr_matrix

from .tokenizer import tokenize

def vectorize_sparse(text: str, vocab_index: dict, vocab_size: int) -> csr_matrix:
    counter = Counter(tokenize(text))
    found = [(vocab_index[token], count) for token, count in counter.items() if token in vocab_index]
    if not found:
        return csr_matrix((1, vocab_size))
    
    indices, data = zip(*found)

    return csr_matrix(
        (data, indices, [0, len(indices)]),
        shape=(1, vocab_size)
    )

if __name__ == '__main__':
    sample_vocabulary = ['study', 'machine', 'learning', 'deep', 'robotics', 'computer', 'vision']
    sample_text = 'I am studying machine learning.'

    normalized_vocab = ["".join(tokenize(w)) for w in sample_vocabulary]

    vocab_index = {w: idx for idx, w in enumerate(normalized_vocab)}
    vocab_size = len(normalized_vocab)

    query_vector = vectorize_sparse(sample_text, vocab_index, vocab_size)
    
    print(f"Shape: {query_vector.shape}")
    print(f"Non-zero elements: {query_vector.nnz}")
    print(f"Data: {query_vector.data}")
    print(f"Full vector: {query_vector.toarray()}")