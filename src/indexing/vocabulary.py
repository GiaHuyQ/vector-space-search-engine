import structlog

from src.preprocessing.tokenizer import tokenize

logger = structlog.get_logger()

def create_vocabulary(corpus: list[str]) -> list[str]:
    logger.info("create_vocabulary_started", input_corpus_size=len(corpus))

    vocabulary = set()

    for doc in corpus:
        vocabulary.update(tokenize(doc))
    
    logger.debug("sorting_vocabulary")
    final_vocabulary = sorted(vocabulary)

    logger.info("vocabulary_creation_completed", vocabulary_size=len(final_vocabulary))
    
    return final_vocabulary

if __name__ == "__main__":
    corpus = [
        "Machine learning is fun",
        "Deep learning is part of machine learning"
    ]

    vocab = create_vocabulary(corpus)
    print(vocab)