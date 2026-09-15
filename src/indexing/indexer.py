import structlog
import pickle
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from scipy.sparse import csr_matrix, save_npz, load_npz

from .corpus import create_corpus
from .vocabulary import create_vocabulary
from .doc_term_matrix import create_doc_term_matrix_sparse

logger = structlog.get_logger()

@dataclass
class Index:
    corpus: list[str]
    vocabulary: list[str]
    doc_term_matrix: csr_matrix

class Indexer:
    """
    Build, save and load vector-space search index.
    """
    @staticmethod
    def build(dataset: pd.DataFrame) -> Index:
        logger.info("indexer_build_started", dataframe_rows=len(dataset))

        corpus = create_corpus(dataset)
        vocabulary = create_vocabulary(corpus)
        doc_term_matrix = create_doc_term_matrix_sparse(corpus, vocabulary)
        
        logger.info(
            "indexer_build_completed",
            status="success",

        )

        return Index(
            corpus=corpus, 
            vocabulary=vocabulary, 
            doc_term_matrix=doc_term_matrix
        )
    
    @staticmethod
    def save(index: Index, path: str | Path):
        path = Path(path)

        logger.info("indexer_save_started", target_path=str(path))

        try:
            path.mkdir(parents=True, exist_ok=True)

            save_npz(path / "doc_term_matrix.npz", index.doc_term_matrix)

            with open(path / "corpus.pkl", "wb") as f:
                pickle.dump(index.corpus, f)

            with open(path / "vocabulary.pkl", "wb") as f:
                pickle.dump(index.vocabulary, f)

            logger.info("indexer_save_completed", target_path=str(path))
        
        except Exception as e:
            logger.exception(
                "indexer_save_failed", 
                target_path=str(path), 
                error_message=str(e)
            )
            raise

    @classmethod
    def load(cls, path: str | Path) -> Index:
        path = Path(path)

        logger.info("indexer_load_started", source_path=(str(path)))

        if not (path / "doc_term_matrix.npz").exists():
            logger.error("indexer_files_missing", source_path=str(path))
            raise FileNotFoundError(f"Index files not found in {path}")
        
        try: 
            doc_term_matrix = load_npz(path / "doc_term_matrix.npz")

            with open(path / "corpus.pkl", "rb") as f:
                corpus = pickle.load(f)

            with open(path / "vocabulary.pkl", "rb") as f:
                vocabulary = pickle.load(f)

            logger.info(
                "indexer_load_completed",
                source_path=str(path),
                corpus_size=len(corpus),
                vocabulary_size=len(vocabulary),
                sparse_matrix_size=doc_term_matrix.shape,
            )
            
            return Index(
                corpus=corpus,
                vocabulary=vocabulary,
                doc_term_matrix=doc_term_matrix
            )
        
        except Exception as e:
            logger.exception(
                "indexer_load_failed",
                source_path=str(path),
                error_message=str(e)
            )
            raise


        
    



    
