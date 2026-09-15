import structlog

from contextlib import asynccontextmanager
from fastapi import FastAPI
import pandas as pd

from src.indexing import Indexer
from src.retrieval import SearchEngine
from src.core.settings import settings

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("app_startup_started", message="System is starting up, checking resources...")

    index_files = [
        settings.INDEX_PATH / "doc_term_matrix.npz",
        settings.INDEX_PATH / "corpus.pkl",
        settings.INDEX_PATH / "vocabulary.pkl"
    ]

    if not all(file.exists() for file in index_files):
        logger.warning(
            "index_files_missing",
            message="Index files missing. Starting a new build process from Raw Data.",
            index_path=str(settings.INDEX_PATH),
        )

        if not settings.RAW_DATA_PATH.exists():
            logger.critical(
                "raw_data_not_found",
                message="Cannot start due to missing Raw Data!",
                raw_data_path=str(settings.RAW_DATA_PATH),
            )

            raise RuntimeError("RAW_DATA_PATH not found")
        
        try:
            logger.info("reading_parquet_started", file=str(settings.RAW_DATA_PATH))
            dataset = pd.read_parquet(settings.RAW_DATA_PATH)
            
            logger.info("dataset_loaded", total_rows=len(dataset))

            logger.info("building_index_started")
            index = Indexer.build(dataset)
            Indexer.save(index, settings.INDEX_PATH)

            logger.info("building_index_completed", message="New index saved successfully.")
        
        except Exception as e:
            logger.exception("building_index_crashed", error_message=str(e))
            raise
   
    else:
        logger.info("index_files_found", message="Exsisting index found, skipping build step.")

    try:
        logger.info("loading_index_into_memory")
        index = Indexer.load(settings.INDEX_PATH)

        logger.info(
            "index_loaded_success",
            vocab_size=len(index.vocabulary),
            corpus_size=len(index.corpus),
        )

    except Exception as e:
        logger.exception("loading_index_crashed", error_message=str(e))
        raise
    
    app.state.search_engine = SearchEngine(
        corpus=index.corpus,
        vocabulary=index.vocabulary,
        doc_term_matrix=index.doc_term_matrix,
    )

    logger.info("search_engine_initialized", message="Search Engine is ready to serve!")
    logger.info("app_startup_completed", status="READY")

    yield

    logger.info("app_shutdown_started", message="Received shutdown signal, cleaning up resources...")
    logger.info("app_shutdown_completed", message="Shutdown safely completed. Goodbye!")