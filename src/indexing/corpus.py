import pandas as pd
import structlog

logger = structlog.get_logger()

def create_corpus(dataset: pd.DataFrame) -> list[str]:
    logger.info("create_corpus_started", input_dataframe_rows=len(dataset))

    corpus = []
    try: 
        for sample in dataset['passages']:
            passage_text = sample['passage_text']

            corpus.extend(passage_text)
    
    except KeyError as e:
        logger.exception("create_corpus_data_format_error", error_message=f"Missing expected key in data: {str(e)}")
        raise

    logger.info("create_corpus_completed", total_passages_extracted=len(corpus))
    
    return corpus