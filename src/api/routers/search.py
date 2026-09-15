import structlog

from fastapi import APIRouter, Request, Query, HTTPException
from src.api.schemas.search import SearchResultItem, SearchResponse

from src.core.settings import settings

logger = structlog.get_logger()

router = APIRouter(prefix="/search", tags=["search"])

@router.get("", response_model=SearchResponse)
def search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=256),
    top_k: int = Query(default=settings.DEFAULT_TOP_K, ge=1, le=50)
):
    
    logger.debug("search_endpoint_called", query_text=q, top_k=top_k)

    engine = getattr(request.app.state, "search_engine", None)
    if engine is None:
        logger.error("search_engine_missing", reason="Engine not initialized in app.state")
        raise HTTPException(
            status_code=503,
            detail="Search engine not initialized"
        )
    
    try:
        results = engine.search(q, top_k)

    except Exception as e:
        logger.exception("search_engine_crashed", error_msg=str(e), query_text=q)
        raise HTTPException(500, "Internal search error")
    
    return SearchResponse(
        query=q,
        top_k=top_k,
        results=[SearchResultItem(score=score, document=doc) for (score, doc) in results]
    )