from pydantic import BaseModel

class SearchResultItem(BaseModel):
    score: float
    document: str

class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[SearchResultItem]