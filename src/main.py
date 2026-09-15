import uvicorn
from fastapi import FastAPI

from src.api.lifespan import lifespan
from src.core.middlewares import StructuredLoggingMiddleware
from src.api.routers.search import router as search_router


app = FastAPI(
    title="Vector Space Search Engine",
    lifespan=lifespan
)

app.add_middleware(StructuredLoggingMiddleware)

app.include_router(search_router)

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)