import time
import uuid
import structlog

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger()

class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        structlog.contextvars.clear_contextvars()

        request_id = str(uuid.uuid4())

        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            client_ip=request.client.host if request.client else "unknown",
            method=request.method,
            path=request.url.query,
        )

        start_time = time.perf_counter()

        logger.info("http_request_started")

        try:
            response = await call_next(request)
            status_code = response.status_code

        except Exception as e:
            status_code = 500
            logger.exception("http_request_crashed_internally", error_message=str(e))
            raise

        process_time = time.perf_counter() - start_time
        process_time_ms = round(process_time * 1000, 2)

        logger.info(
            "http_request_completed",
            status_code=status_code,
            process_time_ms=process_time_ms,
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(process_time_ms)

        return response