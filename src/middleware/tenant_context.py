"""Binds the per-request context resolved by the ``runtime_context`` extension
point around the downstream app."""

from http import HTTPStatus

from starlette.exceptions import HTTPException
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse

from src.evo_extension_points import runtime_context
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TenantContextMiddleware:
    """Pure ASGI, not BaseHTTPMiddleware: that one runs the app in another task,
    where a ContextVar set here is invisible to the lazy DB transaction.
    Added before EvoAuthMiddleware in main.py (Starlette wraps in reverse) so it
    can read ``request.state.user_context``.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        try:
            request = StarletteRequest(scope, receive)
            cid = runtime_context.current_context_id(request)
        except HTTPException as e:
            # A refusal from the extension point is a response, not a failure.
            await self._error_response(e)(scope, receive, send)
            return
        except Exception as e:
            # Resolution failure degrades to "no context bound".
            logger.warning(f"TenantContextMiddleware: context resolution failed: {e}")
            cid = None

        if not cid:
            await self.app(scope, receive, send)
            return

        # Fail-open only on the bind setup: once the app has started, its
        # exceptions propagate (retrying here would run it twice).
        app_started = False
        try:
            async with runtime_context.bind_context(cid):
                app_started = True
                await self.app(scope, receive, send)
        except Exception as e:
            if app_started:
                raise
            logger.warning(f"TenantContextMiddleware: bind_context failed: {e}")
            await self.app(scope, receive, send)

    @staticmethod
    def _error_response(exc: HTTPException) -> JSONResponse:
        try:
            phrase = HTTPStatus(exc.status_code).phrase
        except ValueError:  # non-standard code such as 499
            phrase = "Error"
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": phrase,
                "code": "ERR_" + phrase.upper().replace(" ", "_"),
                "message": str(exc.detail),
            },
            headers=getattr(exc, "headers", None),
        )

