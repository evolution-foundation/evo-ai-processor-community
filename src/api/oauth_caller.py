"""
Who starts and who completes an OAuth integration flow.

The OAuth state is signed together with the caller (see
src.services.oauth_state), so a flow started by one user, or in one account,
cannot be completed by another. The provider service dependencies are wrapped
with `binds_oauth_caller`, which records the caller of the current request on
the service they build.
"""

import functools
import inspect
from typing import Any, Callable, Dict

from fastapi import Request

from src.api.dependencies import get_user_identity
from src.evo_extension_points import runtime_context


def oauth_caller(request: Request) -> Dict[str, Any]:
    user_context = getattr(request.state, "user_context", None)
    return {
        "user": get_user_identity(user_context),
        "context": runtime_context.current_context_id(request) or "",
    }


def binds_oauth_caller(dependency: Callable) -> Callable:
    # functools.wraps keeps the dependency's own signature visible to FastAPI.
    @functools.wraps(dependency)
    async def wrapper(request: Request):
        service = dependency(request)
        if inspect.isawaitable(service):
            service = await service
        service.oauth_caller = oauth_caller(request)
        return service

    return wrapper
