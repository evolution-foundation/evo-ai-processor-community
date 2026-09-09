"""
┌──────────────────────────────────────────────────────────────────────────────┐
│ @author: Davidson Gomes                                                      │
│ @file: main.py                                                               │
│ Developed by: Davidson Gomes                                                 │
│ Creation date: May 13, 2025                                                  │
│ Contact: contato@evolution-api.com                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ @copyright © Evolution API 2025. All rights reserved.                        │
│ Licensed under the Apache License, Version 2.0                               │
│                                                                              │
│ You may not use this file except in compliance with the License.             │
│ You may obtain a copy of the License at                                      │
│                                                                              │
│    http://www.apache.org/licenses/LICENSE-2.0                                │
│                                                                              │
│ Unless required by applicable law or agreed to in writing, software          │
│ distributed under the License is distributed on an "AS IS" BASIS,            │
│ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     │
│ See the License for the specific language governing permissions and          │
│ limitations under the License.                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│ @important                                                                   │
│ For any future changes to the code in this file, it is recommended to        │
│ include, together with the modification, the information of the developer    │
│ who changed it and the date of modification.                                 │
└──────────────────────────────────────────────────────────────────────────────┘
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.config.database import engine, Base
from src.config.settings import settings
from src.utils.logger import setup_logger
from src.middleware.rate_limit import GlobalRateLimitMiddleware, RateLimitMiddleware

from src.services.otel_config import setup_otel
from src.services.permission_service import initialize_permission_service
from src.middleware.evo_auth import EvoAuthMiddleware
from src.middleware.tenant_context import TenantContextMiddleware
from fastapi.exceptions import HTTPException
from src.core.exceptions import BaseAPIException
from src.core.exception_handlers import (
    http_exception_handler,
    base_api_exception_handler
)

setup_result = setup_otel()
if setup_result:
    print("OpenTelemetry initialized successfully for Langfuse integration")
else:
    print(
        "Warning: OpenTelemetry could not be initialized - check Langfuse credentials"
    )

# Initialize permission middleware
initialize_permission_service(settings.EVO_AUTH_BASE_URL)

import src.api.client_routes
import src.api.chat_routes
import src.api.session_routes
import src.api.a2a_routes
import src.api.custom_mcp_servers_routes
import src.api.tools_routes
import src.api.system_routes
import src.api.google_calendar_routes
import src.api.google_sheets_routes
import src.api.github_routes
import src.api.notion_routes
import src.api.supabase_routes
import src.api.linear_routes
import src.api.monday_routes
import src.api.atlassian_routes
import src.api.asana_routes
import src.api.hubspot_routes
import src.api.paypal_routes
import src.api.canva_routes
import src.api.integrations_routes

# Add the root directory to PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Configure logger
logger = setup_logger(__name__)

# FastAPI initialization
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    redirect_slashes=False,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Register exception handlers for standardized error responses
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(BaseAPIException, base_api_exception_handler)


# Before EvoAuth on purpose: it must run after EvoAuth inbound.
app.add_middleware(TenantContextMiddleware)

# EvoAuth middleware for authentication and user context
app.add_middleware(
    EvoAuthMiddleware
)

# CORS - added after other middlewares so it wraps them (last added = first executed in Starlette)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
app.add_middleware(
    GlobalRateLimitMiddleware,
    requests_per_second=settings.RATE_LIMIT_GLOBAL_RPS,
    burst=settings.RATE_LIMIT_GLOBAL_BURST
)

app.add_middleware(
    RateLimitMiddleware,
    requests_per_second=settings.RATE_LIMIT_CLIENT_RPS,
    burst=settings.RATE_LIMIT_CLIENT_BURST,
    cleanup_interval=settings.RATE_LIMIT_CLEANUP_INTERVAL
)

# Static files configuration
static_dir = Path("static")
if not static_dir.exists():
    static_dir.mkdir(parents=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# PostgreSQL configuration
POSTGRES_CONNECTION_STRING = os.getenv(
    "POSTGRES_CONNECTION_STRING", "postgresql://postgres:root@localhost:5432/evo_ai"
)

# Create database tables — exclude tables owned by other services (auth/CRM)
# The `users` table is owned by evo-auth-service. Creating it here as a stub
# (id integer) makes auth's InitSchema skip the real creation via if_not_exists,
# permanently breaking authentication. See: incident multimport/2026-05-27.
_owned_by_other_services = {"users"}
_tables_to_create = [
    table for name, table in Base.metadata.tables.items()
    if name not in _owned_by_other_services
]
Base.metadata.create_all(bind=engine, tables=_tables_to_create, checkfirst=True)

API_PREFIX = "/api/v1"

# Define router references
client_router = src.api.client_routes.router
chat_router = src.api.chat_routes.router
session_router = src.api.session_routes.router
a2a_router = src.api.a2a_routes.router
custom_mcp_server_router = src.api.custom_mcp_servers_routes.router
tools_router = src.api.tools_routes.router
system_router = src.api.system_routes.router
google_calendar_router = src.api.google_calendar_routes.router
google_calendar_callback_router = src.api.google_calendar_routes.callback_router
google_sheets_router = src.api.google_sheets_routes.router
google_sheets_callback_router = src.api.google_sheets_routes.callback_router
github_router = src.api.github_routes.router
github_callback_router = src.api.github_routes.callback_router
notion_router = src.api.notion_routes.router
notion_callback_router = src.api.notion_routes.callback_router
supabase_router = src.api.supabase_routes.router
supabase_callback_router = src.api.supabase_routes.callback_router
linear_router = src.api.linear_routes.router
linear_callback_router = src.api.linear_routes.callback_router
monday_router = src.api.monday_routes.router
monday_callback_router = src.api.monday_routes.callback_router
atlassian_router = src.api.atlassian_routes.router
atlassian_callback_router = src.api.atlassian_routes.callback_router
asana_router = src.api.asana_routes.router
asana_callback_router = src.api.asana_routes.callback_router
hubspot_router = src.api.hubspot_routes.router
hubspot_callback_router = src.api.hubspot_routes.callback_router
paypal_router = src.api.paypal_routes.router
paypal_callback_router = src.api.paypal_routes.callback_router
canva_router = src.api.canva_routes.router
canva_callback_router = src.api.canva_routes.callback_router
integrations_router = src.api.integrations_routes.router

# Include routes
app.include_router(client_router, prefix=API_PREFIX)
app.include_router(custom_mcp_server_router, prefix=API_PREFIX)
app.include_router(chat_router, prefix=API_PREFIX)
app.include_router(session_router, prefix=API_PREFIX)
app.include_router(a2a_router, prefix=API_PREFIX)
app.include_router(tools_router, prefix=API_PREFIX)
app.include_router(google_calendar_router, prefix=API_PREFIX)
app.include_router(google_calendar_callback_router, prefix=API_PREFIX)
app.include_router(google_sheets_router, prefix=API_PREFIX)
app.include_router(google_sheets_callback_router, prefix=API_PREFIX)
app.include_router(github_router, prefix=API_PREFIX)
app.include_router(github_callback_router, prefix=API_PREFIX)
app.include_router(notion_router, prefix=API_PREFIX)
app.include_router(notion_callback_router, prefix=API_PREFIX)
app.include_router(supabase_router, prefix=API_PREFIX)
app.include_router(supabase_callback_router, prefix=API_PREFIX)
app.include_router(linear_router, prefix=API_PREFIX)
app.include_router(linear_callback_router, prefix=API_PREFIX)
app.include_router(monday_router, prefix=API_PREFIX)
app.include_router(monday_callback_router, prefix=API_PREFIX)
app.include_router(atlassian_router, prefix=API_PREFIX)
app.include_router(atlassian_callback_router, prefix=API_PREFIX)
app.include_router(asana_router, prefix=API_PREFIX)
app.include_router(asana_callback_router, prefix=API_PREFIX)
app.include_router(hubspot_router, prefix=API_PREFIX)
app.include_router(hubspot_callback_router, prefix=API_PREFIX)
app.include_router(paypal_router, prefix=API_PREFIX)
app.include_router(paypal_callback_router, prefix=API_PREFIX)
app.include_router(canva_router, prefix=API_PREFIX)
app.include_router(canva_callback_router, prefix=API_PREFIX)
app.include_router(integrations_router, prefix=API_PREFIX)

# System routes (health and ready) - without API prefix for Kubernetes compatibility
app.include_router(system_router)

# Root endpoint removed as requested - system endpoints (/health, /ready) serve as entry points


@app.get("/healthz")
def liveness_check():
    """Kubernetes liveness probe - simple check if app is running"""
    return {"status": "ok"}


@app.get("/readyz")
def readiness_check():
    """Kubernetes readiness probe - checks if app is ready to serve traffic"""
    from src.config.database import get_db
    from sqlalchemy import text
    
    checks = {
        "status": "ready",
        "database": "unknown",
        "redis": "unknown",
    }
    
    # Check database connection
    try:
        db = next(get_db())
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
        db.close()
    except Exception as e:
        checks["database"] = f"error: {str(e)}"
        checks["status"] = "not_ready"
    
    # Check Redis connection
    try:
        from src.config.redis import create_redis_pool
        pool = create_redis_pool()
        checks["redis"] = "ok"
        pool.disconnect()
    except Exception as e:
        checks["redis"] = f"error: {str(e)}"
        checks["status"] = "not_ready"
    
    return checks
