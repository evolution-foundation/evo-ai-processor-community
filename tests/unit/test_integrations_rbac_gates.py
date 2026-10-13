"""EVO-1956 — Contract test for RBAC enforcement on integrations routes.

Every route reachable on the provider router (`router`) MUST carry a
`RequirePermission("integrations", <verb>)` FastAPI dependency. That includes
the OAuth callback: the provider redirects the browser to the frontend, which
completes the flow through the authenticated POST .../callback. No provider
exposes a second, unauthenticated router.

Verifying at the router level (not via HTTP) means we don't need to boot
the middleware stack or mock auth service, and we catch regressions where
someone adds a new endpoint but forgets the gate.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi import APIRouter
from fastapi.params import Depends

from src.middleware.permissions import PermissionChecker


PROVIDER_MODULES = [
    "src.api.github_routes",
    "src.api.google_calendar_routes",
    "src.api.google_sheets_routes",
    "src.api.notion_routes",
    "src.api.linear_routes",
    "src.api.monday_routes",
    "src.api.atlassian_routes",
    "src.api.asana_routes",
    "src.api.hubspot_routes",
    "src.api.paypal_routes",
    "src.api.canva_routes",
    "src.api.supabase_routes",
]

# Must match `integrations` resource actions in
# evo-auth-service `app/models/resource_actions_config.rb`. Actions NOT in
# this set are unseeded and 403 everyone, admin included.
ALLOWED_ACTIONS = {"read", "create", "update", "delete", "connect", "disconnect"}


def _iter_permission_checkers(route) -> list[PermissionChecker]:
    """Walk a route's endpoint signature and dependant tree, collect
    every PermissionChecker instance wired via Depends()."""
    found: list[PermissionChecker] = []
    seen: set[int] = set()

    def walk(dep):
        if dep is None or id(dep) in seen:
            return
        seen.add(id(dep))
        call = getattr(dep, "call", None) or getattr(dep, "dependency", None)
        if isinstance(call, PermissionChecker):
            found.append(call)
        for sub in getattr(dep, "dependencies", []) or []:
            walk(sub)

    # 1. Explicit `dependencies=[...]` on the route (if any)
    for dep in getattr(route, "dependencies", []) or []:
        if isinstance(dep, Depends):
            call = dep.dependency
            if isinstance(call, PermissionChecker):
                found.append(call)

    # 2. Depends() declared in the endpoint signature — FastAPI stores these
    #    on the compiled `Dependant` tree at route.dependant.
    dependant = getattr(route, "dependant", None)
    if dependant is not None:
        walk(dependant)

    return found


def _load_routers():
    """Return {provider_name: main_router}."""
    out = {}
    for mod_path in PROVIDER_MODULES:
        mod = importlib.import_module(mod_path)
        provider = mod_path.rsplit(".", 1)[-1].removesuffix("_routes")
        main = getattr(mod, "router", None)
        assert isinstance(main, APIRouter), f"{mod_path} missing `router` attribute"
        out[provider] = main
    return out


@pytest.mark.parametrize("provider_module", PROVIDER_MODULES)
def test_every_main_router_endpoint_has_integrations_gate(provider_module):
    """Every @router.<verb> handler on a provider MUST carry a
    RequirePermission("integrations", <verb>) FastAPI dependency.
    Regression guard for EVO-1956."""
    mod = importlib.import_module(provider_module)
    router = getattr(mod, "router")

    ungated: list[str] = []
    wrong_resource: list[str] = []
    unknown_action: list[str] = []

    for route in router.routes:
        checkers = _iter_permission_checkers(route)
        label = f"{list(getattr(route, 'methods', {'?'}))} {getattr(route, 'path', route.name)}"
        if not checkers:
            ungated.append(label)
            continue
        for chk in checkers:
            if chk.resource != "integrations":
                wrong_resource.append(f"{label} -> resource={chk.resource!r}")
            if chk.action not in ALLOWED_ACTIONS:
                unknown_action.append(f"{label} -> action={chk.action!r}")

    assert not ungated, (
        f"{provider_module}: found {len(ungated)} main-router route(s) "
        f"without RequirePermission gate — EVO-1956 regression:\n  - " +
        "\n  - ".join(ungated)
    )
    assert not wrong_resource, (
        f"{provider_module}: main-router routes with wrong resource:\n  - " +
        "\n  - ".join(wrong_resource)
    )
    assert not unknown_action, (
        f"{provider_module}: main-router routes with unrecognized action:\n  - " +
        "\n  - ".join(unknown_action)
    )


@pytest.mark.parametrize("provider_module", PROVIDER_MODULES)
def test_provider_has_no_unauthenticated_callback_router(provider_module):
    """The GET callbacks read the agent and the MCP endpoint from the OAuth
    state and ran outside RequirePermission. They are gone; this keeps them
    from coming back."""
    mod = importlib.import_module(provider_module)

    assert not hasattr(mod, "callback_router")


@pytest.mark.parametrize("provider_module", PROVIDER_MODULES)
def test_oauth_callback_is_the_gated_post(provider_module):
    mod = importlib.import_module(provider_module)
    callbacks = [r for r in mod.router.routes if getattr(r, "path", "").endswith("/callback")]

    assert [sorted(r.methods) for r in callbacks] == [["POST"]]
    assert [(c.resource, c.action) for c in _iter_permission_checkers(callbacks[0])] == [("integrations", "connect")]


def test_bulk_integrations_route_is_gated():
    """`GET /agents/{agent_id}/integrations` (integrations_routes.py bulk
    endpoint) is admin-scoped and must carry RequirePermission."""
    from src.api import integrations_routes

    for route in integrations_routes.router.routes:
        checkers = _iter_permission_checkers(route)
        assert checkers, (
            f"integrations_routes: {getattr(route, 'path', route.name)} "
            "missing RequirePermission — EVO-1956 regression"
        )
        assert any(c.resource == "integrations" for c in checkers), (
            f"integrations_routes: {getattr(route, 'path', route.name)} "
            "gated by wrong resource"
        )
