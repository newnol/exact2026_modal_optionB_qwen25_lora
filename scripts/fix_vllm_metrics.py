"""
Robust fix for vLLM prometheus_fastapi_instrumentator crash.

The middleware crashes on vLLM internal _IncludedRouter objects at:
  prometheus_fastapi_instrumentator/routing.py:_get_route_name -> route.path

Strategy:
  1. Find and replace the broken routing.py with a fixed version.
  2. Find and patch vllm's own instrumentator/metrics.py as a backup.
"""

from pathlib import Path
import sys


def _find_site_packages() -> Path:
    for p in sys.path:
        candidate = Path(p)
        if candidate.name == "site-packages":
            return candidate
    return Path("/usr/local/lib/python3.12/site-packages")


def patch_prometheus_routing(site: Path) -> bool:
    """Replace prometheus_fastapi_instrumentator.routing.get_route_name
    with a version that skips non-route objects (e.g. _IncludedRouter)."""
    target = site / "prometheus_fastapi_instrumentator" / "routing.py"
    if not target.exists():
        print(f"[fix] routing.py not found at {target}, skipping")
        return False

    patched_code = '''"""Patched routing module - skips _IncludedRouter objects."""
from typing import Any, Dict, List, Optional, Sequence, Tuple

from prometheus_fastapi_instrumentator.routing import _get_route_name as _original_get_route_name


def _get_route_name(scope: Dict[str, Any], routes: Sequence[Any]) -> Optional[str]:
    """Fallback-safe version that handles _IncludedRouter."""
    for route in routes:
        if not hasattr(route, "path"):
            continue
        try:
            result = _original_get_route_name(scope, [route])
            if result is not None:
                return result
        except (AttributeError, TypeError):
            continue
    return None


def get_route_name(request: Any) -> Optional[str]:
    from prometheus_fastapi_instrumentator.routing import get_route_name as _original
    try:
        return _original(request)
    except AttributeError:
        return None
'''
    target.write_text(patched_code, encoding="utf-8")
    print(f"[fix] Patched {target}")
    return True


def patch_vllm_metrics(site: Path) -> bool:
    candidates = [
        site / "vllm" / "entrypoints" / "serve" / "instrumentator" / "metrics.py",
        site / "vllm" / "entrypoints" / "openai" / "instrumentator.py",
    ]
    patched = False
    for target in candidates:
        if target.exists():
            backup = target.with_suffix(target.suffix + ".bak")
            if not backup.exists():
                target.rename(backup)
            content = target.read_text("utf-8")
            # Replace attach_router with safe version
            if "prometheus_fastapi_instrumentator" in content or "attach_router" in content:
                new_content = '''import prometheus_client
import regex as re
from fastapi import FastAPI, Response
from prometheus_client import make_asgi_app
from starlette.routing import Mount
from vllm.v1.metrics.prometheus import get_prometheus_registry


class PrometheusResponse(Response):
    media_type = prometheus_client.CONTENT_TYPE_LATEST


def attach_router(app: FastAPI):
    registry = get_prometheus_registry()
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)
'''
                target.write_text(new_content, encoding="utf-8")
                print(f"[fix] Patched {target}")
                patched = True
    return patched


def main() -> None:
    site = _find_site_packages()
    print(f"[fix] Site-packages: {site}")

    p1 = patch_prometheus_routing(site)
    p2 = patch_vllm_metrics(site)

    if p1:
        print("[fix] prometheus_fastapi_instrumentator routing.py patched")
    if p2:
        print("[fix] vLLM metrics file patched")
    if not p1 and not p2:
        print("[fix] WARNING: No patches applied; vLLM may still crash on /v1/completions")
    print("[fix] Done")


if __name__ == "__main__":
    main()
