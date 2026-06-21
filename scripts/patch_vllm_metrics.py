from pathlib import Path


TARGET = Path("/usr/local/lib/python3.12/site-packages/vllm/entrypoints/serve/instrumentator/metrics.py")


PATCHED = """# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import prometheus_client
import regex as re
from fastapi import FastAPI, Response
from prometheus_client import make_asgi_app
from starlette.routing import Mount

from vllm.v1.metrics.prometheus import get_prometheus_registry


class PrometheusResponse(Response):
    media_type = prometheus_client.CONTENT_TYPE_LATEST


def attach_router(app: FastAPI):
    \"\"\"Mount prometheus metrics to a FastAPI app.

    This local patch intentionally skips prometheus_fastapi_instrumentator.
    In our deployment, that middleware crashes on vLLM's internal _IncludedRouter
    objects and turns /v1/models and /v1/chat/completions into HTTP 500s.
    We keep the explicit /metrics ASGI route only.
    \"\"\"

    registry = get_prometheus_registry()

    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)
"""


def main() -> None:
    TARGET.write_text(PATCHED, encoding="utf-8")
    print(f"Patched {TARGET}")


if __name__ == "__main__":
    main()
