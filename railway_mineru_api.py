"""Launch mineru-api with the three defaults a public CPU deploy needs.

Upstream's ``mineru-api`` is designed for a GPU box on a trusted network. Two of
its defaults are wrong for a one-click Railway deploy and cannot be changed with
an environment variable, so they are changed here, before the FastAPI app is
imported:

1. ``DEFAULT_BACKEND`` is ``hybrid-engine``, which needs a local VLM and a GPU.
   On a CPU container every request that omits ``backend=`` fails -- including
   the "Try it out" button in the service's own Swagger UI. It is a module
   constant (``mineru/cli/backend_options.py``), read by ``api_request`` through
   a ``from ... import``, so it has to be rebound before that import happens.

2. There is no authentication of any kind: the only middleware upstream installs
   is GZip. A template hands the deployer a public HTTPS URL, so the parsing
   endpoints are gated behind ``MINERU_API_KEY`` when it is set.

Health and documentation routes stay open so Railway's healthcheck and the
deployer's browser keep working.
"""

import hmac
import os
import sys

# --- 1. pipeline as the server default -------------------------------------
# Must precede any import of mineru.cli.fast_api / mineru.cli.api_request.
import mineru.cli.backend_options as backend_options

backend_options.DEFAULT_BACKEND = backend_options.BACKEND_PIPELINE

from mineru.cli.fast_api import app, main  # noqa: E402


# --- 2. optional API-key gate ----------------------------------------------

OPEN_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"})
OPEN_PREFIXES = ("/docs", "/redoc", "/openapi")


class ApiKeyGate:
    """Plain ASGI middleware, deliberately not BaseHTTPMiddleware.

    ``/file_parse`` takes multipart uploads that can be hundreds of megabytes;
    BaseHTTPMiddleware reads the request body into memory to hand it on, which
    would turn every large upload into a memory spike. This one only inspects
    the headers in ``scope`` and then passes the untouched receive channel
    through.
    """

    def __init__(self, app, key: str):
        self.app = app
        self.key = key

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in OPEN_PATHS or path.startswith(OPEN_PREFIXES):
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v for k, v in scope.get("headers", [])}
        supplied = headers.get("x-api-key", b"").decode("latin-1").strip()
        if not supplied:
            auth = headers.get("authorization", b"").decode("latin-1").strip()
            if auth.lower().startswith("bearer "):
                supplied = auth[7:].strip()

        if not hmac.compare_digest(supplied, self.key):
            from starlette.responses import JSONResponse

            response = JSONResponse(
                {
                    "detail": (
                        "Missing or invalid API key. Send it as "
                        "'Authorization: Bearer <MINERU_API_KEY>' or "
                        "'X-API-Key: <MINERU_API_KEY>'."
                    )
                },
                status_code=401,
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


def _install_api_key_gate() -> None:
    key = os.environ.get("MINERU_API_KEY", "").strip()
    if not key:
        print(
            "[railway] MINERU_API_KEY is empty -- the parsing API is OPEN to anyone "
            "with the URL.",
            file=sys.stderr,
        )
        return
    app.add_middleware(ApiKeyGate, key=key)
    print("[railway] API key gate enabled (/health and /docs stay public).")


if __name__ == "__main__":
    print(
        f"[railway] default backend = {backend_options.DEFAULT_BACKEND} "
        f"(upstream ships {backend_options.BACKEND_HYBRID_ENGINE}, which needs a GPU)"
    )
    _install_api_key_gate()
    main()
