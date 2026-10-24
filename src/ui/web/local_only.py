"""Answer only this machine's own page.

The server listens on loopback, but loopback alone does not stop a web page
from reaching it. A site can re-point its own domain at 127.0.0.1 (DNS
rebinding) and then read /api/runs, which holds report text, or teach the
learner a rule. Its requests then carry its own name in ``Host``. A page on
another site can also open the dictation socket, which carries that site in
``Origin``. So every request, HTTP or WebSocket, must name an allowed host in
``Host``, and in ``Origin`` when a browser sends one.

Allowed: loopback names, the configured ``web_host``, and ``web_allowed_hosts``
for anyone who serves the app to another machine on purpose.
"""

from __future__ import annotations

from typing import Iterable, Optional
from urllib.parse import urlsplit

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

LOOPBACK = ("127.0.0.1", "localhost", "::1")


def allowed_hosts(get) -> frozenset:
    """The host names requests may carry, from settings (*get* is ``Settings.get``)."""
    extra = get("web_allowed_hosts") or []
    return frozenset(h.lower() for h in (*LOOPBACK, str(get("web_host") or ""), *extra) if h)


def _host_of(value: Optional[str]) -> Optional[str]:
    """The bare host name in a ``Host`` value or an ``Origin`` URL, lower case."""
    if not value:
        return None
    parsed = urlsplit(value if "//" in value else f"//{value}")
    return (parsed.hostname or "").lower() or None


class LocalOnlyMiddleware:
    def __init__(self, app: ASGIApp, hosts: Iterable[str]) -> None:
        self.app = app
        self.hosts = frozenset(hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
            origin = headers.get("origin")
            refused = _host_of(headers.get("host")) not in self.hosts or (
                origin is not None and _host_of(origin) not in self.hosts
            )
            if refused:
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    await PlainTextResponse("Forbidden host", status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)
