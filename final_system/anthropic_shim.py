"""Anthropic-native shim so Claude Code can run on an Azure AI Foundry resource.

Azure's `/anthropic/v1/messages` speaks the Messages API — the same wire format
Claude Code uses — but rejects the extras Claude Code adds for its own
features. Measured against the live resource, in order:

    ANTHROPIC_BASE_URL=<azure>/anthropic
    -> 400 Unexpected value(s) `advisor-tool-2026-03-01` for `anthropic-beta`
    (strip the header)
    -> 400 context_management: Extra inputs are not permitted
    (strip the body field)
    -> 200, and tool_use round-trips on claude-opus-5 and claude-fable-5.

Every rejection was an *addition* Claude Code makes, never something Azure
needs, so dropping them leaves a request Azure accepts and a response Claude
Code parses unchanged. That is why this is a shim and not a translator: no
field is rewritten, and the model's own output is passed through byte for byte,
streaming included.

The alternative reading — "Claude Code cannot use Azure" — is wrong, and
expensive: it would strand every Claude role on a subscription profile whose
quota is shared with everything else running that day.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: Hop-by-hop headers plus the ones Azure refuses. `anthropic-beta` announces
#: client features the gateway does not implement; it is advisory, so removing
#: it costs nothing on this path.
DROP_HEADERS = frozenset({
    "anthropic-beta", "host", "content-length", "connection",
    "accept-encoding", "transfer-encoding",
})

#: Body fields Claude Code sends that the gateway rejects as "extra inputs".
#: All are client-side features; none affect what the model is asked to do.
DROP_FIELDS = frozenset({
    "context_management", "mcp_servers", "container", "advisor",
})


def _handler(upstream: str, on_event) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # noqa: A003 - stdlib hook
            return

        def do_POST(self) -> None:  # noqa: N802 - stdlib hook
            length = int(self.headers.get("content-length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw) if raw else {}
                dropped = [key for key in DROP_FIELDS if key in payload]
                for key in dropped:
                    payload.pop(key, None)
                if dropped:
                    raw = json.dumps(payload).encode()
            except (ValueError, TypeError):
                dropped = []
            headers = {
                key: value for key, value in self.headers.items()
                if key.lower() not in DROP_HEADERS
            }
            request = urllib.request.Request(
                upstream + self.path, data=raw, headers=headers, method="POST"
            )
            try:
                response = urllib.request.urlopen(request, timeout=1800)
            except urllib.error.HTTPError as error:
                body = error.read()
                on_event(status=error.code, dropped=dropped,
                         detail=body[:400].decode("utf-8", "replace"))
                self._send(error.code, "application/json", body)
                return
            except Exception as exc:  # noqa: BLE001
                body = json.dumps({
                    "type": "error",
                    "error": {"type": "api_error", "message": str(exc)},
                }).encode()
                on_event(status=502, dropped=dropped, detail=str(exc)[:400])
                self._send(502, "application/json", body)
                return
            content_type = response.headers.get("content-type", "application/json")
            if "event-stream" in content_type:
                # Chunked passthrough: the SSE frames are the model's own output
                # and must not be buffered or reshaped.
                self.send_response(response.status)
                self.send_header("content-type", content_type)
                self.send_header("transfer-encoding", "chunked")
                self.end_headers()
                while chunk := response.read(8192):
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            else:
                self._send(response.status, content_type, response.read())

        def _send(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


class AnthropicShim:
    """Loopback shim; `base_url` is what agents get as ANTHROPIC_BASE_URL."""

    def __init__(self, upstream: str, *, on_event=None, host: str = "127.0.0.1"):
        self.upstream = upstream.rstrip("/")
        self.host = host
        self._on_event = on_event or (lambda **_: None)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("shim is not started")
        return f"http://{self.host}:{self._server.server_address[1]}"

    def start(self) -> str:
        self._server = ThreadingHTTPServer(
            (self.host, 0), _handler(self.upstream, self._on_event)
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.2},
            daemon=True, name="anthropic-shim",
        )
        self._thread.start()
        return self.base_url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
