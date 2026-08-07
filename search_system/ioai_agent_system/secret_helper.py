"""Client for the in-memory API-key broker.

Claude Code invokes this file through its ``apiKeyHelper`` setting.  The
long-lived controller owns the secret; neither the helper command line nor the
Claude/Bash environment contains it.
"""

from __future__ import annotations

import os
import socket
import sys


def main() -> int:
    socket_path = os.environ.get("IOAI_SECRET_SOCKET")
    if not socket_path:
        print("IOAI_SECRET_SOCKET is not configured", file=sys.stderr)
        return 2
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(5)
        client.connect(socket_path)
        client.sendall(b"GET\n")
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    secret = b"".join(chunks).splitlines()[0].decode("utf-8")
    if not secret:
        print("secret broker returned an empty credential", file=sys.stderr)
        return 3
    sys.stdout.write(secret)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

