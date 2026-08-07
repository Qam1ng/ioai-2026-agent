"""Static safety checks against the Kaggle scoring-kernel environment.

Two hard constraints from ``docs/QA-NOTES.md`` govern the submitted artifact:

1. The kernel runs against a **fixed package list** — no ``pip install``.
2. The kernel runs with **internet disabled**.

Both failures are *silent until submission*: the kernel dies at import time or
at the first network call, the run is scored as an error, and one of the 50
submissions per problem is gone.  Locally the same code works fine, because
the agent's own machine has internet and whatever packages it felt like
installing.  That asymmetry is exactly what this module exists to close: it
reads a candidate kernel's source and answers "would this have died on
Kaggle?" without spending anything.

Everything here is a *static* check on the AST plus string literals.  It is
deliberately conservative in one direction only: it can produce false alarms
(a violation string that turns out to be harmless), but it should not stay
silent about a real network call or a real missing import.  Callers are
expected to surface violations, not to hard-refuse on them — see
``swarm/mirror/dryrun.py``, which records them and runs the kernel anyway.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

#: The pinned environment we check against.  Currently the IOAI-2025 list — see
#: the header inside the file itself for why, and for the replace-me note.
PINNED_ENV_PATH = Path(__file__).with_name("kaggle_env.txt")

#: Import name -> distribution name, for the cases where they differ.
#:
#: Without this map every ``import sklearn`` looks like a violation, because
#: the requirements file spells it ``scikit-learn``.  A wrong entry here turns
#: into a false "missing package" alarm, so the map is explicit and hand-checked
#: rather than derived from installed metadata (the agent's local site-packages
#: is NOT the Kaggle environment and must never be consulted).
IMPORT_ALIASES: dict[str, str] = {
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "magic": "python-magic",
    "Levenshtein": "python-Levenshtein",
    "jwt": "PyJWT",
    "OpenSSL": "pyOpenSSL",
    "Crypto": "pycryptodome",
    "Cryptodome": "pycryptodomex",
    "serial": "pyserial",
    "google": "protobuf",  # google.protobuf / google.colab-style namespace
    "attr": "attrs",
    "attrs": "attrs",
    "IPython": "ipython",
    "mpl_toolkits": "matplotlib",
    "pkg_resources": "setuptools",
    "setuptools": "setuptools",
    "fitz": "PyMuPDF",
    "pytorch_lightning": "pytorch-lightning",
    "lightning": "lightning",
    "torch_geometric": "torch-geometric",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "kaggle_environments": "kaggle-environments",
    "kaggle_secrets": "kaggle",
    "kaggle_web_client": "kaggle",
    "kagglehub": "kagglehub",
    "huggingface_hub": "huggingface-hub",
    "hf_transfer": "hf-transfer",
    "sentence_transformers": "sentence-transformers",
    "sksurv": "scikit-survival",
    "sentencepiece": "sentencepiece",
    "pydantic_core": "pydantic-core",
    "ruamel": "ruamel.yaml",
    "zoneinfo": "backports.zoneinfo",
    "memory_profiler": "memory-profiler",
    "usb": "pyusb",
    "wandb": "wandb",
    "cupy": "cupy-cuda12x",
    "MySQLdb": "mysqlclient",
    "psycopg2": "psycopg2-binary",
    "win32com": "pywin32",
    "gi": "PyGObject",
    "cairo": "pycairo",
    "_cffi_backend": "cffi",
}

#: Modules that live in the kernel image but are not pip packages and not in
#: ``sys.stdlib_module_names``.  Treating these as violations is pure noise.
ALWAYS_ALLOWED: frozenset[str] = frozenset(
    {
        "__future__",
        "__main__",
        "builtins",
        "kaggle_secrets",
        "kaggle_web_client",
        "kaggle_datasets",
    }
)

# --------------------------------------------------------------------- network

#: Modules whose *presence* in a scoring kernel means a network call.
#: ``urllib`` is deliberately absent: ``urllib.parse`` is offline-safe and
#: common, so urllib is matched at attribute level instead (see below).
_NETWORK_MODULES: frozenset[str] = frozenset(
    {
        "requests",
        "urllib3",
        "httpx",
        "aiohttp",
        "http.client",
        "httplib2",
        "ftplib",
        "smtplib",
        "telnetlib",
        "socket",
        "socketserver",
        "xmlrpc",
        "boto3",
        "botocore",
        "gdown",
        "wget",
        "kagglehub",
    }
)

#: Dotted call targets that download something.  Matched on the *suffix* of the
#: dotted name so ``hub.snapshot_download`` and ``huggingface_hub.snapshot_download``
#: both hit.
_DOWNLOAD_CALLS: tuple[str, ...] = (
    "urlopen",
    "urlretrieve",
    "hf_hub_download",
    "snapshot_download",
    "model_download",
    "dataset_download",
    "competition_download",
    "kagglehub.login",
    "torch.hub.load",
    "hub.load_state_dict_from_url",
    "load_state_dict_from_url",
    "nltk.download",
    "requests.get",
    "requests.post",
    "requests.head",
    "requests.put",
    "requests.request",
    "requests.Session",
    "httpx.get",
    "httpx.post",
    "httpx.Client",
    "urllib.request.urlopen",
    "urllib.request.urlretrieve",
)

#: Shell-ish substrings inside string literals that mean "fetch from the net"
#: or "install a package".  Kernels reach the shell via ``os.system``,
#: ``subprocess``, or the notebook ``!``/``get_ipython().system`` escape, all of
#: which end up as a string literal in the AST.
_SHELL_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bpip\s+install\b", "pip install"),
    (r"\bpip3\s+install\b", "pip3 install"),
    (r"\bconda\s+install\b", "conda install"),
    (r"\bapt(-get)?\s+install\b", "apt install"),
    (r"\bwget\b", "wget"),
    (r"\bcurl\s+-", "curl"),
    (r"\bgit\s+clone\b", "git clone"),
    (r"https?://", "http(s) URL"),
)

#: ``from_pretrained`` / ``load_dataset`` args that are local paths, not hub ids.
_LOCAL_PATH_PREFIXES = ("/", "./", "../", "~")


# ---------------------------------------------------------------------- parsing


def _canon(name: str) -> str:
    """PEP 503-style normalisation so ``scikit_learn`` == ``scikit-learn``."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def parse_env(path: str | Path | None = None) -> dict[str, str]:
    """Read a pinned requirements file into ``{canonical name: version}``.

    Keys are canonicalised (lowercase, ``-``-separated) so lookups do not have
    to care how the organizers spelled a package.  A VCS requirement
    (``git+https://…/CLIP.git``) has no version, so it maps to ``""`` — the
    package is *present*, we just cannot pin it.
    """
    p = Path(path) if path is not None else PINNED_ENV_PATH
    env: dict[str, str] = {}
    if not p.exists():
        return env

    for raw in p.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue

        if "://" in line:  # VCS / direct URL requirement
            if "#egg=" in line:
                name = line.split("#egg=", 1)[1].split("&", 1)[0]
            else:
                tail = line.split("#", 1)[0].rstrip("/").rsplit("/", 1)[-1]
                name = re.sub(r"\.git$", "", tail)
            if name:
                env.setdefault(_canon(name), "")
            continue

        # name[extra]==1.2.3 ; python_version < "3.13"
        line = line.split(";", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*(?:[=<>!~]=?\s*(.+))?$", line)
        if not m:
            continue
        env[_canon(m.group(1))] = (m.group(2) or "").strip()
    return env


def scan_imports(code: str) -> set[str]:
    """Return the set of **top-level** module names a kernel imports.

    Top-level only: ``import torch.nn.functional as F`` contributes ``torch``,
    because availability is decided per distribution, not per submodule.
    Relative imports (``from . import x``) are skipped — a single-file kernel
    has no package around it, and if one appears it is the author's own code.

    Raises ``SyntaxError`` if the code does not parse.  That is deliberate: a
    kernel that cannot be parsed cannot be submitted either, and the caller
    should see the real error.
    """
    tree = ast.parse(code)
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: not a distribution
                continue
            if node.module:
                mods.add(node.module.split(".", 1)[0])
    return mods


def _is_available(module: str, env: dict[str, str]) -> bool:
    """Is ``module`` importable in the pinned kernel environment?"""
    if module in ALWAYS_ALLOWED:
        return True
    if module in sys.stdlib_module_names:
        return True
    if _canon(module) in env:
        return True
    alias = IMPORT_ALIASES.get(module)
    if alias and _canon(alias) in env:
        return True
    return False


def check_imports(code: str, env: dict[str, str] | None = None) -> list[str]:
    """Report imports that would fail on the pinned Kaggle kernel.

    Returns human-readable violation strings; empty means clean.  ``env``
    defaults to :func:`parse_env` of :data:`PINNED_ENV_PATH`.

    Note the direction of the check: we ask "is it in the *pin*", never "is it
    installed *here*".  The agent box has internet and an arbitrary venv; the
    scoring kernel has neither, and confusing the two is the whole failure mode.
    """
    if env is None:
        env = parse_env()
    try:
        mods = scan_imports(code)
    except SyntaxError as exc:
        return [f"kernel does not parse: {exc.__class__.__name__}: {exc}"]

    out: list[str] = []
    for mod in sorted(mods):
        if _is_available(mod, env):
            continue
        alias = IMPORT_ALIASES.get(mod)
        hint = f" (distribution '{alias}')" if alias else ""
        out.append(
            f"import '{mod}'{hint} is not in the pinned Kaggle environment "
            f"({PINNED_ENV_PATH.name}) and is not stdlib — the kernel will "
            f"ImportError before it trains anything"
        )
    return out


# ---------------------------------------------------------------------- network


def _dotted(node: ast.AST) -> str:
    """Best-effort dotted name for a call target (``a.b.c`` -> ``'a.b.c'``)."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        return ""
    return ".".join(reversed(parts))


def _looks_like_hub_id(value: str) -> bool:
    """True for ``bert-base-uncased`` / ``org/model``, false for local paths.

    A local path is the *fix* for this violation (mount the weights as a Kaggle
    dataset and point at ``/kaggle/input/...``), so anything that is clearly a
    path is not reported.  Variables and f-strings are not reported either —
    they cannot be resolved statically and guessing produces noise.
    """
    v = value.strip()
    if not v:
        return False
    if v.startswith(_LOCAL_PATH_PREFIXES):
        return False
    if "kaggle/input" in v or "kaggle/working" in v:
        return False
    return True


def check_no_network(code: str) -> list[str]:
    """Report anything that would need the internet inside a scoring kernel.

    Covers the five ways we have actually seen a kernel reach the network:

    * importing/using an HTTP client (``requests``, ``httpx``, ``urllib.request``…),
    * ``from_pretrained("bert-base-uncased")`` — a *bare hub id* makes
      transformers hit huggingface.co; the same call with a local path is fine
      (see ``memory/lessons/kaggle-kernel-input-paths.md``),
    * ``pip install`` / ``conda install`` smuggled through a shell string,
    * ``wget`` / ``curl`` / ``git clone`` in a shell string,
    * ``kagglehub`` / ``hf_hub_download`` / ``torch.hub.load`` downloads.

    Returns human-readable strings with line numbers; empty means clean.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"kernel does not parse: {exc.__class__.__name__}: {exc}"]

    out: list[str] = []
    seen: set[str] = set()

    def add(line: int, msg: str) -> None:
        key = f"{line}:{msg}"
        if key not in seen:
            seen.add(key)
            out.append(f"line {line}: {msg}")

    for node in ast.walk(tree):
        # --- network client imports -------------------------------------
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _NETWORK_MODULES:
                    add(node.lineno, f"imports network client '{alias.name}' — internet is disabled")
                elif alias.name.startswith("urllib.request"):
                    add(node.lineno, "imports 'urllib.request' — internet is disabled")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".", 1)[0]
            if not node.level and (root in _NETWORK_MODULES or mod.startswith("urllib.request")):
                add(node.lineno, f"imports from network client '{mod}' — internet is disabled")

        # --- download-ish calls ------------------------------------------
        elif isinstance(node, ast.Call):
            name = _dotted(node.func)
            leaf = name.rsplit(".", 1)[-1] if name else ""
            if name:
                for pat in _DOWNLOAD_CALLS:
                    hit = (
                        name == pat
                        or name.endswith("." + pat)
                        or ("." not in pat and leaf == pat)
                    )
                    if hit:
                        add(node.lineno, f"calls '{name}()' which downloads over the network")
                        break

            # from_pretrained("bert-base") -> hub fetch; local path -> fine.
            if leaf in ("from_pretrained", "from_config", "load_dataset", "get_tokenizer"):
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if _looks_like_hub_id(arg.value):
                            add(
                                node.lineno,
                                f"{leaf}({arg.value!r}) uses a bare hub id, not a local path — "
                                f"it will try to reach the hub; point it at /kaggle/input/...",
                            )
                        break

            # pretrained=True on a model constructor pulls weights from the net.
            for kw in node.keywords:
                if kw.arg == "pretrained" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    add(
                        node.lineno,
                        f"'{name or 'call'}(pretrained=True)' downloads weights — "
                        f"load them from a mounted /kaggle/input dataset instead",
                    )
                if (
                    kw.arg == "weights"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                    and _looks_like_hub_id(kw.value.value)
                ):
                    add(
                        node.lineno,
                        f"'{name or 'call'}(weights={kw.value.value!r})' downloads pretrained "
                        f"weights over the network",
                    )

        # --- shell strings ------------------------------------------------
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if len(text) > 4000:  # a data blob, not a command
                continue
            for pattern, label in _SHELL_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    add(
                        node.lineno,
                        f"string literal contains '{label}' — no package installs and no "
                        f"network access inside the scoring kernel",
                    )
    return out


def check_kernel(code: str, env: dict[str, str] | None = None) -> list[str]:
    """Convenience: every static check, in one list.

    This is what the dry-run records before executing a kernel.
    """
    return check_imports(code, env) + check_no_network(code)
