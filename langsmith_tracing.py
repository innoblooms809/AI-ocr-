"""
langsmith_tracing.py — LangSmith traces for the ViaExam Python services.

The same file ships with the AI evaluator (AI--main) and the OCR server (AI-OCR); each server
calls configure() once with its own name.

On when LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set (each server reads its git-ignored
.env.local). Traces go to the agent address LANGSMITH_AGENT_ID + LANGSMITH_AGENT_ENVIRONMENT
(langsmith>=0.15; LANGSMITH_PROJECT must not be set next to it, or the SDK traces nothing). Both
servers report to the same agent; the `service` metadata tells them apart.

    root(...)            the run of one request: a new trace, or a child of the caller's run
    span(...)            a step inside it (an agent, a question, an OCR batch)
    log_llm(...)         one model call, with its real start / first-token / end times and tokens
    outbound_headers()   `langsmith-trace` + `baggage` (+ X-Trace-Thread-Id) for a call to the other
                         service, so its work appears inside this trace
    inbound(headers)     continue a trusted caller's trace; X-Trace-Thread-Id puts the traces of one
                         answer sheet (its OCR, its evaluation, re-evaluations) in one thread

What leaves the server: every input and output passes through _clean(): never images or base64
data, text trimmed to LANGSMITH_MAX_TEXT_CHARS (default 4000; the prompt and reply of a model call
to LANGSMITH_MAX_PROMPT_CHARS, default 20000). LANGSMITH_HIDE_INPUTS / LANGSMITH_HIDE_OUTPUTS=true
send no inputs / outputs at all: names, timings, token counts and metadata only.

Tracing never breaks a request: every failure here is logged once and ignored.
"""
from __future__ import annotations

import contextvars
import dataclasses
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Mapping, Optional

logger = logging.getLogger("langsmith_tracing")

try:
    import langsmith as _ls
    from langsmith.run_trees import RunTree as _RunTree
except Exception:  # pragma: no cover - the package comes with langgraph / is in requirements
    _ls = None
    _RunTree = None

SDK_VERSION = getattr(_ls, "__version__", "") if _ls is not None else ""
THREAD_HEADER = "X-Trace-Thread-Id"   # one thread per answer sheet (the backend's run key)
RUN_HEADER = "X-Trace-Run-Id"         # the backend's run id (one evaluation attempt of a sheet)
MAX_TEXT = int(os.getenv("LANGSMITH_MAX_TEXT_CHARS", "4000"))
MAX_PROMPT = int(os.getenv("LANGSMITH_MAX_PROMPT_CHARS", "20000"))
_MAX_ITEMS = 40
_MAX_KEYS = 80
_MAX_DEPTH = 6
_ID_RE = re.compile(r"[^A-Za-z0-9._:\-]")
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")

_service: Dict[str, str] = {"name": "python-service", "version": ""}
_current: "contextvars.ContextVar[Optional[Any]]" = contextvars.ContextVar("ls_current_run", default=None)
_inbound: "contextvars.ContextVar[Optional[Dict[str, Any]]]" = contextvars.ContextVar("ls_inbound", default=None)
_client_lock = threading.Lock()
_client_obj: Any = None
_warned: set = set()
_project_url: Dict[str, Any] = {"url": "", "tried_at": 0.0}


def configure(service: str, version: str = "") -> None:
    """Name this server in every run it sends (metadata `service`, tag `service:<name>`)."""
    _service.update(name=service, version=version)


# ── Settings ────────────────────────────────────────────────────────────────────

def _env(name: str, legacy: bool = True) -> str:
    value = os.environ.get(f"LANGSMITH_{name}", "")
    if not value and legacy:
        value = os.environ.get(f"LANGCHAIN_{name}", "")
    return value.strip()


def _flag(name: str) -> bool:
    return _env(name).lower() == "true"


def enabled() -> bool:
    """Tracing is on: the SDK imports, LANGSMITH_TRACING=true and an API key is set."""
    if _ls is None or _RunTree is None:
        return False
    on = any(os.environ.get(n, "").strip().lower() == "true"
             for n in ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"))
    return on and bool(_env("API_KEY"))


def _version(v: str) -> tuple:
    return tuple(int(p) for p in re.findall(r"\d+", v)[:3]) or (0,)


def status() -> Dict[str, Any]:
    """Where traces go and what is wrong with the settings, for /health and the dashboard."""
    agent_id = _env("AGENT_ID", legacy=False)
    agent_env = _env("AGENT_ENVIRONMENT", legacy=False)
    project = _env("PROJECT") or os.environ.get("LANGCHAIN_SESSION", "").strip()
    problems: List[str] = []
    if _ls is None:
        problems.append("the langsmith package is not installed (pip install 'langsmith>=0.15')")
    elif (agent_id or agent_env) and _version(SDK_VERSION) < (0, 15):
        problems.append(f"langsmith {SDK_VERSION} ignores LANGSMITH_AGENT_ID: upgrade to langsmith>=0.15")
    if bool(agent_id) != bool(agent_env):
        problems.append("set both LANGSMITH_AGENT_ID and LANGSMITH_AGENT_ENVIRONMENT")
    if agent_id and project:
        problems.append("LANGSMITH_PROJECT is set next to the agent address: the SDK then traces nothing; unset it")
    tracing_on = any(os.environ.get(n, "").strip().lower() == "true"
                     for n in ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"))
    if tracing_on and not _env("API_KEY"):
        problems.append("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is not set")
    return {
        "enabled": enabled(),
        "service": _service["name"],
        "sdk_version": SDK_VERSION,
        "endpoint": _env("ENDPOINT") or "https://api.smith.langchain.com",
        "destination": (f"agent {agent_id} / {agent_env}" if agent_id else f"project {project or 'default'}"),
        "agent_id": agent_id or None,
        "agent_environment": agent_env or None,
        "hide_inputs": _flag("HIDE_INPUTS"),
        "hide_outputs": _flag("HIDE_OUTPUTS"),
        "max_text_chars": MAX_TEXT,
        "project_url": _project_url["url"] or None,
        "problems": problems,
    }


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(f"[LangSmith] {message} (tracing continues without it; the request is not affected)")


# ── What leaves the server ──────────────────────────────────────────────────────

def _clean_text(text: str, limit: int) -> str:
    if text.startswith("data:") and ";base64," in text[:100]:
        return f"<image data: {len(text) * 3 // 4 // 1024} KB, not sent>"
    if len(text) > 512 and " " not in text[:512] and _BASE64_RE.match(text[:2048]):
        return f"<binary data: {len(text) * 3 // 4 // 1024} KB, not sent>"
    if len(text) > limit:
        return text[:limit] + f" … [+{len(text) - limit} characters not sent]"
    return text


def _clean(value: Any, limit: int = MAX_TEXT, depth: int = 0) -> Any:
    """A copy safe to send: no images / base64, text trimmed, big collections cut short."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _clean_text(value, limit)
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes, not sent>"
    if depth >= _MAX_DEPTH:
        return f"<{type(value).__name__}>"
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= _MAX_KEYS:
                out["…"] = f"+{len(value) - _MAX_KEYS} more keys"
                break
            out[str(k)] = _clean(v, limit, depth + 1)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        out_list = [_clean(v, limit, depth + 1) for v in items[:_MAX_ITEMS]]
        if len(items) > _MAX_ITEMS:
            out_list.append(f"… +{len(items) - _MAX_ITEMS} more")
        return out_list
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"_type": type(value).__name__,
                **_clean({f.name: getattr(value, f.name, None) for f in dataclasses.fields(value)}, limit, depth + 1)}
    if hasattr(value, "model_dump"):
        try:
            return _clean(value.model_dump(), limit, depth + 1)
        except Exception:
            pass
    return _clean_text(repr(value), 300)


def _hide_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    if _flag("HIDE_INPUTS"):
        return {}
    return _clean(inputs, MAX_PROMPT if "messages" in inputs else MAX_TEXT)


def _hide_outputs(outputs: Dict[str, Any]) -> Dict[str, Any]:
    if _flag("HIDE_OUTPUTS"):
        usage = outputs.get("usage_metadata") if isinstance(outputs, dict) else None
        return {"usage_metadata": usage} if usage else {}
    return _clean(outputs, MAX_PROMPT if "choices" in outputs else MAX_TEXT)


def _client() -> Any:
    """One client per process, sending through _clean()."""
    global _client_obj
    if _client_obj is None:
        with _client_lock:
            if _client_obj is None:
                _client_obj = _ls.Client(hide_inputs=_hide_inputs, hide_outputs=_hide_outputs)
    return _client_obj


# ── Runs ────────────────────────────────────────────────────────────────────────

class Span:
    """A run of the trace, or nothing when tracing is off (every method is then a no-op)."""

    __slots__ = ("run", "outputs", "error")

    def __init__(self, run: Any = None):
        self.run = run
        self.outputs: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.run is not None

    @property
    def trace_id(self) -> str:
        return str(self.run.trace_id) if self.run is not None else ""

    @property
    def run_id(self) -> str:
        return str(self.run.id) if self.run is not None else ""

    def end(self, outputs: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        """Outputs (and error) of this run; it is sent when its `with` block ends."""
        if outputs is not None:
            self.outputs = outputs
        if error:
            self.error = error

    def add_metadata(self, **metadata: Any) -> None:
        if self.run is not None:
            try:
                self.run.add_metadata(metadata)
            except Exception:
                pass


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, timezone.utc)


def _tags(tags: Optional[List[str]]) -> List[str]:
    return sorted({f"service:{_service['name']}", *(tags or [])})


def _base_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"service": _service["name"]}
    if _service["version"]:
        meta["service_version"] = _service["version"]
    info = _inbound.get() or {}
    for key in ("thread_id", "backend_run_id"):
        if info.get(key):
            meta[key] = info[key]
    for key, value in (metadata or {}).items():
        if value is not None and value != "":
            meta[key] = value
    return meta


@contextmanager
def _activate(run: Any) -> Iterator[Span]:
    """Make `run` the parent of the runs started inside the block; send its end on exit."""
    span = Span(run)
    token = _current.set(run) if run is not None else None
    try:
        yield span
    except BaseException as e:
        span.error = span.error or f"{type(e).__name__}: {e}"
        raise
    finally:
        if token is not None:
            _current.reset(token)
        if run is not None:
            try:
                run.end(outputs=span.outputs, error=span.error)
                run.patch()
            except Exception as e:
                _warn_once("finish", f"could not send the end of a run: {e}")


def _inbound_parent() -> Any:
    info = _inbound.get() or {}
    headers = info.get("parent_headers")
    if not headers:
        return None
    try:
        return _RunTree.from_headers(headers, client=_client())
    except Exception as e:
        _warn_once("inbound", f"could not continue the caller's trace from its headers: {e}")
        return None


@contextmanager
def root(name: str, *, run_type: str = "chain", inputs: Optional[Dict[str, Any]] = None,
         metadata: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None) -> Iterator[Span]:
    """The run of one request: a child of the current run, else of the trusted caller's run
    (see inbound()), else the start of a new trace."""
    if not enabled():
        yield Span()
        return
    run = None
    try:
        meta = _base_metadata(metadata)
        parent = _current.get() or _inbound_parent()
        if parent is not None:
            run = parent.create_child(name, run_type, inputs=inputs or {}, tags=_tags(tags), extra={"metadata": meta})
        else:
            run = _RunTree(name=name, run_type=run_type, inputs=inputs or {}, tags=_tags(tags),
                           extra={"metadata": meta}, client=_client())
        run.post()
    except Exception as e:
        _warn_once("root", f"could not start a trace: {e}")
        run = None
    with _activate(run) as span:
        yield span


@contextmanager
def span(name: str, *, run_type: str = "chain", inputs: Optional[Dict[str, Any]] = None,
         metadata: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None) -> Iterator[Span]:
    """A step inside the current run (nothing when there is none, or tracing is off)."""
    parent = _current.get() if enabled() else None
    if parent is None:
        yield Span()
        return
    run = None
    try:
        run = parent.create_child(name, run_type, inputs=inputs or {}, tags=_tags(tags),
                                  extra={"metadata": {k: v for k, v in (metadata or {}).items() if v is not None}})
        run.post()
    except Exception as e:
        _warn_once("span", f"could not start a run: {e}")
        run = None
    with _activate(run) as s:
        yield s


def log_llm(name: str, *, start: float, end: float, first_token: Optional[float] = None,
            prompt: str = "", images: Optional[List[str]] = None, reply: str = "", error: str = "",
            usage: Optional[Dict[str, int]] = None, metadata: Optional[Dict[str, Any]] = None,
            model: str = "", provider: str = "transformers", max_tokens: Optional[int] = None,
            tags: Optional[List[str]] = None) -> None:
    """One finished model call under the current run: prompt (images as short notes, never the
    pictures), reply, token usage, and a `new_token` event at the first token (LangSmith shows the
    time to first token from it). start / end / first_token: time.time() values."""
    parent = _current.get() if enabled() else None
    if parent is None:
        return
    try:
        if images:
            content: Any = [{"type": "text", "text": f"[image: {d}]"} for d in images] + [{"type": "text", "text": prompt}]
        else:
            content = prompt
        meta: Dict[str, Any] = {"ls_provider": provider, "ls_model_name": model, "ls_model_type": "chat",
                                "ls_temperature": 0.0}
        if max_tokens:
            meta["ls_max_tokens"] = max_tokens
        meta.update({k: v for k, v in (metadata or {}).items() if v is not None})
        if usage:
            meta["usage_metadata"] = usage
        run = parent.create_child(name, "llm", inputs={"messages": [{"role": "user", "content": content}]},
                                  start_time=_dt(start), tags=_tags(tags), extra={"metadata": meta})
        if first_token:
            run.add_event({"name": "new_token", "time": _dt(first_token).isoformat()})
        outputs = None
        if not error:
            outputs = {"choices": [{"message": {"role": "assistant", "content": reply}}]}
            if usage:
                outputs["usage_metadata"] = usage
        run.end(outputs=outputs, error=error or None, end_time=_dt(max(end, start)))
        run.post()
    except Exception as e:
        _warn_once("llm", f"could not send a model call: {e}")


# ── Across services ─────────────────────────────────────────────────────────────

def outbound_headers() -> Dict[str, str]:
    """Headers for a call to the other service: its runs then join this trace (and thread)."""
    run = _current.get() if enabled() else None
    headers: Dict[str, str] = {}
    if run is not None:
        try:
            headers.update(run.to_headers())
        except Exception as e:
            _warn_once("outbound", f"could not build trace headers: {e}")
    thread = thread_id()
    if thread:
        headers[THREAD_HEADER] = thread
    return headers


@contextmanager
def inbound(headers: Optional[Mapping[str, Any]]) -> Iterator[None]:
    """For the length of a request: continue its caller's trace (`langsmith-trace` + `baggage`)
    and take its thread id (X-Trace-Thread-Id) and run id (X-Trace-Run-Id).

    Use only on requests that passed the service's token check: these headers decide where runs
    are recorded, so they must come from our own services."""
    info: Dict[str, Any] = {}
    try:
        lower = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        if lower.get("langsmith-trace"):
            info["parent_headers"] = {k: lower[k] for k in ("langsmith-trace", "baggage") if lower.get(k)}
        for key, header in (("thread_id", THREAD_HEADER), ("backend_run_id", RUN_HEADER)):
            value = _ID_RE.sub("", lower.get(header.lower(), ""))[:128]
            if value:
                info[key] = value
    except Exception:
        info = {}
    token = _inbound.set(info or None)
    try:
        yield
    finally:
        _inbound.reset(token)


def inbound_info() -> Dict[str, Any]:
    """What the caller of the current request sent: thread_id, backend_run_id (both may be missing)."""
    info = _inbound.get() or {}
    return {k: v for k, v in info.items() if k != "parent_headers"}


def thread_id() -> str:
    """The thread of the current request (from the caller, or the current run's metadata)."""
    info = _inbound.get() or {}
    if info.get("thread_id"):
        return info["thread_id"]
    run = _current.get()
    if run is not None:
        try:
            return str((run.extra or {}).get("metadata", {}).get("thread_id") or "")
        except Exception:
            return ""
    return ""


@contextmanager
def no_auto_tracing() -> Iterator[None]:
    """Keep LangChain / LangGraph from tracing on their own inside this block (our spans trace it).

    With langchain-core < 1.8 their runs ignore the agent address and land in a project named
    "default"; they would also send the whole graph state. Our spans don't depend on this switch."""
    try:
        from langsmith.run_helpers import tracing_context
        cm = tracing_context(enabled=False)
    except Exception:
        yield
        return
    with cm:
        yield


def flush(timeout: float = 10.0) -> None:
    """Send what is still queued (at shutdown, and at the end of scripts)."""
    if _client_obj is not None:
        try:
            _client_obj.flush(timeout=timeout)
        except TypeError:
            _client_obj.flush()
        except Exception:
            pass


# ── Links ───────────────────────────────────────────────────────────────────────

def project_url(retry_after_s: float = 300.0) -> str:
    """Web link of the project the traces land in (the agent environment's project), or "".

    Two API calls, made once (again after a failure, at most every 5 minutes): call it from a
    worker thread, never on the evaluation path."""
    if _project_url["url"] or not enabled() or time.time() - _project_url["tried_at"] < retry_after_s:
        return _project_url["url"]
    _project_url["tried_at"] = time.time()
    try:
        client = _client()
        host = client._host_url
        tenant = client._get_tenant_id()
        agent_id = _env("AGENT_ID", legacy=False)
        agent_env = _env("AGENT_ENVIRONMENT", legacy=False).lower()
        if agent_id and agent_env:
            data = client.request_with_retries("GET", f"/api/v1/agents/{agent_id}").json()
            session = next(e["tracer_session_id"] for e in data.get("environments", [])
                           if str(e.get("environment", "")).lower() == agent_env)
        else:
            session = client.read_project(project_name=_env("PROJECT") or "default").id
        _project_url["url"] = f"{host}/o/{tenant}/projects/p/{session}"
    except Exception as e:
        _warn_once("project_url", f"could not work out the project link: {e}")
    return _project_url["url"]


def trace_url(trace_id: str) -> str:
    """Web link of one trace, when the project link is known."""
    base = _project_url["url"]
    return f"{base}/r/{trace_id}?trace_id={trace_id}&poll=true" if base and trace_id else ""
