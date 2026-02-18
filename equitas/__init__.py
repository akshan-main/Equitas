"""Equitas: A corruption robustness benchmark for multi-LLM committees."""

__version__ = "0.1.0"

# --- httpx compatibility shim ---
# The dev httpx at ~/projects/httpx doesn't handle event_hooks=None in its
# Client.__init__. Patch it so openai's SyncHttpxClientWrapper works.
import httpx._client as _httpx_client

_orig_init = _httpx_client.Client.__init__


def _patched_init(self, *args, **kwargs):
    if "event_hooks" in kwargs and kwargs["event_hooks"] is None:
        kwargs["event_hooks"] = {}
    return _orig_init(self, *args, **kwargs)


_httpx_client.Client.__init__ = _patched_init
