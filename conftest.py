import urllib.request

import pytest

try:  # Optional requests dependency
    import requests  # type: ignore
except Exception:  # pragma: no cover - requests may be absent
    requests = None

try:  # Torch may be unavailable in some environments
    import torch.hub as torch_hub  # type: ignore
except Exception:  # pragma: no cover - torch may be absent
    torch_hub = None

try:  # Training module may be absent in plugin-only deployments
    from training.utils.seed import set_seed
except ImportError:  # pragma: no cover - training may be absent
    set_seed = None





@pytest.fixture(autouse=True)
def _set_deterministic_seed():
    if set_seed is not None:
        set_seed(1234)


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    def _deny(*args, **kwargs):  # pragma: no cover - behavior-only
        raise RuntimeError("Network access disabled during tests")

    monkeypatch.setattr(urllib.request, "urlopen", _deny)
    if requests is not None:
        monkeypatch.setattr(requests, "get", _deny, raising=False)
        monkeypatch.setattr(requests, "post", _deny, raising=False)
    if torch_hub is not None:
        monkeypatch.setattr(torch_hub, "load", _deny, raising=False)


@pytest.fixture(autouse=True)
def _segmenter_env(monkeypatch):
    monkeypatch.setenv("SEGMENTER_SKIP_AUTO_INSTALL", "1")
    # Keep deterministic sampling defaults across tests
    monkeypatch.setenv("PYTHONHASHSEED", "0")
