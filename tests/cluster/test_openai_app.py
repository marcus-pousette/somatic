"""The OpenAI app is compatible with existing UIs: CORS, models, completions."""

from __future__ import annotations

import pytest


class _FakeEngine:
    """A ClusterEngine stand-in — no workers, deterministic replies."""

    layout = "[0,1)@localhost"

    async def start(self) -> None: ...
    async def close(self) -> None: ...

    async def generate(self, messages, *, max_new_tokens=256, on_token=None, boundary_strategy=None):
        return "pong"


def _client():
    from fastapi.testclient import TestClient

    from soup.serving.openai_app import build_openai_app

    app = build_openai_app(_FakeEngine(), served_model_name="test/model")
    return TestClient(app)


def test_cors_preflight_allows_browser_uis() -> None:
    client = _client()
    resp = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "*"


def test_models_discovery() -> None:
    client = _client()
    data = client.get("/v1/models").json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "test/model"


def test_chat_completions_accepts_auth_header_and_returns_reply() -> None:
    client = _client()
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-anything"},
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "pong"


def test_legacy_completions_endpoint() -> None:
    client = _client()
    resp = client.post("/v1/completions", json={"model": "x", "prompt": "hello", "max_tokens": 4})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "pong"


def test_lan_ip_returns_a_string() -> None:
    from soup.cluster.launcher import _lan_ip

    ip = _lan_ip()
    assert isinstance(ip, str) and ip.count(".") == 3
