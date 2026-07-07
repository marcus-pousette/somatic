"""The cluster's HTTP surface: the OpenAI API + a self-contained chat page.

`build_cluster_app` takes the proven OpenAI app and layers on two routes:
  * `GET /`            — the single-file chat UI (no build step, ships in the wheel)
  * `GET /api/status`  — the resolved split (model, per-host layer ranges, RAM)

The chat page talks to the same-origin `/v1/chat/completions` (streaming), so a
user just opens the URL the launcher prints and starts typing.
"""

from __future__ import annotations

from importlib import resources


def build_cluster_app(engine, *, model_name: str, status: dict | None = None):
    from fastapi.responses import HTMLResponse, JSONResponse

    from soup.serving.openai_app import build_openai_app

    app = build_openai_app(engine, served_model_name=model_name)
    chat_html = resources.files("soup.cluster.assets").joinpath("chat.html").read_text(
        encoding="utf-8"
    )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return chat_html

    @app.get("/api/status")
    async def api_status() -> JSONResponse:
        return JSONResponse(status or {"model": model_name, "workers": engine.layout})

    return app
