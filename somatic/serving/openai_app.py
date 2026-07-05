"""The OpenAI-compatible FastAPI app over a ClusterEngine.

Exposes `/v1/chat/completions` (streaming SSE + non-streaming), `/v1/models`, and
`/health`. Kept dependency-light and framework-plain so any OpenAI-ecosystem
client (the openai SDK, Open WebUI, LangChain, curl) works unchanged. The chat
UI and cluster-status routes are layered on separately in
`somatic.cluster.server` so this stays a clean, reusable API surface.
"""

import json
import time


def build_openai_app(engine, *, served_model_name: str):
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(title="somatic split-model server")
    app.state.engine = engine
    app.state.served_model_name = served_model_name

    # Open the API to any origin so browser-based chat UIs (Open WebUI, LibreChat,
    # Chatbox, …) can call it directly. This is a personal/home tool with no
    # secrets behind it; an Authorization header, if sent, is accepted and ignored.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:
        await engine.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await engine.close()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model": served_model_name, "workers": engine.layout}

    @app.get("/v1/models")
    async def list_models() -> dict:
        return {
            "object": "list",
            "data": [
                {"id": served_model_name, "object": "model", "owned_by": "somatic-cluster"}
            ],
        }

    def _messages(payload: dict) -> list[dict[str, str]]:
        messages = []
        for item in payload.get("messages", []):
            content = item.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            messages.append({"role": item.get("role", "user"), "content": content})
        return messages

    def _max_tokens(payload: dict) -> int:
        for key in ("max_completion_tokens", "max_tokens"):
            value = payload.get(key)
            if isinstance(value, int) and value > 0:
                return value
        return 256

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        messages = _messages(payload)
        max_new_tokens = _max_tokens(payload)
        created = int(time.time())
        completion_id = f"chatcmpl-{created}"

        def chunk(delta: dict, finish_reason=None) -> str:
            body = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": served_model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(body)}\n\n"

        if payload.get("stream"):
            import asyncio

            async def event_stream():
                queue: asyncio.Queue = asyncio.Queue()
                sentinel = object()
                failure: dict = {}

                async def on_token(delta: str) -> None:
                    await queue.put(delta)

                async def run() -> None:
                    try:
                        await engine.generate(
                            messages, max_new_tokens=max_new_tokens, on_token=on_token
                        )
                    except Exception as exc:  # surface, don't crash the stream
                        failure["error"] = str(exc)
                    finally:
                        await queue.put(sentinel)

                task = asyncio.create_task(run())
                try:
                    yield chunk({"role": "assistant"})
                    while True:
                        item = await queue.get()
                        if item is sentinel:
                            break
                        yield chunk({"content": item})
                    reason = "stop" if "error" not in failure else "error"
                    yield chunk({}, finish_reason=reason)
                    yield "data: [DONE]\n\n"
                finally:
                    # If the client disconnected mid-stream (GeneratorExit) or on any
                    # exit, stop the generation so it doesn't run on, holding the
                    # engine lock and filling the queue.
                    if not task.done():
                        task.cancel()

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        reply = await engine.generate(messages, max_new_tokens=max_new_tokens)
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": served_model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
            }
        )

    @app.post("/v1/completions")
    async def completions(request: Request):
        # Legacy text-completion endpoint some clients fall back to. We treat the
        # prompt as a single user turn so it still works on a chat model.
        payload = await request.json()
        prompt = payload.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "".join(str(p) for p in prompt)
        created = int(time.time())
        reply = await engine.generate(
            [{"role": "user", "content": str(prompt)}],
            max_new_tokens=_max_tokens(payload),
        )
        return JSONResponse(
            {
                "id": f"cmpl-{created}",
                "object": "text_completion",
                "created": created,
                "model": served_model_name,
                "choices": [{"index": 0, "text": reply, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            }
        )

    return app
