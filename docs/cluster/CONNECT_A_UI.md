# Use Computer Soup with an existing chat UI

Computer Soup serves an **OpenAI-compatible API**, so you don't need its built-in chat
page — point any app that speaks the OpenAI protocol at your split model and use
the UI you already like. The built-in page at `/` is just a zero-setup fallback.

## 1. Start the cluster, exposed to your network

```bash
soup run Qwen/Qwen3-1.7B --host localhost --host user@other-mac --expose
```

`--expose` binds the API to `0.0.0.0` so an app on another device (or in a Docker
container) can reach it. Computer Soup prints the address to use:

```
  chat UI        http://192.168.1.42:8000/
  OpenAI API     http://192.168.1.42:8000/v1
  model name     Qwen/Qwen3-1.7B
```

Point your app at the **OpenAI API** URL. **No API key is required** — send any
non-empty key if the app insists on one.

> Leave off `--expose` to keep it on `127.0.0.1` (only apps on the same machine
> can reach it). If your app runs in Docker, use `--expose` and reach the driver
> at its LAN IP (or `host.docker.internal` from the container).

## 2. Point your UI at it

| App | Where to put the URL |
|-----|----------------------|
| **Open WebUI** | Settings → Connections → OpenAI API: Base URL `http://<driver-ip>:8000/v1`, any API key. |
| **LibreChat** | In `librechat.yaml` / env: a custom endpoint with `baseURL: http://<driver-ip>:8000/v1`. |
| **Chatbox / Jan / BoltAI** | Add an "OpenAI-compatible" / custom provider, API host `http://<driver-ip>:8000/v1`, any key. |
| **`openai` SDK** | `OpenAI(base_url="http://<driver-ip>:8000/v1", api_key="soup")` |
| **`curl`** | `curl http://<driver-ip>:8000/v1/chat/completions -d '{"model":"<id>","messages":[...]}'` |

The model shows up under its Hugging Face id (e.g. `Qwen/Qwen3-1.7B`) via
`GET /v1/models`, so most apps auto-discover it.

## What's supported

- `GET  /v1/models` — model discovery
- `POST /v1/chat/completions` — streaming (SSE) and non-streaming
- `POST /v1/completions` — legacy text completion (for older clients)
- **CORS** is open (`Access-Control-Allow-Origin: *`), so browser-based UIs work.
- `GET /health` — liveness + the current layer split.

## From Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.1.42:8000/v1", api_key="soup")
stream = client.chat.completions.create(
    model="Qwen/Qwen3-1.7B",
    messages=[{"role": "user", "content": "Explain layer-split inference."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Notes

- **One generation at a time.** Computer Soup serves a home cluster, not a multi-tenant
  gateway — concurrent requests are serialised. A UI that fires several requests
  at once will see them run one after another.
- **No auth.** Anyone who can reach the address can use it. Only `--expose` on a
  network you trust.
- Prefer the built-in page? It's at `http://<driver-ip>:8000/`.
