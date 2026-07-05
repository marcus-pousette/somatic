# Run a model split across your machines

Somatic runs an LLM that's too big for one machine by splitting its transformer
layers across the machines you have, and serving it behind an OpenAI-compatible
API with a built-in chat page. No machine ever holds the whole model.

## Quickstart

On the machine you'll drive from (the "driver"):

```bash
somatic run Qwen/Qwen3-1.7B --host localhost --host user@other-machine
```

That's it. Somatic will:

1. **read the model** — exact per-layer and head bytes from the safetensors headers (no weights downloaded just to plan);
2. **probe each machine's free RAM**;
3. **auto-split** the layers so each machine's slice fits its RAM (capacity-proportional — a bigger machine takes more layers);
4. **launch a worker** on each machine (over SSH for remote ones), each loading only its own layer slice;
5. **serve** an OpenAI API + a chat page on the driver.

```
somatic ▸ Qwen/Qwen3-1.7B  28 layers · 0.094 GiB/layer · head 0.58 GiB (bf16)
somatic ▸ free RAM   localhost: 18.1 GiB   other-machine: 24.1 GiB
somatic ▸ plan       localhost [0,10)   other-machine [10,28)
somatic ▸ ready.  chat http://127.0.0.1:8000/   api http://127.0.0.1:8000/v1
```

Open `http://127.0.0.1:8000/` and start typing, or point any OpenAI client at
`http://127.0.0.1:8000/v1`. Stop it with `somatic down`.

**Prefer your own chat app?** Somatic's API is OpenAI-compatible, so you can use
Open WebUI, LibreChat, Chatbox, the `openai` SDK, etc. instead of the built-in
page — add `--expose` and see [CONNECT_A_UI.md](CONNECT_A_UI.md).

> A model that fits on the first machine stays there — Somatic only splits when
> it must, because a split adds network hops. To force a split (or place layers
> yourself), pin ranges: `--host localhost:0-14 --host user@other:14-28`.

## The SDK

The CLI is a thin skin over a Python API:

```python
from somatic import Cluster

with Cluster.launch("Qwen/Qwen3-1.7B",
                    ["localhost", "user@other-machine"]) as c:
    print(c.plan_.layout)                       # [0,10)@localhost -> [10,28)@other-machine
    print(c.chat("Explain layer-split inference in one line."))
    for delta in c.stream("Now in French."):    # streaming
        print(delta, end="", flush=True)

    client = c.openai()                          # a stock openai.OpenAI client
    client.chat.completions.create(model=c.model,
        messages=[{"role": "user", "content": "hi"}])
```

`Cluster.plan(model, hosts)` does the whole probe+fit as a dry run without
launching anything — the answer to "will this fit, and how would it split?".

## Which models work

Any **Llama-family** Hugging Face decoder model — one whose backbone exposes
`model.embed_tokens`, `model.layers`, `model.norm`, `model.rotary_emb`. That
covers Llama, Qwen, Mistral, Gemma, Phi, DeepSeek, Yi, SmolLM, and most popular
open models. GPT-2 (`transformer.h`), GPT-NeoX (`gpt_neox.layers`), and Mamba
(state-space) are not yet supported.

## Setting up the machines: `somatic provision`

Each machine needs this repo, a Python env with the deps, and the model in its
Hugging Face cache. One command sets all of that up:

```bash
somatic provision --host user@other-machine --model Qwen/Qwen3-1.7B
```

This pushes the current code, ensures the deps (`uv sync` if needed), and warms
the model cache — and it's idempotent, so re-running an already-set-up host is a
fast "ready ✓". For a machine with slow or no internet, copy the model straight
from this one instead of downloading it there:

```bash
somatic provision --host user@other-machine --model Qwen/Qwen3-1.7B --push-model
```

(`--push-model` rsyncs the model from your machine's cache and repairs the
`refs/main` pointer that a raw cache copy leaves empty.) Use `--no-env` to skip
the dependency step (code + model only).

SSH to remote machines must be key-based and non-interactive. After provisioning,
`somatic run` runs a **preflight** and still refuses with an actionable message if
anything is missing — a far better failure than a silent hang.

## Flags

| flag | default | meaning |
|------|---------|---------|
| `--host`, `-h` | (required) | a machine: `localhost` or `user@ip`, optionally `:start-end` to pin layers. Repeat; first is the driver. |
| `--precision` | `bf16` | weight dtype the cluster holds (`bf16`/`fp16`/`fp32`). |
| `--port` | `8000` | OpenAI API + chat UI port on the driver. |
| `--expose` | | bind to `0.0.0.0` so a UI on another device / in Docker can reach the API. |
| `--headroom` | `0.80` | fraction of each machine's free RAM the fit may use. |
| `--plan-only` | | print the split and exit; launch nothing. |
| `--mode` | `relay` | boundary wire: `relay` / `exact` / `compact` (see below). |

Manage runs with `somatic ps` (list) and `somatic down [--sweep]` (stop / nuke orphans).

## Boundary modes and `somatic verify`

Workers exchange hidden states over the network. `--mode` picks how those are put
on the wire — all three are **model-general** (no per-model training):

| mode | strategy | wire bytes | fidelity |
|------|----------|-----------|----------|
| `exact` | identity | full precision (1×) | **provably identical** to a single-machine run |
| `relay` (default) | fp16 | ~0.5× | near-exact |
| `compact` | int8 | ~0.25× | more lossy |

`exact` sends the full-precision hidden state, so its output is bit-identical to
running the whole model on one machine — the differentiator, with no quality loss.
`relay` and `compact` trade measured fidelity for bandwidth over slower networks.

**Don't guess — measure it for your model:**

```bash
somatic verify Qwen/Qwen3-1.7B --host localhost --host user@other-machine
```

```
  exact  (identity)  deterministic: yes ✓
                     wire 8192 B / boundary·token  (this is the reference)

  mode     strategy        prompts exact  token match   wire vs exact
  relay    fp16            5/5            100%          4096 B  (0.50×)
  compact  int8_symmetric  4/5             91%          2048 B  (0.25×)
```

This runs prompts through the split cluster under each mode and reports how often
the generated tokens match the exact reference and what each mode costs on the
wire — so you can pick the mode that fits your network and quality bar. (An
*exact-and-compressed* mode via learned, margin-triggered codecs is proven for
specific models in the research track but is not yet model-general.)
