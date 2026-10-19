# OpenRouter Image API Node (v2) Specification

## Overview
A new node, `OpenRouterImageAPINode`, that generates images through OpenRouter's dedicated Image API (`POST /api/v1/images`) instead of chat completions.

The existing `OpenRouterImageGenNode` stays as it is for workflow compatibility. It sends images through `/chat/completions` with `image_config`.

Reasons for a separate node:
- **Multiple images per call:** the Image API has a real `n` parameter (1–10). Chat completions returns one image per call, so 4 candidates cost 4 nodes.
- **Real size controls:** `resolution` and `aspect_ratio` are top-level and documented. The v1 `provider.data.size` hack now returns `400 provider: Unrecognized key: "data"`.
- **Clean response:** `data[].b64_json` with `usage.cost`. No chat message parsing.
- **Per-model capabilities:** `GET /api/v1/images/models` reports `supported_parameters`, so the node can drop parameters a model doesn't support instead of failing.

Source: <https://openrouter.ai/docs/guides/overview/multimodal/image-generation.md>

## API Contract (from OpenRouter docs)

### Request: `POST {BASE_URL}/images`
| Field | Type | Notes |
|---|---|---|
| `model` | string | Required |
| `prompt` | string | Required |
| `n` | int 1–10 | Upper bound. Providers may return fewer; single-image providers reject `n > 1` |
| `resolution` | `512` `768` `1K` `1.5K` `2K` `4K` | Size tier |
| `aspect_ratio` | `auto`, `1:1`, `2:3`, `3:4`, `16:9`, … | Providers clamp to what they support |
| `size` | tier or `WxH` | **Not used by this node.** It conflicts with `resolution`/`aspect_ratio` (400) |
| `seed` | int | Where supported |
| `quality` | `auto` `low` `medium` `high` | |
| `background` | `auto` `transparent` `opaque` | `transparent` needs png/webp |
| `output_format` | `png` `jpeg` `webp` | Node always requests `png` when supported |
| `input_references` | `[{type: "image_url", image_url: {url}}]` | HTTP(S) or base64 data URLs; count limit varies by provider |
| `provider` | `{only, order, ignore, sort, allow_fallbacks, options}` | Routing only |

### Response
```json
{"created": 1748372400,
 "data": [{"b64_json": "<base64>", "media_type": "image/png"}],
 "usage": {"prompt_tokens": 0, "completion_tokens": 4175, "total_tokens": 4175, "cost": 0.04}}
```
- If a generation doesn't complete, the API returns `502` rather than a partial result.
- A conflicting `size` returns `400`.

### Capabilities: `GET {BASE_URL}/images/models`
`data[]` entries have `id`, `name`, `supported_parameters` (the union across endpoints), `supports_streaming` and `endpoints`. **A parameter missing from `supported_parameters` is unsupported.** For details per provider, see `GET /images/models/{id}/endpoints`.

## Node Definition

- File: `node_image_api.py`
- Class / mapping key: `OpenRouterImageAPINode`
- Display name: `OpenRouter Image API Node`
- `CATEGORY = "LLM"`
- Registration: add it to `__init__.py` the same way as the other nodes. Add the ID to `DYNAMIC_IMAGE_NODE_IDS` in `web/openrouter_dynamic_inputs.js` so `image_1..image_N` sockets appear as you connect images.

### Inputs
| Name | Type | Default | Sent as |
|---|---|---|---|
| `api_key` | STRING | `""` | Resolved through `shared.get_api_key` (field → `LLM_KEY` → `openrouter_api_key.json`) |
| `prompt` | STRING multiline | `""` | `prompt` |
| `model` | COMBO | first model | `model`; list from `GET /images/models`, cached like `shared.fetch_filtered_models` |
| `n` | INT 1–10 | 1 | `n` (omitted when 1) |
| `aspect_ratio` | COMBO `auto, 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9, 9:21, 1:2, 2:1, 1:4, 4:1, 1:8, 8:1` | `auto` | `aspect_ratio` (omitted when `auto`) |
| `resolution` | COMBO `default, 512, 768, 1K, 1.5K, 2K, 4K` | `default` | `resolution` (omitted when `default`) |
| `quality` | COMBO `auto, low, medium, high` | `auto` | `quality` (omitted when `auto`) |
| `background` | COMBO `auto, transparent, opaque` | `auto` | `background` (omitted when `auto`) |
| `seed` | INT, `control_after_generate: "fixed"` | 0 | `seed` |
| `request_timeout` | INT | `shared.DEFAULT_REQUEST_TIMEOUT` | requests timeout |
| `retries` | INT 0–5 | 1 | Extra attempts per request on transient failures (last required input) |
| `prompt_input` (optional) | STRING, forceInput | — | Overrides `prompt` when non-empty, same as v1 |
| `image_1..image_N` (dynamic) | IMAGE | — | `input_references`, in socket order |

The aspect-ratio dropdown holds plain ratio strings, so no parsing is needed. This is unlike the v1 `"2:3 (832x1248)"` labels.

### Outputs
`RETURN_TYPES = ("IMAGE", "STRING", "STRING", "FLOAT")`, `RETURN_NAMES = ("images", "Stats", "Credits", "cost")`

- `images`: every returned image as one batch `[B, H, W, 3]`, where B is however many the provider returned (≤ `n`).
- `Stats`: model, image count, output size, elapsed time, `usage.cost`.
- `Credits`: `shared.fetch_credits`.
- `cost`: `usage.cost` summed over the split calls as a float; `0.0` when no call reported a cost.

## Behaviour

1. **Reference images:** every frame of every connected `image_N` becomes its own `input_references` entry, encoded as a PNG data URL (off the event loop, so concurrent nodes overlap). A batch on one socket therefore sends all its images. This fixes v1, which uses only frame 0 (`shared.image_to_base64` warns and drops the rest). Add a `shared.image_batch_to_base64_list` helper and leave v1 untouched.
2. **Capability filtering:** before sending, look up the model's `supported_parameters` from the cached `/images/models` data. Drop any optional field (`n`, `seed`, `aspect_ratio`, `resolution`, `quality`, `background`, `output_format`) that the model doesn't list, and print one line naming what was dropped. If the cache has no entry for the model, send everything.
   **Endpoint check:** for models in the cache, also fetch `GET /images/models/{id}/endpoints` (off the event loop, cached per model for `cache_duration`) and test the actual `aspect_ratio`, `resolution`, `quality`, `background`, per-call `n` and reference count against each endpoint's `supported_parameters` (`enum` values, `range` min/max; unknown descriptor types accept; a parameter an endpoint doesn't list counts as unsupported). If every endpoint accepts, send as is. If only some do, add `provider: {"only": [<provider_tag>, ...]}` to every call. If none does, raise `RuntimeError` before any request, e.g. `google/gemini-3.1-flash-image does not accept resolution=768 (allowed: 512, 1K, 2K, 4K)`. If the fetch fails, print one line and send without the check.
3. **Decoding:** decode each `data[].b64_json` with Pillow → RGB → float tensor. If the images come back at different sizes, resize them to the first image's size with `comfy.utils.common_upscale` (lanczos) before `torch.cat`, so the batch is always valid.
4. **Errors raise:** on HTTP errors, an empty `data` or a decode failure, raise `RuntimeError` with OpenRouter's error message. **Don't return a black placeholder.** v1 returns a 64×64 black image, which ComfyUI caches as a successful result, so you have to change the seed to retry. With an exception, ComfyUI stops the run, shows the error on the node and caches nothing.
5. **Retries:** each request retries up to `retries` times, waiting 2 s × attempt, on HTTP 408/429/500/502/503/504, content-moderation blocks ("moderation" in the message, or `reasons`/`flagged_input` in the error metadata) and `requests` timeouts or connection errors. Other errors raise at once. Chunks retry independently, so one failing chunk does not discard the others.
6. **Error detail:** error text appends `provider_name`, `reasons` and `raw` (first 300 characters) from the error metadata, e.g. `... 400: <message> (provider: Google; reasons: ...; raw: ...)`. `flagged_input` is never included.
7. **Caching:** no `IS_CHANGED`. ComfyUI's input-signature cache already covers every widget and upstream link, so re-runs with identical inputs are free and changing the seed re-rolls.
8. **No streaming in v2.** ComfyUI can't show partial images from this node type, and partial previews aren't billed anyway.
9. **Logging:** one line per request (model, n, ratio, resolution, reference count) and one per response (image count, cost). Never log base64 or the API key.

### Payload example
Character sheet branch: reference plus approved front view, 2:3 at 2K.
```json
{"model": "google/gemini-3.1-flash-image",
 "prompt": "Exact side profile, the character faces screen-left ...",
 "aspect_ratio": "2:3", "resolution": "2K", "seed": 201, "output_format": "png",
 "input_references": [
   {"type": "image_url", "image_url": {"url": "data:image/png;base64,<reference>"}},
   {"type": "image_url", "image_url": {"url": "data:image/png;base64,<front view>"}}]}
```

## Impact on `character_sheet.json`
- **Front view:** if the chosen model lists `n`, the 4 candidate nodes and 3 `ImageBatch` nodes collapse into one node with `n = 4` feeding Preview Chooser.
- **Branches:** each branch swaps `OpenRouterImageGenNode` for `OpenRouterImageAPINode`, with `aspect_ratio` `2:3`/`3:4` and `resolution` `2K`. The socket wiring (`prompt_input`, `image_1`, `image_2`) stays the same.

## Tests (`tests/test_image_api.py`, `unittest`, no network)
- **Payload builder:** defaults omit optional fields; `auto`/`default` are never sent; `size` is never sent; `prompt_input` overrides `prompt`.
- **Capability filter:** unsupported keys are dropped; an unknown model keeps them all.
- **References:** 2 sockets, one holding a 3-frame batch, give 4 `input_references` in order.
- **Response:** 2 images at different sizes decode to a `[2, H, W, 3]` batch at the first size; an empty `data` raises.
- **HTTP errors:** 400 and 502 raise `RuntimeError` with the API message (mocked `requests.post`).

## Manual verification (spends credit)
1. With `google/gemini-3.1-flash-image`, `n=1`, `2:3`, `2K`: check the output really is 2:3 at about 2K, and check `usage.cost` in Stats.
2. Same model with `n=4`: does it return 4 images, or is `n` dropped by capability filtering? Record which, because it decides the character-sheet front-view layout.
3. Two references (character + front view) → profile. Confirm both reach the provider through the request log.
4. Bad model or prompt → the node shows a red error and re-running retries without a seed change.

## Open questions
- Which image models list `n` and `seed` in `supported_parameters`? Check `/images/models` before relying on either in the workflow.
- Provider limits on how many `input_references` are accepted and how large they can be (not documented).
- Whether `/images/models` needs the API key or is public. Send the key if available, as for the chat model list.
