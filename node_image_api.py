"""
node_image_api.py — OpenRouter Image API Node

Generates images through POST /api/v1/images (not chat completions).
Supports n images per call, aspect_ratio / resolution controls, and drops
parameters the selected model does not list in supported_parameters.
"""

import asyncio
import base64
import io
import time

import numpy as np
import requests
import torch
from PIL import Image

import comfy.utils

from . import openrouter_shared as shared

RETRY_STATUSES = (408, 429, 500, 502, 503, 504)
OPTIONAL_PARAMS = ("n", "seed", "aspect_ratio", "resolution", "quality", "background", "output_format")


class OpenRouterImageAPINode:
    """
    ComfyUI node for the OpenRouter Image API.

    Returns four outputs:
      1) "images" : every returned image as one batch [B, H, W, 3]
      2) "Stats"  : model, image count, output size, elapsed time, cost
      3) "Credits": remaining OpenRouter account balance
      4) "cost"   : summed usage.cost of the calls as a float (0.0 if none reported)
    Errors raise RuntimeError so ComfyUI shows them and caches nothing.
    """

    models_cache = None
    model_params = {}
    last_fetch_time = 0
    cache_duration = shared.CACHE_DURATION
    default_request_timeout = shared.DEFAULT_REQUEST_TIMEOUT
    min_request_timeout = shared.MIN_REQUEST_TIMEOUT
    max_request_timeout = shared.MAX_REQUEST_TIMEOUT

    _fallback_models = [
        "google/gemini-3.1-flash-image",
        "openai/gpt-image-1",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "model": (cls.fetch_openrouter_models(),),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "aspect_ratio": (["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9",
                                  "21:9", "9:21", "1:2", "2:1", "1:4", "4:1", "1:8", "8:1"], {"default": "auto"}),
                "resolution": (["default", "512", "768", "1K", "1.5K", "2K", "4K"], {"default": "default"}),
                "quality": (["auto", "low", "medium", "high"], {"default": "auto"}),
                "background": (["auto", "transparent", "opaque"], {"default": "auto"}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": "fixed"
                }),
                "request_timeout": ("INT", {
                    "default": cls.default_request_timeout,
                    "min": cls.min_request_timeout,
                    "max": cls.max_request_timeout,
                    "step": 1,
                    "display": "number",
                }),
                "retries": ("INT", {"default": 1, "min": 0, "max": 5}),
            },
            "optional": {
                "prompt_input": ("STRING", {"forceInput": True}),
                # image_1..image_N are added dynamically by openrouter_dynamic_inputs.js
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("images", "Stats", "Credits", "cost")

    FUNCTION = "generate_images"
    CATEGORY = "LLM"

    @classmethod
    def fetch_openrouter_models(cls):
        """Fetches image model IDs and their supported_parameters via GET /api/v1/images/models."""
        current_time = time.time()
        if cls.models_cache is None or (current_time - cls.last_fetch_time > cls.cache_duration):
            try:
                api_key = shared.get_api_key("")
                response = requests.get(
                    f"{shared.BASE_URL}/images/models",
                    headers=shared.build_standard_headers(api_key) if api_key else {},
                    timeout=shared.DEFAULT_REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json().get("data", [])
                cls.model_params = {m["id"]: m.get("supported_parameters") or {} for m in data if m.get("id")}
                cls.models_cache = sorted(cls.model_params) or cls._fallback_models[:]
                cls.last_fetch_time = current_time
            except Exception as e:
                print(f"[ImageAPINode] Error fetching image models: {e}")
                if cls.models_cache is None:
                    cls.models_cache = cls._fallback_models[:]
        return cls.models_cache

    @classmethod
    def build_payload(cls, model, prompt, n, aspect_ratio, resolution, quality, background, seed, references):
        payload = {"model": model, "prompt": prompt, "seed": seed, "output_format": "png"}
        if n > 1:
            payload["n"] = n
        if aspect_ratio != "auto":
            payload["aspect_ratio"] = aspect_ratio
        if resolution != "default":
            payload["resolution"] = resolution
        if quality != "auto":
            payload["quality"] = quality
        if background != "auto":
            payload["background"] = background
        if references:
            payload["input_references"] = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                for b64 in references
            ]
        return payload

    @classmethod
    def drop_unsupported(cls, payload):
        """Removes optional fields missing from the model's supported_parameters. Unknown models keep everything."""
        supported = cls.model_params.get(payload["model"])
        if supported is None:
            return []
        dropped = [k for k in OPTIONAL_PARAMS if k in payload and k not in supported]
        for k in dropped:
            del payload[k]
        return dropped

    @classmethod
    def max_images_per_call(cls, model):
        """The model's n range max from supported_parameters; 1 if n is unsupported, 10 for unknown models."""
        supported = cls.model_params.get(model)
        if supported is None:
            return 10
        if "n" not in supported:
            return 1
        return supported["n"].get("max", 10)

    @staticmethod
    async def post_request(api_key, payload, timeout, retries=0):
        for attempt in range(retries + 1):
            try:
                response = await asyncio.to_thread(
                    requests.post,
                    f"{shared.BASE_URL}/images",
                    headers=shared.build_standard_headers(api_key),
                    json=payload,
                    timeout=timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                error, transient = e, True
            else:
                if response.status_code < 400:
                    return response.json()
                metadata = {}
                try:
                    body = response.json()["error"]
                    message = body["message"]
                    metadata = body.get("metadata") or {}
                except (ValueError, KeyError, TypeError, AttributeError):
                    message = response.text[:200]
                details = []
                if isinstance(metadata, dict):
                    if metadata.get("provider_name"):
                        details.append(f"provider: {metadata['provider_name']}")
                    if metadata.get("reasons"):
                        reasons = metadata["reasons"]
                        details.append(f"reasons: {', '.join(map(str, reasons)) if isinstance(reasons, list) else reasons}")
                    if metadata.get("raw"):
                        details.append(f"raw: {str(metadata['raw'])[:300]}")
                    moderated = "reasons" in metadata or "flagged_input" in metadata
                else:
                    moderated = False
                error = RuntimeError(f"OpenRouter Image API error {response.status_code}: {message}"
                                     + (f" ({'; '.join(details)})" if details else ""))
                transient = response.status_code in RETRY_STATUSES or moderated or "moderation" in str(message).lower()
            if not transient or attempt == retries:
                raise error
            print(f"[ImageAPINode] Retrying ({attempt + 1}/{retries}) after: {error}")
            await asyncio.sleep(2 * (attempt + 1))

    @staticmethod
    def decode_images(data):
        if not data:
            raise RuntimeError("OpenRouter returned no images")
        frames = []
        for item in data:
            img = Image.open(io.BytesIO(base64.b64decode(item["b64_json"]))).convert("RGB")
            frames.append(torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0))
        height, width = frames[0].shape[1:3]
        for i, frame in enumerate(frames):
            if frame.shape[1:3] != (height, width):
                frame = comfy.utils.common_upscale(frame.movedim(-1, 1), width, height, "lanczos", "disabled")
                frames[i] = frame.movedim(1, -1)
        return torch.cat(frames, dim=0)

    async def generate_images(self, api_key, prompt, model, n=1, aspect_ratio="auto", resolution="default",
                              quality="auto", background="auto", seed=0, request_timeout=120,
                              retries=1, prompt_input=None, **kwargs):
        api_key = shared.get_api_key(api_key)
        if not api_key:
            raise RuntimeError("API Key not provided. Set LLM_KEY env var or use openrouter_api_key.json")

        effective_prompt = (
            prompt_input if prompt_input is not None and prompt_input.strip()
            else prompt
        )
        validated_timeout = shared.validate_request_timeout(
            request_timeout, self.min_request_timeout,
            self.max_request_timeout, self.default_request_timeout
        )

        references = []
        for key in sorted((k for k in kwargs if k.startswith("image_")), key=lambda x: int(x.split("_")[1])):
            if kwargs[key] is not None:
                references.extend(await asyncio.to_thread(shared.image_batch_to_base64_list, kwargs[key]))

        # Split n into calls the model accepts (e.g. Gemini allows only n=1); each call gets its own seed
        per_call = min(n, self.max_images_per_call(model))
        counts = [per_call] * (n // per_call) + ([n % per_call] if n % per_call else [])
        payloads = [self.build_payload(model, effective_prompt, count, aspect_ratio, resolution,
                                       quality, background, seed + i, references)
                    for i, count in enumerate(counts)]
        dropped = [self.drop_unsupported(p) for p in payloads][0]
        if dropped:
            print(f"[ImageAPINode] {model} does not support: {', '.join(dropped)}. Not sending.")
        print(f"[ImageAPINode] Request: model={model} n={n} calls={len(payloads)} aspect_ratio={aspect_ratio} "
              f"resolution={resolution} references={len(references)}")

        # Async + worker threads let these calls and other independent image nodes run concurrently
        start_time = time.time()
        results = await asyncio.gather(*(self.post_request(api_key, p, validated_timeout, retries) for p in payloads))
        elapsed = time.time() - start_time

        images = self.decode_images([item for r in results for item in (r.get("data") or [])])
        costs = [r.get("usage", {}).get("cost") for r in results]
        cost = sum(c for c in costs if c is not None) if any(c is not None for c in costs) else None
        print(f"[ImageAPINode] Response: {images.shape[0]} image(s), cost={cost}")

        stats = (
            f"Model: {model}, Images: {images.shape[0]}, "
            f"Size: {images.shape[2]}x{images.shape[1]}, "
            f"Time: {elapsed:.1f}s, Cost: {cost}"
        )
        credits = await asyncio.to_thread(shared.fetch_credits, api_key, validated_timeout)
        return (images, stats, credits, cost or 0.0)


NODE_CLASS_MAPPINGS = {
    "OpenRouterImageAPINode": OpenRouterImageAPINode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterImageAPINode": "OpenRouter Image API Node"
}
