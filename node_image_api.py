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
from urllib.parse import quote

import numpy as np
import requests
import torch
from PIL import Image

import comfy.utils
from comfy_execution.graph_utils import ExecutionBlocker

from . import openrouter_shared as shared

OPTIONAL_PARAMS = ("n", "seed", "aspect_ratio", "resolution", "quality", "background", "output_format")
ENUM_CHECKS = ("aspect_ratio", "resolution", "quality", "background")


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
    failures = {}
    endpoints_cache = {}
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
                "on_error": (["block", "raise"], {"default": "block", "tooltip": "block: a failed request blocks only what depends on this node, and the rest of the run finishes. raise: the whole run stops."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
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
    def IS_CHANGED(cls, unique_id=None, **kwargs):
        """A blocked result is cached like any other; counting failures per node makes the next queue run it again."""
        return cls.failures.get(unique_id, 0)

    @classmethod
    def fetch_openrouter_models(cls):
        """Lists image model IDs and their supported_parameters via GET /api/v1/images/models; fetched in the background."""
        def load():
            api_key = shared.get_api_key()
            response = requests.get(
                f"{shared.BASE_URL}/images/models",
                headers=shared.build_standard_headers(api_key) if api_key else {},
                timeout=shared.DEFAULT_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            cls.model_params = {m["id"]: m.get("supported_parameters") or {} for m in data if m.get("id")}
            cls.models_cache = sorted(cls.model_params) or cls._fallback_models[:]

        return shared.cached_models(cls, load)

    @classmethod
    def VALIDATE_INPUTS(cls, model):
        return cls.models_cache is None or model in cls.models_cache or f"{model} is not in the OpenRouter model list"

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

    @classmethod
    def fetch_endpoints(cls, model, api_key):
        """Per-provider endpoints and supported_parameters via GET /api/v1/images/models/{id}/endpoints."""
        cached = cls.endpoints_cache.get(model)
        if cached and time.time() - cached[0] <= cls.cache_duration:
            return cached[1]
        response = requests.get(
            f"{shared.BASE_URL}/images/models/{quote(model, safe='/')}/endpoints",
            headers=shared.build_standard_headers(api_key),
            timeout=shared.DEFAULT_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        endpoints = response.json()["endpoints"]
        cls.endpoints_cache[model] = (time.time(), endpoints)
        return endpoints

    @staticmethod
    def accepts(descriptor, value):
        """Whether an endpoint descriptor takes the value. A missing descriptor does not; unknown types do."""
        if descriptor is None:
            return False
        if descriptor.get("type") == "enum":
            return value in descriptor.get("values", [])
        if descriptor.get("type") == "range":
            return descriptor.get("min", value) <= value <= descriptor.get("max", value)
        return True

    @staticmethod
    def allowed_values(endpoints, key):
        allowed = []
        for endpoint in endpoints:
            descriptor = endpoint["supported_parameters"].get(key) or {}
            if descriptor.get("type") == "enum":
                allowed.extend(map(str, descriptor.get("values", [])))
            elif descriptor.get("type") == "range":
                allowed.append(f"{descriptor.get('min')}-{descriptor.get('max')}")
        return ", ".join(dict.fromkeys(allowed)) or "not supported"

    @classmethod
    def route_endpoints(cls, model, checks, endpoints):
        """Returns provider tags to restrict routing to, None if every endpoint accepts the checks. Raises if none do."""
        compatible = [e for e in endpoints if all(cls.accepts(e["supported_parameters"].get(k), v) for k, v in checks)]
        if len(compatible) == len(endpoints):
            return None
        if not compatible:
            failed = [(k, v) for k, v in checks
                      if not any(cls.accepts(e["supported_parameters"].get(k), v) for e in endpoints)] or checks
            raise RuntimeError(f"{model} does not accept " + ", ".join(
                f"{k}={v} (allowed: {cls.allowed_values(endpoints, k)})" for k, v in failed))
        return sorted({e["provider_tag"] for e in compatible} - {e["provider_tag"] for e in endpoints if e not in compatible})

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
                              retries=1, on_error="block", unique_id=None, prompt_input=None, **kwargs):
        api_key = shared.get_api_key()
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
        if model in self.model_params:
            try:
                endpoints = await asyncio.to_thread(self.fetch_endpoints, model, api_key)
            except (requests.RequestException, KeyError, ValueError) as e:
                print(f"[ImageAPINode] Endpoint check skipped for {model}: {e}")
            else:
                checks = [(k, payloads[0][k]) for k in ENUM_CHECKS if k in payloads[0]]
                checks += [("n", count) for count in sorted({p["n"] for p in payloads if "n" in p})]
                if references:
                    checks.append(("input_references", len(references)))
                providers = self.route_endpoints(model, checks, endpoints)
                if providers:
                    print(f"[ImageAPINode] Routing {model} to: {', '.join(providers)}")
                    for p in payloads:
                        p["provider"] = {"only": providers}
        print(f"[ImageAPINode] Request: model={model} n={n} calls={len(payloads)} aspect_ratio={aspect_ratio} "
              f"resolution={resolution} references={len(references)}")

        # Async + worker threads let these calls and other independent image nodes run concurrently
        start_time = time.time()
        try:
            results = await asyncio.gather(*(shared.post_with_retries(f"{shared.BASE_URL}/images", api_key, p, validated_timeout, retries, "Image API") for p in payloads))
            elapsed = time.time() - start_time
            images = self.decode_images([item for r in results for item in (r.get("data") or [])])
        except (RuntimeError, requests.RequestException) as e:
            if on_error == "raise":
                raise
            self.failures[unique_id] = self.failures.get(unique_id, 0) + 1
            print(f"[ImageAPINode] Failed, blocking what depends on it: {e}")
            return (ExecutionBlocker(f"{model}: {e}"), f"FAILED: {e}", "", 0.0)
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
