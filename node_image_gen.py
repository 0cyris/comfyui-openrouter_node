"""
node_image_gen.py — OpenRouter Image Generation Node

Targets image-generation models on OpenRouter (e.g. DALL-E 3, Flux, Stable Diffusion).
Accepts a text prompt and optional reference images; outputs an IMAGE tensor.
The model dropdown is filtered to models whose output modality includes "image".
"""

import requests
import json
import time
import hashlib
import torch
from . import openrouter_shared as shared


class OpenRouterImageGenNode:
    """
    ComfyUI node for image generation via OpenRouter.

    Filters the model list to image-output models only.
    Returns three outputs:
      1) "image"  : the generated image tensor [1, H, W, 3]
      2) "Stats"  : tokens per second, prompt tokens, model used
      3) "Credits": remaining OpenRouter account balance
    """

    models_cache = None
    last_fetch_time = 0
    cache_duration = 3600
    default_request_timeout = shared.DEFAULT_REQUEST_TIMEOUT
    min_request_timeout = shared.MIN_REQUEST_TIMEOUT
    max_request_timeout = shared.MAX_REQUEST_TIMEOUT

    _fallback_models = [
        "error_fetching_models",
        "openai/dall-e-3",
        "black-forest-labs/flux-1-schnell",
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
                    "default": "A beautiful landscape"
                }),
                "model": (cls.fetch_openrouter_models(),),
                "web_search": ("BOOLEAN", {"default": False}),
                "cheapest": ("BOOLEAN", {"default": False}),
                "fastest": ("BOOLEAN", {"default": False}),
                "aspect_ratio": (shared.ASPECT_RATIO_OPTIONS, {"default": "auto"}),
                "image_resolution": (shared.IMAGE_RESOLUTION_OPTIONS, {"default": "1K"}),
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
            },
            "optional": {
                "prompt_input": ("STRING", {"forceInput": True}),
                # image_1..image_N are added dynamically by openrouter_dynamic_inputs.js
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "Stats", "Credits")

    FUNCTION = "generate_image"
    CATEGORY = "LLM"

    @classmethod
    def fetch_openrouter_models(cls):
        """Fetches image-output model IDs via GET /api/v1/models?output_modalities=image."""
        return shared.fetch_filtered_models(cls, "image", cls._fallback_models, "[ImageGenNode]")

    def generate_image(self, api_key, prompt, model,
                       web_search=False, cheapest=False, fastest=False,
                       aspect_ratio="auto", image_resolution="1K", seed=0,
                       request_timeout=120, prompt_input=None, **kwargs):
        """
        Sends an image generation request to OpenRouter.

        Optional reference images (image_1..image_N) are accepted via **kwargs
        and included as multimodal content blocks (for img2img capable models).

        Returns (image_tensor, stats_str, credits_str).
        """
        placeholder_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        # Resolve API key and prompt
        api_key = shared.get_api_key(api_key)
        if not api_key:
            return (
                placeholder_image,
                "Stats N/A",
                "Error: API Key not provided. Set LLM_KEY env var or use openrouter_api_key.json",
            )

        effective_prompt = (
            prompt_input if prompt_input is not None and prompt_input.strip()
            else prompt
        )

        validated_timeout = shared.validate_request_timeout(
            request_timeout, self.min_request_timeout,
            self.max_request_timeout, self.default_request_timeout
        )
        modified_model = shared.apply_model_modifiers(model, web_search, cheapest, fastest)
        headers = shared.build_standard_headers(api_key)

        # Build user content blocks: prompt text + any reference images
        user_content_blocks = [{"type": "text", "text": effective_prompt}]

        image_keys = sorted(
            [k for k in kwargs if k.startswith("image_")],
            key=lambda x: int(x.split("_")[1])
        )
        for key in image_keys:
            if kwargs[key] is not None:
                try:
                    img_str = shared.image_to_base64(kwargs[key])
                    user_content_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_str}"}
                    })
                except Exception as e:
                    return (
                        placeholder_image,
                        "Stats N/A",
                        f"Error processing {key}: {e}",
                    )

        has_ref_images = len(user_content_blocks) > 1
        messages = [{
            "role": "user",
            "content": user_content_blocks if has_ref_images else effective_prompt
        }]

        data = {
            "model": modified_model,
            "messages": messages,
            "seed": seed,
        }

        # Pass image dimensions via provider data when a specific size is requested
        size_tuple = shared.parse_image_size(aspect_ratio, image_resolution)
        if size_tuple:
            w, h = size_tuple
            data["provider"] = {"data": {"size": f"{w}x{h}"}}

        url = f"{shared.BASE_URL}/chat/completions"

        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=data, timeout=validated_timeout)
            response.raise_for_status()
            end_time = time.time()

            result = response.json()
            debug_str = json.dumps(result, default=str)
            print(f"[ImageGenNode] API response ({len(debug_str)} chars): {debug_str[:500]}")

            if not result.get("choices") or not result["choices"][0].get("message"):
                raise ValueError("Invalid response format from API: 'choices' or 'message' missing.")

            message = result["choices"][0]["message"]
            image_tensor = placeholder_image

            # Extract image from OpenRouter's images field
            if message.get("images"):
                print(f"[ImageGenNode] Found {len(message['images'])} image(s) in response")
                try:
                    first_image = message["images"][0]
                    image_url = first_image["image_url"]["url"]
                    if image_url.startswith("data:image"):
                        base64_str = image_url.split(",", 1)[1]
                        image_tensor = shared.base64_to_image(base64_str)
                    else:
                        print(f"[ImageGenNode] Unsupported image URL format: {image_url[:50]}...")
                except Exception as e:
                    print(f"[ImageGenNode] Error extracting image from response: {e}")
            else:
                # Fallback: check legacy multimodal content list
                content = message.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "image_url":
                            image_url = block["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                base64_str = image_url.split(",", 1)[1]
                                try:
                                    image_tensor = shared.base64_to_image(base64_str)
                                    break
                                except Exception as e:
                                    print(f"[ImageGenNode] Error decoding fallback image: {e}")
                else:
                    print("[ImageGenNode] No image found in response.")

            # Stats
            api_usage = result.get("usage", {})
            prompt_tokens = api_usage.get("prompt_tokens", 0)
            completion_tokens = api_usage.get("completion_tokens", 0)
            elapsed = end_time - start_time
            response_ms = result.get("response_ms")
            if response_ms and response_ms > 0:
                tps = completion_tokens / (response_ms / 1000.0)
            elif elapsed > 0:
                tps = completion_tokens / elapsed
            else:
                tps = 0

            stats = shared.build_stats_text(
                tps, prompt_tokens, completion_tokens, 0.0, modified_model,
                extra_parts=[f"Size: {size_tuple[0]}x{size_tuple[1]}"] if size_tuple else None
            )
            credits = shared.fetch_credits(api_key, validated_timeout)
            return (image_tensor, stats, credits)

        except requests.exceptions.RequestException as e:
            error_msg = f"API Request Error: {str(e)}"
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_msg += f" | Details: {e.response.json()}"
                except json.JSONDecodeError:
                    error_msg += f" | Status: {e.response.status_code}"
            else:
                error_msg += " (Network or connection issue)"
            print(f"[ImageGenNode] ERROR: {error_msg}")
            return (placeholder_image, "Stats N/A due to error", error_msg)
        except Exception as e:
            print(f"[ImageGenNode] ERROR: Node Error: {str(e)}")
            return (placeholder_image, "Stats N/A due to error", f"Node Error: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, api_key, prompt, model,
                   web_search=False, cheapest=False, fastest=False,
                   aspect_ratio="auto", image_resolution="1K", seed=0,
                   request_timeout=120, prompt_input=None, **kwargs):
        """Check if any input that affects the output has changed."""
        image_hashes = []
        for key in sorted(
            [k for k in kwargs if k.startswith("image_")],
            key=lambda x: int(x.split("_")[1])
        ):
            img = kwargs[key]
            if img is not None and isinstance(img, torch.Tensor):
                try:
                    h = hashlib.sha256(img.cpu().numpy().tobytes()).hexdigest()
                    image_hashes.append(h)
                except Exception:
                    image_hashes.append("hash_error")
            else:
                image_hashes.append(None)

        try:
            timeout_int = int(request_timeout)
            timeout_int = max(cls.min_request_timeout, min(cls.max_request_timeout, timeout_int))
        except (ValueError, TypeError):
            timeout_int = cls.default_request_timeout

        return (
            api_key, prompt, model, web_search, cheapest, fastest,
            aspect_ratio, image_resolution, seed, timeout_int,
            prompt_input, tuple(image_hashes)
        )


NODE_CLASS_MAPPINGS = {
    "OpenRouterImageGenNode": OpenRouterImageGenNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterImageGenNode": "OpenRouter Image Generation Node"
}
