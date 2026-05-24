"""
node_video_gen.py — OpenRouter Video Generation Node

Targets video-generation models on OpenRouter (e.g. RunwayML, Luma).
Accepts a text prompt and optional parameters; outputs a video file path or URL.
Handles async polling until the video is generated.
"""

import requests
import json
import time
import hashlib
import torch
from . import openrouter_shared as shared


class OpenRouterVideoGenNode:
    """
    ComfyUI node for video generation via OpenRouter.

    Filters the model list to video-output models only.
    Handles async job submission and polling.
    Returns three outputs:
      1) "video_path" : the generated video file path or URL
      2) "Stats"      : job ID, status, generation time
      3) "Credits"    : remaining OpenRouter account balance
    """

    models_cache = None
    last_fetch_time = 0
    cache_duration = 3600
    default_request_timeout = shared.DEFAULT_REQUEST_TIMEOUT
    min_request_timeout = shared.MIN_REQUEST_TIMEOUT
    max_request_timeout = shared.MAX_REQUEST_TIMEOUT

    _fallback_models = [
        "error_fetching_models",
        "runwayml/gen-3-alpha",
        "luma/dream-machine",
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
                    "default": "A beautiful landscape with mountains and a sunset"
                }),
                "model": (cls.fetch_openrouter_models(),),
                "web_search": ("BOOLEAN", {"default": False}),
                "cheapest": ("BOOLEAN", {"default": False}),
                "fastest": ("BOOLEAN", {"default": False}),
                "aspect_ratio": (shared.ASPECT_RATIO_OPTIONS, {"default": "auto"}),
                "duration": ("INT", {
                    "default": 6,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "display": "number",
                }),
                "resolution": ("STRING", {
                    "default": "720p",
                    "multiline": False,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": "fixed"
                }),
                "generate_audio": ("BOOLEAN", {"default": False}),
                "request_timeout": ("INT", {
                    "default": cls.default_request_timeout,
                    "min": cls.min_request_timeout,
                    "max": cls.max_request_timeout,
                    "step": 1,
                    "display": "number",
                }),
                "poll_interval": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "display": "number",
                }),
            },
            "optional": {
                "prompt_input": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_path", "Stats", "Credits")

    FUNCTION = "generate_video"
    CATEGORY = "LLM"

    @classmethod
    def fetch_openrouter_models(cls):
        """Fetches video-output model IDs via GET /api/v1/models?output_modalities=video."""
        return shared.fetch_filtered_models(cls, "video", cls._fallback_models, "[VideoGenNode]")

    def generate_video(self, api_key, prompt, model,
                       web_search=False, cheapest=False, fastest=False,
                       aspect_ratio="auto", duration=6, resolution="720p",
                       seed=0, generate_audio=False, request_timeout=120,
                       poll_interval=5, prompt_input=None):
        """
        Submits a video generation request to OpenRouter and polls for completion.

        Returns (video_path_or_url, stats_str, credits_str).
        """
        error_placeholder = "video_generation_failed"

        # Resolve API key and prompt
        api_key = shared.get_api_key(api_key)
        if not api_key:
            return (
                error_placeholder,
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

        # Build request data
        data = {
            "model": modified_model,
            "prompt": effective_prompt,
        }

        # Add optional parameters
        if duration:
            data["duration"] = duration
        if seed != 0:
            data["seed"] = seed
        if generate_audio:
            data["generate_audio"] = generate_audio
        if resolution:
            data["resolution"] = resolution

        # Handle aspect ratio if specified
        if aspect_ratio != "auto":
            # Try to extract aspect ratio string
            import re
            match = re.search(r'(\d+):(\d+)', aspect_ratio)
            if match:
                data["aspect_ratio"] = f"{match.group(1)}:{match.group(2)}"

        url = f"{shared.BASE_URL}/videos"

        try:
            # Submit video generation request
            print(f"[VideoGenNode] Submitting video generation request for model: {modified_model}")
            start_time = time.time()
            response = requests.post(url, headers=headers, json=data, timeout=validated_timeout)
            response.raise_for_status()

            result = response.json()
            job_id = result.get("id")
            if not job_id:
                raise ValueError("No job ID returned from API")

            print(f"[VideoGenNode] Received job ID: {job_id}, status: {result.get('status')}")

            # Poll for completion
            video_url = None
            poll_count = 0
            max_polls = max(1, validated_timeout // poll_interval)

            while poll_count < max_polls:
                poll_count += 1
                time.sleep(poll_interval)

                status_url = f"{shared.BASE_URL}/videos/{job_id}"
                status_response = requests.get(status_url, headers=headers, timeout=validated_timeout)
                status_response.raise_for_status()
                status_data = status_response.json()
                status = status_data.get("status")

                print(f"[VideoGenNode] Poll #{poll_count} - Status: {status}")

                if status == "completed":
                    # Get video content URL
                    content_urls = status_data.get("unsigned_urls", [])
                    if content_urls:
                        video_url = content_urls[0]
                        print(f"[VideoGenNode] Video generated successfully: {video_url[:100]}...")
                    break
                elif status in ["failed", "cancelled", "expired"]:
                    error_msg = status_data.get("error", status)
                    raise ValueError(f"Video generation {status}: {error_msg}")

            end_time = time.time()
            elapsed = end_time - start_time

            if not video_url:
                raise ValueError(f"Video generation timed out after {elapsed:.1f}s")

            # Stats
            usage = status_data.get("usage", {})
            cost_usd = usage.get("USD", 0.0)

            stats = (
                f"Job ID: {job_id}, "
                f"Time: {elapsed:.1f}s, "
                f"Duration: {duration}s, "
                f"Cost: ${cost_usd:.3f}, "
                f"Model: {modified_model}"
            )

            credits = shared.fetch_credits(api_key, validated_timeout)
            return (video_url, stats, credits)

        except requests.exceptions.RequestException as e:
            error_msg = f"API Request Error: {str(e)}"
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_msg += f" | Details: {e.response.json()}"
                except json.JSONDecodeError:
                    error_msg += f" | Status: {e.response.status_code}"
            else:
                error_msg += " (Network or connection issue)"
            print(f"[VideoGenNode] ERROR: {error_msg}")
            return (error_placeholder, "Stats N/A due to error", error_msg)
        except Exception as e:
            print(f"[VideoGenNode] ERROR: {str(e)}")
            return (error_placeholder, "Stats N/A due to error", f"Node Error: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, api_key, prompt, model,
                   web_search=False, cheapest=False, fastest=False,
                   aspect_ratio="auto", duration=6, resolution="720p",
                   seed=0, generate_audio=False, request_timeout=120,
                   poll_interval=5, prompt_input=None):
        """Check if any input that affects the output has changed."""
        try:
            timeout_int = int(request_timeout)
            timeout_int = max(cls.min_request_timeout, min(cls.max_request_timeout, timeout_int))
        except (ValueError, TypeError):
            timeout_int = cls.default_request_timeout

        return (
            api_key, prompt, model, web_search, cheapest, fastest,
            aspect_ratio, duration, resolution, seed, generate_audio,
            timeout_int, prompt_input
        )


NODE_CLASS_MAPPINGS = {
    "OpenRouterVideoGenNode": OpenRouterVideoGenNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterVideoGenNode": "OpenRouter Video Generation Node"
}
