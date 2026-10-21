"""
node_video_gen.py — OpenRouter Video Generation Node

Targets video-generation models on OpenRouter (e.g. RunwayML, Luma).
Accepts a text prompt and optional parameters; outputs a VIDEO object.
Handles async polling until the video is generated.
"""

import re
import requests
import time
import hashlib
import torch
import comfy.model_management
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
                "first_frame": ("IMAGE", {"forceInput": True}),
                "last_frame": ("IMAGE", {"forceInput": True}),
                # reference_1, reference_2, etc. are added dynamically by openrouter_dynamic_inputs.js
                "video_job_id": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Resume a job that was already submitted: polls and downloads it without submitting again. The prompt and media inputs are ignored.",
                }),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "Stats", "Credits")

    FUNCTION = "generate_video"
    CATEGORY = "LLM"

    @classmethod
    def fetch_openrouter_models(cls):
        """Fetches video-output model IDs via GET /api/v1/models?output_modalities=video."""
        return shared.fetch_filtered_models(cls, "video")

    @classmethod
    def VALIDATE_INPUTS(cls, model):
        return cls.models_cache is None or model in cls.models_cache or f"{model} is not in the OpenRouter model list"

    @staticmethod
    def build_payload(model, prompt, duration, seed, generate_audio, resolution,
                      aspect_ratio, first_frame, last_frame, references):
        data = {
            "model": model,
            "prompt": prompt,
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
            match = re.search(r'(\d+):(\d+)', aspect_ratio)
            if match:
                data["aspect_ratio"] = f"{match.group(1)}:{match.group(2)}"

        # Handle frame images (first/last frame for video generation)
        frame_images = []
        if first_frame is not None:
            frame_b64 = shared.image_to_base64(first_frame)
            frame_images.append({
                "type": "first_frame",
                "url": f"data:image/png;base64,{frame_b64}"
            })
            print("[VideoGenNode] Added first_frame for video generation")

        if last_frame is not None:
            frame_b64 = shared.image_to_base64(last_frame)
            frame_images.append({
                "type": "last_frame",
                "url": f"data:image/png;base64,{frame_b64}"
            })
            print("[VideoGenNode] Added last_frame for video generation")

        if frame_images:
            data["frame_images"] = frame_images

        # Handle reference images (for guidance/conditioning)
        input_references = []
        ref_keys = sorted(
            [k for k in references if k.startswith("reference_")],
            key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 0
        )
        for key in ref_keys:
            if references[key] is not None:
                ref_b64 = shared.image_to_base64(references[key])
                input_references.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{ref_b64}"}
                })
                print(f"[VideoGenNode] Added reference image from {key}")

        if input_references:
            data["input_references"] = input_references
        return data

    @staticmethod
    def check(response, action):
        """Raises with OpenRouter's own message and provider detail when a request failed."""
        if response.status_code < 400:
            return
        try:
            error = response.json()["error"]
            message = error["message"]
            metadata = error.get("metadata") or {}
        except (ValueError, KeyError, TypeError, AttributeError):
            message, metadata = response.text[:200], {}
        details = [f"{key}: {str(metadata[key])[:300]}" for key in ("provider_name", "reasons", "raw")
                   if isinstance(metadata, dict) and metadata.get(key)]
        raise RuntimeError(f"OpenRouter video {action} error {response.status_code}: {message}"
                           + (f" ({'; '.join(details)})" if details else ""))

    @staticmethod
    def wait(seconds):
        for _ in range(seconds):
            comfy.model_management.throw_exception_if_processing_interrupted()
            time.sleep(1)

    def poll(self, job_id, headers, timeout, poll_interval):
        """Returns the job's status data once it has completed."""
        start_time = time.time()
        for poll_count in range(1, max(1, timeout // poll_interval) + 1):
            comfy.model_management.throw_exception_if_processing_interrupted()
            response = requests.get(f"{shared.BASE_URL}/videos/{job_id}", headers=headers, timeout=timeout)
            self.check(response, "status")
            status_data = response.json()
            status = status_data.get("status")
            print(f"[VideoGenNode] Poll #{poll_count} - Status: {status}")

            if status == "completed":
                return status_data
            if status in ["failed", "cancelled", "expired"]:
                raise RuntimeError(f"Video generation {status}: {status_data.get('error', status)}")
            self.wait(poll_interval)
        raise RuntimeError(f"Video generation timed out after {time.time() - start_time:.1f}s")

    def generate_video(self, api_key, prompt, model,
                       web_search=False, cheapest=False, fastest=False,
                       aspect_ratio="auto", duration=6, resolution="720p",
                       seed=0, generate_audio=False, request_timeout=120,
                       poll_interval=5, prompt_input=None, first_frame=None,
                       last_frame=None, video_job_id="", **kwargs):
        """
        Submits a video generation request to OpenRouter and polls for completion.
        With video_job_id set, resumes that job instead of submitting a new one.

        Returns (video_object, stats_str, credits_str).
        """
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
        modified_model = shared.apply_model_modifiers(model, web_search, cheapest, fastest)
        headers = shared.build_standard_headers(api_key)

        start_time = time.time()
        job_id = video_job_id.strip()
        if job_id:
            print(f"[VideoGenNode] Resuming video job: {job_id}")
        else:
            data = self.build_payload(modified_model, effective_prompt, duration, seed, generate_audio,
                                      resolution, aspect_ratio, first_frame, last_frame, kwargs)
            print(f"[VideoGenNode] Submitting video generation request for model: {modified_model}")
            response = requests.post(f"{shared.BASE_URL}/videos", headers=headers, json=data, timeout=validated_timeout)
            self.check(response, "submit")
            result = response.json()
            job_id = result.get("id")
            if not job_id:
                raise RuntimeError("OpenRouter returned no job ID for the video request")
            print(f"[VideoGenNode] Received job ID: {job_id}, status: {result.get('status')}")

        try:
            status_data = self.poll(job_id, headers, validated_timeout, poll_interval)
            content_urls = status_data.get("unsigned_urls")
            if not content_urls:
                raise RuntimeError("Video generation completed but returned no video URL")

            # OpenRouter content endpoint requires the bearer token
            print(f"[VideoGenNode] Downloading video from: {content_urls[0][:100]}...")
            response = requests.get(content_urls[0], headers=headers, timeout=validated_timeout)
            self.check(response, "download")
            frames_tensor, video_metadata = shared.video_bytes_to_frames(response.content)
            video_fps = video_metadata["fps"]
            video_object = shared.build_video_object(
                frames_tensor, fps=video_fps, audio=None, metadata=video_metadata
            )
        except comfy.model_management.InterruptProcessingException:
            print(f"[VideoGenNode] Interrupted. Set video_job_id to {job_id} to resume.")
            raise
        except Exception as e:
            raise RuntimeError(f"{e} | Video job: {job_id}. Set video_job_id to this ID to resume.") from e

        elapsed = time.time() - start_time
        usage = status_data.get("usage", {})
        cost_usd = usage.get("USD", 0.0)
        frame_count = video_metadata.get("frame_count", 0)

        stats = (
            f"Job ID: {job_id}, "
            f"Frames: {frame_count}, "
            f"FPS: {video_fps:.1f}, "
            f"Time: {elapsed:.1f}s, "
            f"Cost: ${cost_usd:.3f}, "
            f"Model: {modified_model}"
        )

        credits = shared.fetch_credits(api_key, validated_timeout)
        return (video_object, stats, credits)

    @classmethod
    def IS_CHANGED(cls, api_key, prompt, model,
                   web_search=False, cheapest=False, fastest=False,
                   aspect_ratio="auto", duration=6, resolution="720p",
                   seed=0, generate_audio=False, request_timeout=120,
                   poll_interval=5, prompt_input=None, first_frame=None,
                   last_frame=None, video_job_id="", **kwargs):
        """Check if any input that affects the output has changed."""
        import hashlib

        try:
            timeout_int = int(request_timeout)
            timeout_int = max(cls.min_request_timeout, min(cls.max_request_timeout, timeout_int))
        except (ValueError, TypeError):
            timeout_int = cls.default_request_timeout

        # Hash frame images
        frame_hashes = []
        for frame in [first_frame, last_frame]:
            if frame is not None and isinstance(frame, torch.Tensor):
                try:
                    h = hashlib.sha256(frame.cpu().numpy().tobytes()).hexdigest()
                    frame_hashes.append(h)
                except Exception:
                    frame_hashes.append("hash_error")
            else:
                frame_hashes.append(None)

        # Hash reference images
        ref_keys = sorted(
            [k for k in kwargs if k.startswith("reference_")],
            key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 0
        )
        for key in ref_keys:
            img = kwargs[key]
            if img is not None and isinstance(img, torch.Tensor):
                try:
                    h = hashlib.sha256(img.cpu().numpy().tobytes()).hexdigest()
                    frame_hashes.append(h)
                except Exception:
                    frame_hashes.append("hash_error")
            else:
                frame_hashes.append(None)

        return (
            api_key, prompt, model, web_search, cheapest, fastest,
            aspect_ratio, duration, resolution, seed, generate_audio,
            timeout_int, prompt_input, tuple(frame_hashes), video_job_id
        )


NODE_CLASS_MAPPINGS = {
    "OpenRouterVideoGenNode": OpenRouterVideoGenNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterVideoGenNode": "OpenRouter Video Generation Node"
}
