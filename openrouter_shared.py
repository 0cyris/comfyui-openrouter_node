"""
openrouter_shared.py — Shared infrastructure for all OpenRouter ComfyUI nodes.

Provides: API key resolution, model fetching/filtering, credit checking,
image and audio conversion helpers, token counting, and request utilities.
All node classes import from this module to avoid duplicating logic.
"""

import requests
import json
import time
import base64
import io
import os
import re
import numpy as np
import torch
import tiktoken
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_REQUEST_TIMEOUT = 120
MIN_REQUEST_TIMEOUT = 1
MAX_REQUEST_TIMEOUT = 3600
CACHE_DURATION = 3600  # seconds

REASONING_EFFORT_OPTIONS = ("auto", "none", "minimal", "low", "medium", "high", "xhigh")
DEFAULT_REASONING_EFFORT = "auto"

ASPECT_RATIO_OPTIONS = [
    "auto",
    "1:1 (1024x1024)",
    "2:3 (832x1248)",
    "3:2 (1248x832)",
    "3:4 (864x1184)",
    "4:3 (1184x864)",
    "4:5 (896x1152)",
    "5:4 (1152x896)",
    "9:16 (768x1344)",
    "16:9 (1344x768)",
    "21:9 (1536x672)",
    "1:4 (google/gemini-3.1-flash-image-preview (Nano Banana 2) only)",
    "4:1 (google/gemini-3.1-flash-image-preview (Nano Banana 2) only)",
    "1:8 (google/gemini-3.1-flash-image-preview (Nano Banana 2) only)",
    "8:1 (google/gemini-3.1-flash-image-preview (Nano Banana 2) only)",
]
IMAGE_RESOLUTION_OPTIONS = ["1K", "2K", "4K"]
RESOLUTION_MULTIPLIERS = {"1K": 1, "2K": 2, "4K": 4}

VOICE_OPTIONS = ("alloy", "echo", "fable", "onyx", "nova", "shimmer")
AUDIO_FORMAT_OPTIONS = ("mp3", "pcm", "wav", "opus", "aac", "flac")

# ── Module-level raw model cache ──────────────────────────────────────────────
# One network fetch, shared across all node classes.

_raw_models_cache = None   # list[dict] from /api/v1/models
_last_fetch_time = 0.0
_cache_was_refreshed = False  # set True each time the cache is repopulated

# ── Optional audio libraries ──────────────────────────────────────────────────

try:
    import torchaudio as _torchaudio
    _HAS_TORCHAUDIO = True
except ImportError:
    _torchaudio = None
    _HAS_TORCHAUDIO = False

try:
    import soundfile as _soundfile
    _HAS_SOUNDFILE = True
except ImportError:
    _soundfile = None
    _HAS_SOUNDFILE = False


# ── API key resolution ────────────────────────────────────────────────────────

def get_api_key(api_key_ui):
    """
    Resolves the API key from (in priority order):
    1. UI input field (if non-empty)
    2. Environment variable 'LLM_KEY'
    3. JSON config file 'openrouter_api_key.json' in this module's directory
    """
    if api_key_ui and api_key_ui.strip():
        return api_key_ui.strip()

    env_key = os.environ.get("LLM_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "openrouter_api_key.json"
    )
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                file_key = config.get("api_key")
                if file_key and file_key.strip():
                    return file_key.strip()
        except Exception as e:
            print(f"Error reading openrouter_api_key.json: {e}")

    return ""


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def build_standard_headers(api_key):
    """Returns the standard HTTP headers for OpenRouter API requests."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/comfyui-openrouter",
        "X-Title": "ComfyUI OpenRouter LLM Node",
    }


# ── Model fetching & filtering ────────────────────────────────────────────────

def fetch_all_models_raw():
    """
    Fetches the full model list from the OpenRouter API with 1-hour caching.

    Returns:
        (model_list, was_refreshed)
        - model_list: list of model dicts from /api/v1/models
        - was_refreshed: True when the cache was just repopulated this call,
          signalling per-node filtered caches to rebuild.
    """
    global _raw_models_cache, _last_fetch_time, _cache_was_refreshed
    current_time = time.time()
    _cache_was_refreshed = False
    if _raw_models_cache is None or (current_time - _last_fetch_time > CACHE_DURATION):
        try:
            response = requests.get(
                f"{BASE_URL}/models",
                timeout=DEFAULT_REQUEST_TIMEOUT
            )
            response.raise_for_status()
            _raw_models_cache = response.json()["data"]
            _last_fetch_time = current_time
            _cache_was_refreshed = True
        except requests.exceptions.RequestException as e:
            print(f"[openrouter_shared] Error fetching models: {e}")
            if _raw_models_cache is None:
                _raw_models_cache = []
    return _raw_models_cache, _cache_was_refreshed


def filter_models_by_output(models, output_modality):
    """
    Filters model dicts to those whose output modality matches.

    OpenRouter architecture.modality format: "input->output"
    Examples: "text->text", "text+image->text", "text->image", "text->audio"

    Args:
        models: list of model dicts from fetch_all_models_raw()
        output_modality: "text", "image", or "audio"

    Returns:
        Sorted list of matching model ID strings.
    """
    result = []
    for m in models:
        arch = m.get("architecture", {})
        modality_str = arch.get("modality", "")
        if "->" in modality_str:
            output_part = modality_str.split("->", 1)[1]
        else:
            output_part = modality_str
        if output_modality in output_part:
            result.append(m["id"])
    return sorted(result)


# ── Credits ───────────────────────────────────────────────────────────────────

def fetch_credits(api_key, timeout=None):
    """
    Fetches credit balance from the OpenRouter API.
    Returns a formatted string: "Remaining: $X.XXX"
    """
    if not api_key:
        return "API Key not provided."

    url = f"{BASE_URL}/credits"
    headers = build_standard_headers(api_key)

    try:
        validated_timeout = validate_request_timeout(
            timeout if timeout is not None else DEFAULT_REQUEST_TIMEOUT
        )
        response = requests.get(url, headers=headers, timeout=validated_timeout)
        response.raise_for_status()

        result = response.json()
        if (
            "data" in result
            and "total_credits" in result["data"]
            and "total_usage" in result["data"]
        ):
            total_credits = result["data"]["total_credits"]
            total_usage = result["data"]["total_usage"]
            remaining = total_credits - total_usage
            return f"Remaining: ${remaining:.3f}"
        else:
            return "Could not parse credit data from response."

    except requests.exceptions.RequestException as e:
        error_message = f"Error fetching credits: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            error_message += (
                f" | Status Code: {e.response.status_code}"
                f" | Response: {e.response.text[:200]}"
            )
        return error_message
    except json.JSONDecodeError:
        return "Error fetching credits: Could not decode JSON response."


# ── Validation ────────────────────────────────────────────────────────────────

def validate_temperature(temperature):
    """Clamps temperature to [0.0, 2.0]."""
    try:
        temp = float(temperature)
        return max(0.0, min(2.0, temp))
    except (ValueError, TypeError):
        return 1.0


def validate_request_timeout(request_timeout,
                              min_rt=MIN_REQUEST_TIMEOUT,
                              max_rt=MAX_REQUEST_TIMEOUT,
                              default_rt=DEFAULT_REQUEST_TIMEOUT):
    """Clamps request timeout to [min_rt, max_rt]."""
    try:
        timeout = int(request_timeout)
        return max(min_rt, min(max_rt, timeout))
    except (ValueError, TypeError):
        return default_rt


def validate_reasoning_effort(reasoning_effort,
                               options=REASONING_EFFORT_OPTIONS,
                               default=DEFAULT_REASONING_EFFORT):
    """Validates OpenRouter reasoning effort against known options."""
    if isinstance(reasoning_effort, str):
        normalized = reasoning_effort.strip().lower()
        if normalized in options:
            return normalized
    return default


# ── Model modifiers ───────────────────────────────────────────────────────────

def apply_model_modifiers(model, web_search, cheapest, fastest):
    """
    Applies OpenRouter routing modifiers to the model ID:
      :online  — enables web search (web_search=True)
      :floor   — routes to cheapest provider (cheapest=True, default)
      :nitro   — routes to fastest provider (fastest=True, overrides floor)
    """
    modified_model = model
    if web_search and ":online" not in modified_model:
        modified_model = f"{modified_model}:online"
    if ":online" not in modified_model:
        if cheapest and ":floor" not in modified_model:
            modified_model = f"{modified_model}:floor"
        elif fastest and not cheapest and ":nitro" not in modified_model:
            modified_model = f"{modified_model}:nitro"
    return modified_model


# ── Image helpers ─────────────────────────────────────────────────────────────

def image_to_base64(image):
    """
    Converts a ComfyUI IMAGE tensor (BHWC, float 0-1) to a base64 PNG string.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input 'image' is not a torch.Tensor")

    if image.ndim == 4:
        if image.shape[0] != 1:
            print(f"Warning: Image batch size is {image.shape[0]}, using only the first image.")
        image = image.squeeze(0)  # HWC

    if image.ndim != 3:
        raise ValueError(f"Unexpected image dimensions: {image.shape}. Expected HWC.")

    image_np = image.cpu().numpy()
    if image_np.dtype != np.uint8:
        if image_np.min() < 0 or image_np.max() > 1:
            print("Warning: Image tensor values outside [0, 1] range. Clamping.")
            image_np = np.clip(image_np, 0, 1)
        image_np = (image_np * 255).astype(np.uint8)

    pil_image = Image.fromarray(image_np, 'RGB')
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def base64_to_image(base64_str):
    """
    Converts a base64 image string to a ComfyUI image tensor [1, H, W, 3], values in [0, 1].
    Returns a 64×64 zero tensor on failure.
    """
    try:
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        img = img.convert("RGB")
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # [1, H, W, 3]
        print(f"Successfully converted base64 to image tensor: {img_tensor.shape}")
        return img_tensor
    except Exception as e:
        print(f"Error in base64_to_image: {e}")
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


# ── Image size parsing ────────────────────────────────────────────────────────

def parse_image_size(aspect_ratio, image_resolution):
    """
    Parses pixel dimensions from an ASPECT_RATIO_OPTIONS dropdown value and
    applies the IMAGE_RESOLUTION_OPTIONS multiplier.

    Returns (width, height) or None for "auto" / entries without pixel dims.
    """
    if aspect_ratio == "auto":
        return None
    match = re.search(r'\((\d+)x(\d+)\)', aspect_ratio)
    if not match:
        return None  # Gemini-specific ratios without pixel dims
    base_w = int(match.group(1))
    base_h = int(match.group(2))
    mult = RESOLUTION_MULTIPLIERS.get(image_resolution, 1)
    return (base_w * mult, base_h * mult)


# ── Token counting ────────────────────────────────────────────────────────────

def count_tokens(text, model):
    """
    Counts tokens using tiktoken. Falls back to a character-based estimate.
    """
    if not text or not isinstance(text, str):
        return 0

    base_model = model.split(':')[0] if ':' in model else model
    encoding_name = "cl100k_base"

    try:
        cl100k_models = [
            "openai/gpt-4", "openai/gpt-3.5", "openai/gpt-4o",
            "anthropic/claude",
            "google/gemini",
            "meta-llama/llama-2", "meta-llama/llama-3",
            "mistralai/mistral", "mistralai/mixtral",
        ]
        if any(base_model.startswith(p) for p in cl100k_models):
            encoding_name = "cl100k_base"

        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text, disallowed_special=()))
    except Exception as e:
        print(
            f"Warning: Tiktoken error for model '{model}' "
            f"(encoding: '{encoding_name}'): {e}. Falling back to estimation."
        )
        return max(1, round(len(text) / 4))


# ── Stats formatting ──────────────────────────────────────────────────────────

def build_stats_text(tps, prompt_tokens, completion_tokens, validated_temp,
                     modified_model, extra_parts=None):
    """
    Formats the standard Stats output string.

    Args:
        extra_parts: optional list of additional "Key: value" strings to append.
    """
    text = (
        f"TPS: {tps:.2f}, "
        f"Prompt Tokens: {prompt_tokens}, "
        f"Completion Tokens: {completion_tokens}, "
        f"Temp: {validated_temp:.1f}, "
        f"Model: {modified_model}"
    )
    if extra_parts:
        text += ", " + ", ".join(extra_parts)
    return text


# ── Audio decoding ────────────────────────────────────────────────────────────

def decode_audio_bytes(audio_bytes, fmt):
    """
    Decodes raw audio bytes into a ComfyUI AUDIO dict.

    Returns {"waveform": Tensor[1, C, N], "sample_rate": int}.

    PCM is handled first because it is headerless raw samples — container
    decoders (torchaudio, soundfile, wave) cannot parse it.

    Fallback chain for all other formats:
    1. torchaudio (if installed) — handles mp3/wav/flac/opus/aac
    2. soundfile (if installed)  — handles wav/flac/ogg
    3. stdlib wave               — WAV only
    4. Silent 1-second mono placeholder at 44100 Hz
    """
    # ── PCM: raw 16-bit signed integer samples, mono, 24000 Hz ───────────────
    # OpenAI-compatible TTS endpoints return headerless little-endian int16 PCM.
    # No container decoder can parse this; convert directly.
    if fmt == "pcm":
        try:
            pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            # [N] → [1, 1, N]  (batch=1, channels=1, samples=N)
            waveform = torch.from_numpy(pcm).unsqueeze(0).unsqueeze(0)
            return {"waveform": waveform, "sample_rate": 24000}
        except Exception as e:
            print(f"[openrouter_shared] PCM decode failed: {e}")
            return {"waveform": torch.zeros((1, 1, 44100), dtype=torch.float32), "sample_rate": 44100}

    buf = io.BytesIO(audio_bytes)

    if _HAS_TORCHAUDIO:
        try:
            buf.seek(0)
            waveform, sample_rate = _torchaudio.load(buf)
            # waveform: [C, N] → [1, C, N]
            return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        except Exception as e:
            print(f"[openrouter_shared] torchaudio decode failed: {e}")

    if _HAS_SOUNDFILE:
        try:
            buf.seek(0)
            data, sample_rate = _soundfile.read(buf, dtype="float32", always_2d=True)
            # data: [N, C] → [C, N] → [1, C, N]
            waveform = torch.from_numpy(data.T).unsqueeze(0)
            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception as e:
            print(f"[openrouter_shared] soundfile decode failed: {e}")

    if fmt == "wav":
        try:
            import wave
            buf.seek(0)
            with wave.open(buf) as wf:
                n_channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            waveform = torch.from_numpy(
                pcm.reshape(-1, n_channels).T  # [C, N]
            ).unsqueeze(0)  # [1, C, N]
            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception as e:
            print(f"[openrouter_shared] wave stdlib decode failed: {e}")

    print("[openrouter_shared] All audio decode methods failed; returning silent placeholder")
    return {"waveform": torch.zeros((1, 1, 44100), dtype=torch.float32), "sample_rate": 44100}
