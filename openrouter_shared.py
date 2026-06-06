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


def filter_models_by_input(models, input_modality):
    """
    Filters model dicts to those whose input modality matches.

    OpenRouter architecture.modality format: "input->output"
    Examples: "audio->text" (Whisper), "text+audio->text" (multimodal)

    Args:
        models: list of model dicts from fetch_all_models_raw()
        input_modality: "audio", "text", "image", etc.

    Returns:
        Sorted list of matching model ID strings.
    """
    result = []
    for m in models:
        arch = m.get("architecture", {})
        modality_str = arch.get("modality", "")
        if "->" in modality_str:
            input_part = modality_str.split("->", 1)[0]
        else:
            input_part = modality_str
        if input_modality in input_part:
            result.append(m["id"])
    return sorted(result)


def fetch_filtered_models(cls, output_modality, fallback_models, log_prefix):
    """
    Shared model-list fetcher for all modality-specific nodes.

    Fetches GET /api/v1/models?output_modalities=<output_modality> with a
    per-class 1-hour cache.  Each node passes its own class so the cache is
    stored in the node's class attributes (cls.models_cache / cls.last_fetch_time)
    rather than in a global dict — this keeps each node's list independent.

    cls must expose:
        cls.models_cache     — None or list[str]
        cls.last_fetch_time  — float (Unix timestamp, default 0)
        cls.cache_duration   — int seconds (default 3600)
        cls._fallback_models — list[str] used when the API call fails and
                               there is no existing cache

    Args:
        cls:              The calling node class (passed as the first arg of
                          a classmethod).
        output_modality:  Value for the ?output_modalities= query parameter
                          (e.g. "speech", "transcription", "image", "rerank").
        fallback_models:  list[str] of hardcoded model IDs used on error.
        log_prefix:       String prepended to log messages, e.g. "[SpeechNode]".

    Returns:
        list[str] of model IDs, sorted alphabetically.
    """
    current_time = time.time()
    if cls.models_cache is None or (current_time - cls.last_fetch_time > cls.cache_duration):
        try:
            response = requests.get(
                f"{BASE_URL}/models",
                params={"output_modalities": output_modality},
                timeout=DEFAULT_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            models = sorted(m["id"] for m in data if m.get("id"))
            cls.models_cache = models if models else fallback_models[:]
            cls.last_fetch_time = current_time
        except Exception as e:
            print(f"{log_prefix} Error fetching {output_modality} models: {e}")
            if cls.models_cache is None:
                cls.models_cache = fallback_models[:]
    return cls.models_cache


def get_model_details(model_id, api_key=None, timeout=None):
    """
    Fetches details for a specific model from the /models endpoint.

    Returns a dict with model info or empty dict on failure.
    Includes: id, name, supported_voices, architecture, etc.
    """
    try:
        validated_timeout = validate_request_timeout(
            timeout if timeout is not None else DEFAULT_REQUEST_TIMEOUT
        )
        headers = {}
        if api_key:
            headers = build_standard_headers(api_key)

        # Fetch speech models to get supported_voices (faster than fetching all models)
        response = requests.get(
            f"{BASE_URL}/models",
            params={"output_modalities": "speech"},
            timeout=validated_timeout,
            headers=headers,
        )
        response.raise_for_status()
        models = response.json().get("data", [])

        print(f"[openrouter_shared] Fetched {len(models)} speech models")

        for m in models:
            if m.get("id") == model_id:
                supported_voices = m.get("supported_voices")
                print(f"[openrouter_shared] Found model {model_id}")
                print(f"[openrouter_shared]   supported_voices: {supported_voices}")
                return m

        print(f"[openrouter_shared] Model {model_id} not found in speech models list")
        return {}
    except Exception as e:
        print(f"[openrouter_shared] Error fetching model details for {model_id}: {e}")
        import traceback
        print(f"[openrouter_shared] Traceback: {traceback.format_exc()}")
        return {}


def get_model_supported_voices(model_id, api_key=None, timeout=None):
    """
    Fetches the list of supported voices for a TTS model.

    Returns: list[str] of voice IDs, or empty list if none or model not found.
    """
    print(f"[openrouter_shared] get_model_supported_voices called for {model_id}")
    model_details = get_model_details(model_id, api_key, timeout)

    if not model_details:
        print(f"[openrouter_shared] No model details found")
        return []

    voices = model_details.get("supported_voices")
    print(f"[openrouter_shared] Raw supported_voices value: {voices} (type: {type(voices)})")

    if voices is None:
        print(f"[openrouter_shared] supported_voices is None")
        return []

    if isinstance(voices, list):
        print(f"[openrouter_shared] Returning {len(voices)} voices")
        return voices

    print(f"[openrouter_shared] supported_voices is not a list: {type(voices)}")
    return []


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


# ── Audio encoding ────────────────────────────────────────────────────────────

def encode_audio_to_base64(audio_dict, fmt="wav"):
    """
    Encodes a ComfyUI AUDIO dict to a base64 string for the STT API.

    audio_dict: {"waveform": Tensor[1, C, N], "sample_rate": int}
    fmt: target container format.
         "wav" always works (stdlib wave).
         All other formats require torchaudio (torchcodec).

    Returns: raw base64 string (NOT a data URI — OpenRouter wants plain base64).
    Raises: RuntimeError if fmt != "wav" and torchaudio is unavailable.
    """
    waveform = audio_dict["waveform"]   # [1, C, N]
    sample_rate = audio_dict["sample_rate"]

    if waveform.ndim == 3:
        waveform = waveform.squeeze(0)  # → [C, N]

    buf = io.BytesIO()

    if fmt == "wav":
        import wave as _wave
        wf_np = waveform.cpu().numpy()          # [C, N]
        n_channels = wf_np.shape[0]
        # [C, N] → [N, C] interleaved int16
        pcm = np.clip(wf_np.T * 32767, -32768, 32767).astype(np.int16)
        with _wave.open(buf, "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(2)          # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
    elif _HAS_TORCHAUDIO:
        _torchaudio.save(buf, waveform, sample_rate, format=fmt)
    else:
        raise RuntimeError(
            f"torchaudio is required to encode audio as '{fmt}'. "
            "Install torchcodec or select 'wav' as the audio_format."
        )

    return base64.b64encode(buf.getvalue()).decode("utf-8")


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


# ── Video handling ───────────────────────────────────────────────────────────

def download_file(url, timeout=None, headers=None):
    """
    Downloads a file from a URL and returns the raw bytes.

    Args:
        url: The URL to download from.
        timeout: Optional request timeout in seconds.
        headers: Optional dict of HTTP headers (e.g. for Authorization).

    Returns: bytes or None on failure.
    """
    try:
        validated_timeout = validate_request_timeout(
            timeout if timeout is not None else DEFAULT_REQUEST_TIMEOUT
        )
        response = requests.get(
            url,
            timeout=validated_timeout,
            stream=True,
            headers=headers or {},
        )
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"[openrouter_shared] Error downloading file: {e}")
        return None


def video_bytes_to_frames(video_bytes, extract_fps=True):
    """
    Converts raw video bytes to a frame tensor and metadata.

    Returns: tuple(
        frames_tensor: (num_frames, height, width, 3) in [0, 1] float,
        metadata: dict with fps, frame_count, duration, width, height
    )

    On failure, returns (placeholder_tensor, empty_metadata).
    """
    try:
        import cv2
        import tempfile

        if not video_bytes or len(video_bytes) == 0:
            raise ValueError("Empty video bytes")

        print(f"[openrouter_shared] Processing {len(video_bytes)} bytes of video data")

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(video_bytes)
            tmp_path = tmp_file.name

        print(f"[openrouter_shared] Temp video written to: {tmp_path}")

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video file with OpenCV - check codec support")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"[openrouter_shared] Video properties: {frame_count} frames, {width}x{height}, {fps:.2f} fps")

            if frame_count <= 0:
                raise ValueError(f"Invalid frame count: {frame_count}")

            frames = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB and normalize to [0, 1]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_float = frame_rgb.astype(np.float32) / 255.0
                frames.append(frame_float)
                frame_idx += 1

            cap.release()

            if not frames:
                raise ValueError("No frames extracted from video")

            # Stack frames into tensor: (num_frames, height, width, 3)
            frames_array = np.stack(frames, axis=0)
            frames_tensor = torch.from_numpy(frames_array).to(dtype=torch.float32)

            duration = frame_count / fps if fps > 0 else 0
            metadata = {
                "fps": float(fps),
                "frame_count": frame_count,
                "duration": float(duration),
                "width": width,
                "height": height,
            }

            print(f"[openrouter_shared] Extracted {frame_count} frames at {fps:.2f} fps, {width}x{height}")
            return frames_tensor, metadata

        finally:
            os.unlink(tmp_path)

    except ImportError as e:
        print(f"[openrouter_shared] OpenCV (cv2) is required for video processing. Install it with: pip install opencv-python. Error: {e}")
        return _placeholder_video_tensor(), {}
    except Exception as e:
        import traceback
        print(f"[openrouter_shared] Error converting video bytes to frames: {e}")
        print(f"[openrouter_shared] Traceback: {traceback.format_exc()}")
        return _placeholder_video_tensor(), {}


def _placeholder_video_tensor():
    """Returns a single black frame as placeholder."""
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def _import_comfy_video_classes():
    """
    Attempts to import ComfyUI's VideoFromComponents and VideoComponents classes
    from various known locations. Returns (VideoFromComponents, VideoComponents)
    or (None, None) if unavailable.
    """
    import_paths = [
        ("comfy_api.latest._input_impl.video_types",
         "comfy_api.latest._util.video_types"),
        ("comfy_api.input_impl.video_types",
         "comfy_api.util.video_types"),
        ("comfy_api.input_impl",
         "comfy_api.util"),
    ]
    for impl_path, util_path in import_paths:
        try:
            impl_mod = __import__(impl_path, fromlist=["VideoFromComponents"])
            util_mod = __import__(util_path, fromlist=["VideoComponents"])
            return (
                getattr(impl_mod, "VideoFromComponents"),
                getattr(util_mod, "VideoComponents"),
            )
        except (ImportError, AttributeError):
            continue
    return None, None


def build_video_object(frames_tensor, fps, audio=None, metadata=None):
    """
    Constructs a ComfyUI VIDEO object from frame tensor and metadata.

    Uses VideoFromComponents/VideoComponents when available (ComfyUI core),
    otherwise returns a minimal compatible wrapper that implements the
    VideoInput interface (get_components, get_dimensions, save_to).

    Args:
        frames_tensor: (N, H, W, 3) tensor with values in [0, 1].
        fps: Frame rate as float or Fraction.
        audio: Optional AUDIO dict {"waveform": tensor, "sample_rate": int}.
        metadata: Optional metadata dict.

    Returns: A VIDEO object compatible with ComfyUI's SaveVideo node.
    """
    from fractions import Fraction

    if isinstance(fps, Fraction):
        frame_rate = fps
    else:
        try:
            fps_float = float(fps) if fps else 24.0
            if fps_float <= 0:
                fps_float = 24.0
            # Use limit_denominator for fractional fps like 29.97
            frame_rate = Fraction(fps_float).limit_denominator(1000)
        except (ValueError, TypeError):
            frame_rate = Fraction(24, 1)

    VideoFromComponents, VideoComponents = _import_comfy_video_classes()

    if VideoFromComponents is not None and VideoComponents is not None:
        try:
            components = VideoComponents(
                images=frames_tensor,
                frame_rate=frame_rate,
                audio=audio,
                metadata=metadata or {},
            )
            return VideoFromComponents(components)
        except Exception as e:
            print(f"[openrouter_shared] Failed to build ComfyUI VideoFromComponents: {e}, falling back to wrapper")

    return _FallbackVideoObject(frames_tensor, frame_rate, audio, metadata or {})


class _FallbackVideoObject:
    """
    Minimal VideoInput-compatible wrapper used when ComfyUI's video classes
    cannot be imported. Implements get_components, get_dimensions, and save_to.
    """

    def __init__(self, images, frame_rate, audio=None, metadata=None):
        self._images = images
        self._frame_rate = frame_rate
        self._audio = audio
        self._metadata = metadata or {}

    def get_components(self):
        """Returns a namespace-like object with images, frame_rate, audio, metadata."""
        from types import SimpleNamespace
        return SimpleNamespace(
            images=self._images,
            frame_rate=self._frame_rate,
            audio=self._audio,
            metadata=self._metadata,
            alpha=None,
        )

    def get_dimensions(self):
        """Returns (width, height) from the frame tensor shape."""
        if self._images is None or self._images.ndim < 3:
            return (0, 0)
        # tensor shape: (N, H, W, 3)
        height = int(self._images.shape[1])
        width = int(self._images.shape[2])
        return (width, height)

    def get_duration(self):
        if self._images is None or self._frame_rate == 0:
            return 0.0
        return float(self._images.shape[0]) / float(self._frame_rate)

    def get_frame_rate(self):
        return self._frame_rate

    def get_frame_count(self):
        if self._images is None:
            return 0
        return int(self._images.shape[0])

    def save_to(self, path, format=None, codec=None, metadata=None):
        """
        Saves the video frames to disk using OpenCV (mp4 + mp4v codec).
        """
        try:
            import cv2
        except ImportError:
            raise RuntimeError("OpenCV is required for saving video. Install with: pip install opencv-python")

        if self._images is None or self._images.shape[0] == 0:
            raise ValueError("No frames to save")

        width, height = self.get_dimensions()
        fps = float(self._frame_rate) if float(self._frame_rate) > 0 else 24.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        try:
            frames_np = (self._images.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            for frame in frames_np:
                # Convert RGB -> BGR for OpenCV
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        finally:
            writer.release()
