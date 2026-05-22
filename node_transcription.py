"""
node_transcription.py — OpenRouter Speech-to-Text Node

POST /api/v1/audio/transcriptions
Content-Type: application/json

Request body:
  model        — STT model identifier (e.g. openai/whisper-large-v3)
  input_audio  — {data: <base64>, format: "wav"|"mp3"|...}
  language     — optional ISO-639-1 code; omit for auto-detection
  temperature  — optional sampling temperature

Response:
  {text, usage: {seconds, input_tokens, output_tokens, cost}}
"""

import requests
import json
import time
import hashlib
import torch
from . import openrouter_shared as shared


class OpenRouterTranscriptionNode:
    """
    ComfyUI node for speech-to-text via OpenRouter's /v1/audio/transcriptions endpoint.

    Accepts a ComfyUI AUDIO input; returns the transcribed text plus stats.
    Model list is filtered to models whose input modality includes audio.

    Returns three outputs:
      1) "transcription": the recognised text string
      2) "Stats"        : timing, model, audio duration, token usage
      3) "Credits"      : remaining OpenRouter account balance
    """

    models_cache = None
    last_fetch_time = 0
    cache_duration = 3600
    default_request_timeout = shared.DEFAULT_REQUEST_TIMEOUT
    min_request_timeout = shared.MIN_REQUEST_TIMEOUT
    max_request_timeout = shared.MAX_REQUEST_TIMEOUT

    _fallback_models = [
        "openai/whisper-large-v3",
        "openai/whisper-large-v3-turbo",
        "openai/whisper-1",
    ]

    # Formats we can encode the ComfyUI waveform to before sending.
    # "wav" always works (stdlib); others require torchaudio/torchcodec.
    _send_formats = ["wav", "mp3", "flac", "ogg", "webm", "m4a", "aac"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "audio": ("AUDIO",),
                "model": (cls.fetch_openrouter_models(),),
                # ISO-639-1 language hint; leave blank for auto-detection
                "language": ("STRING", {
                    "multiline": False,
                    "default": "",
                }),
                # Format used when encoding the waveform for the API.
                # wav works without extra libraries; others need torchcodec.
                "audio_format": (cls._send_formats, {"default": "wav"}),
                "request_timeout": ("INT", {
                    "default": cls.default_request_timeout,
                    "min": cls.min_request_timeout,
                    "max": cls.max_request_timeout,
                    "step": 1,
                    "display": "number",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("transcription", "Stats", "Credits")

    FUNCTION = "transcribe"
    CATEGORY = "LLM"

    @classmethod
    def fetch_openrouter_models(cls):
        """
        Fetches STT model IDs via GET /api/v1/models?output_modalities=transcription.

        By analogy with the speech node (?output_modalities=speech), this query
        parameter returns only transcription models, avoiding chat-completion
        models that accept audio input (e.g. gpt-4o-audio-preview) which are
        incompatible with the /audio/transcriptions endpoint.

        Falls back to the hardcoded Whisper list if the fetch fails or returns nothing.
        """
        current_time = time.time()
        if cls.models_cache is None or (current_time - cls.last_fetch_time > cls.cache_duration):
            try:
                response = requests.get(
                    f"{shared.BASE_URL}/models",
                    params={"output_modalities": "transcription"},
                    timeout=shared.DEFAULT_REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json().get("data", [])
                models = sorted(m["id"] for m in data if m.get("id"))
                cls.models_cache = models if models else cls._fallback_models[:]
                cls.last_fetch_time = current_time
            except Exception as e:
                print(f"[TranscriptionNode] Error fetching STT models: {e}")
                if cls.models_cache is None:
                    cls.models_cache = cls._fallback_models[:]
        return cls.models_cache

    def transcribe(self, api_key, audio, model,
                   language="", audio_format="wav",
                   request_timeout=120):
        """
        Encodes the ComfyUI AUDIO waveform to base64 and calls
        POST /v1/audio/transcriptions.

        Returns (transcription_text, stats_str, credits_str).
        """
        api_key = shared.get_api_key(api_key)
        if not api_key:
            return (
                "",
                "Stats N/A",
                "Error: API Key not provided. Set LLM_KEY env var or use openrouter_api_key.json",
            )

        validated_timeout = shared.validate_request_timeout(
            request_timeout, self.min_request_timeout,
            self.max_request_timeout, self.default_request_timeout
        )

        # Encode waveform → base64
        try:
            audio_b64 = shared.encode_audio_to_base64(audio, audio_format)
        except Exception as e:
            return ("", "Stats N/A", f"Error encoding audio: {e}")

        headers = shared.build_standard_headers(api_key)
        url = f"{shared.BASE_URL}/audio/transcriptions"

        data = {
            "model": model,
            "input_audio": {
                "data": audio_b64,
                "format": audio_format,
            },
        }
        if language and language.strip():
            data["language"] = language.strip()

        try:
            start_time = time.time()
            response = requests.post(
                url, headers=headers, json=data, timeout=validated_timeout
            )
            response.raise_for_status()
            elapsed = time.time() - start_time

            result = response.json()
            transcription = result.get("text", "")

            usage = result.get("usage") or {}
            audio_seconds = usage.get("seconds", 0.0)
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cost = usage.get("cost", 0.0)

            stats = (
                f"Model: {model}, "
                f"Audio: {audio_seconds:.1f}s, "
                f"Input tokens: {input_tokens}, "
                f"Output tokens: {output_tokens}, "
                f"Cost: ${cost:.4f}, "
                f"Elapsed: {elapsed:.2f}s"
            )

            credits = shared.fetch_credits(api_key, validated_timeout)
            return (transcription, stats, credits)

        except requests.exceptions.RequestException as e:
            status = None
            detail = ""
            if hasattr(e, "response") and e.response is not None:
                status = e.response.status_code
                try:
                    detail = f" | Details: {e.response.json()}"
                except (json.JSONDecodeError, Exception):
                    detail = f" | HTTP {status}"
            else:
                detail = " (Network or connection issue)"
            error_msg = f"API Request Error: {str(e)}{detail}"
            print(f"[TranscriptionNode] ERROR: {error_msg}")
            return ("", "Stats N/A due to error", error_msg)
        except Exception as e:
            print(f"[TranscriptionNode] ERROR: {str(e)}")
            return ("", "Stats N/A due to error", f"Node Error: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, api_key, audio, model,
                   language="", audio_format="wav",
                   request_timeout=120):
        """Hash the waveform so re-runs only happen when audio actually changes."""
        try:
            waveform = audio["waveform"]
            if isinstance(waveform, torch.Tensor):
                audio_hash = hashlib.sha256(
                    waveform.cpu().numpy().tobytes()
                ).hexdigest()
            else:
                audio_hash = str(waveform)
        except Exception:
            audio_hash = "hash_error"

        try:
            timeout_int = int(request_timeout)
            timeout_int = max(cls.min_request_timeout, min(cls.max_request_timeout, timeout_int))
        except (ValueError, TypeError):
            timeout_int = cls.default_request_timeout

        return (api_key, model, language, audio_format, timeout_int, audio_hash)


NODE_CLASS_MAPPINGS = {
    "OpenRouterTranscriptionNode": OpenRouterTranscriptionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterTranscriptionNode": "OpenRouter Transcription Node"
}
