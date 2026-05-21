"""
node_speech.py — OpenRouter Text-to-Speech Node

Uses the dedicated POST /v1/audio/speech endpoint (OpenAI-compatible TTS API).
This is distinct from audio-output chat-completion models — use this node for
pure TTS/speech synthesis tasks.

Request fields:
  input           — text to synthesise
  model           — TTS model ID (e.g. openai/tts-1, elevenlabs/…)
  voice           — speaker preset
  response_format — audio container / codec
  speed           — playback rate (0.25–4.0)

Response: application/octet-stream — raw audio bytes decoded into a
          ComfyUI AUDIO dict {"waveform": Tensor[1,C,N], "sample_rate": int}.
"""

import requests
import json
import time
import torch
from . import openrouter_shared as shared


class OpenRouterSpeechNode:
    """
    ComfyUI node for text-to-speech via OpenRouter's /v1/audio/speech endpoint.

    Filters the model list to audio-output models only.
    Returns three outputs:
      1) "audio"  : ComfyUI AUDIO dict {"waveform": Tensor[1,C,N], "sample_rate": int}
      2) "Stats"  : timing, model, format, voice
      3) "Credits": remaining OpenRouter account balance
    """

    models_cache = None
    last_fetch_time = 0
    cache_duration = 3600
    default_request_timeout = shared.DEFAULT_REQUEST_TIMEOUT
    min_request_timeout = shared.MIN_REQUEST_TIMEOUT
    max_request_timeout = shared.MAX_REQUEST_TIMEOUT

    _fallback_models = [
        "openai/tts-1",
        "openai/tts-1-hd",
        "elevenlabs/eleven-turbo-v2",
        "elevenlabs/eleven-multilingual-v2",
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
                    "default": "Hello, how can I help you today?"
                }),
                "model": (cls.fetch_openrouter_models(),),
                # Free-text so any provider's voice name works.
                # OpenAI voices: alloy echo fable onyx nova shimmer
                # Google Gemini voices: Aoede Charon Fenrir Kore Puck (and others)
                # ElevenLabs: use the voice ID or name from your ElevenLabs account
                "voice": ("STRING", {
                    "multiline": False,
                    "default": "alloy",
                }),
                "output_format": (list(shared.AUDIO_FORMAT_OPTIONS), {"default": "mp3"}),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.25,
                    "max": 4.0,
                    "step": 0.05,
                    "display": "number",
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
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "Stats", "Credits")

    FUNCTION = "generate_speech"
    CATEGORY = "LLM"

    @classmethod
    def fetch_openrouter_models(cls):
        """
        Fetches TTS-capable model IDs via GET /api/v1/models?output_modalities=speech.

        This dedicated query parameter returns only speech-synthesis models,
        avoiding the general audio-output filter which also matches chat-completion
        models (e.g. gpt-4o-audio-preview) that are incompatible with /audio/speech.

        Falls back to the hardcoded list if the fetch fails or returns nothing.
        """
        current_time = time.time()
        if cls.models_cache is None or (current_time - cls.last_fetch_time > cls.cache_duration):
            try:
                response = requests.get(
                    f"{shared.BASE_URL}/models",
                    params={"output_modalities": "speech"},
                    timeout=shared.DEFAULT_REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json().get("data", [])
                models = sorted(m["id"] for m in data if m.get("id"))
                cls.models_cache = models if models else cls._fallback_models[:]
                cls.last_fetch_time = current_time
            except Exception as e:
                print(f"[SpeechNode] Error fetching TTS models: {e}")
                if cls.models_cache is None:
                    cls.models_cache = cls._fallback_models[:]
        return cls.models_cache

    def _silent_audio(self):
        """Returns a silent 1-second mono audio placeholder."""
        return {
            "waveform": torch.zeros((1, 1, 44100), dtype=torch.float32),
            "sample_rate": 44100
        }

    def generate_speech(self, api_key, prompt, model,
                        voice="alloy", output_format="mp3", speed=1.0,
                        request_timeout=120, prompt_input=None):
        """
        Calls POST /v1/audio/speech on OpenRouter.

        The TTS endpoint accepts:
          input           — text to synthesise
          model           — TTS model slug
          voice           — speaker preset
          response_format — desired audio format
          speed           — playback rate (0.25–4.0)

        The response body is raw audio bytes (application/octet-stream).

        Returns (audio_dict, stats_str, credits_str).
        """
        silent = self._silent_audio()

        # Resolve API key
        api_key = shared.get_api_key(api_key)
        if not api_key:
            return (
                silent,
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

        # Clamp speed to documented range
        try:
            speed_f = float(speed)
            speed_f = max(0.25, min(4.0, speed_f))
        except (ValueError, TypeError):
            speed_f = 1.0

        headers = shared.build_standard_headers(api_key)
        url = f"{shared.BASE_URL}/audio/speech"

        data = {
            "model": model,
            "input": effective_prompt,
            "voice": voice,
            "response_format": output_format,
            "speed": speed_f,
        }

        try:
            start_time = time.time()
            response = requests.post(
                url, headers=headers, json=data, timeout=validated_timeout
            )
            response.raise_for_status()
            elapsed = time.time() - start_time

            audio_bytes = response.content
            if not audio_bytes:
                return (silent, "Stats N/A", "Empty audio response body.")

            print(
                f"[SpeechNode] Received {len(audio_bytes)} bytes "
                f"in {elapsed:.2f}s (format={output_format})"
            )

            audio_dict = shared.decode_audio_bytes(audio_bytes, output_format)

            stats = (
                f"TPS: N/A, Prompt Tokens: N/A, Completion Tokens: N/A, "
                f"Temp: N/A, Model: {model}, "
                f"Format: {output_format}, Voice: {voice}, Speed: {speed_f:.2f}, "
                f"Elapsed: {elapsed:.2f}s"
            )

            credits = shared.fetch_credits(api_key, validated_timeout)
            return (audio_dict, stats, credits)

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

            # Surface actionable guidance for the most common failures.
            # Not all models listed by ?output_modalities=speech are compatible
            # with the /audio/speech endpoint — some (e.g. Gemini TTS, Voxtral)
            # use a different underlying API and will fail here.
            if status == 404:
                error_msg += (
                    f"\n\nModel '{model}' is not available via /audio/speech. "
                    "It may use a different API format (e.g. streaming chat completions). "
                    "Known-working alternatives: openai/tts-1, openai/tts-1-hd, "
                    "elevenlabs/eleven-turbo-v2"
                )
            elif status == 500:
                error_msg += (
                    f"\n\nModel '{model}' returned a server error from /audio/speech. "
                    "It may not implement the OpenAI-compatible TTS format. "
                    "Known-working alternatives: openai/tts-1, openai/tts-1-hd, "
                    "elevenlabs/eleven-turbo-v2"
                )

            print(f"[SpeechNode] ERROR: {error_msg}")
            return (silent, "Stats N/A due to error", error_msg)
        except Exception as e:
            print(f"[SpeechNode] ERROR: Node Error: {str(e)}")
            return (silent, "Stats N/A due to error", f"Node Error: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, api_key, prompt, model,
                   voice="alloy", output_format="mp3", speed=1.0,
                   request_timeout=120, prompt_input=None):
        """Check if any input that affects the output has changed."""
        try:
            timeout_int = int(request_timeout)
            timeout_int = max(cls.min_request_timeout, min(cls.max_request_timeout, timeout_int))
        except (ValueError, TypeError):
            timeout_int = cls.default_request_timeout

        try:
            speed_f = round(max(0.25, min(4.0, float(speed))), 4)
        except (ValueError, TypeError):
            speed_f = 1.0

        return (api_key, prompt, model, voice, output_format, speed_f, timeout_int, prompt_input)


NODE_CLASS_MAPPINGS = {
    "OpenRouterSpeechNode": OpenRouterSpeechNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterSpeechNode": "OpenRouter Speech Node"
}
