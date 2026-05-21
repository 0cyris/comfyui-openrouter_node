"""
node_audio.py — OpenRouter Audio Generation Node

Targets audio-output models on OpenRouter (e.g. OpenAI TTS via OpenRouter).
Accepts a text prompt with voice and format selection; outputs an AUDIO tensor.
The model dropdown is filtered to models whose output modality includes "audio".
"""

import requests
import json
import time
import base64
import torch
from . import openrouter_shared as shared


class OpenRouterAudioNode:
    """
    ComfyUI node for audio generation via OpenRouter.

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
        "error_fetching_models",
        "openai/tts-1",
        "openai/tts-1-hd",
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
                "voice": (list(shared.VOICE_OPTIONS), {"default": "alloy"}),
                "output_format": (list(shared.AUDIO_FORMAT_OPTIONS), {"default": "mp3"}),
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

    FUNCTION = "generate_audio"
    CATEGORY = "LLM"

    @classmethod
    def fetch_openrouter_models(cls):
        """
        Fetches audio-output model IDs from the OpenRouter API, with caching.
        """
        raw, was_refreshed = shared.fetch_all_models_raw()
        if cls.models_cache is None or was_refreshed:
            filtered = shared.filter_models_by_output(raw, "audio")
            cls.models_cache = filtered if filtered else cls._fallback_models[:]
        return cls.models_cache

    def _silent_audio(self):
        """Returns a silent 1-second mono audio placeholder."""
        return {
            "waveform": torch.zeros((1, 1, 44100), dtype=torch.float32),
            "sample_rate": 44100
        }

    def generate_audio(self, api_key, prompt, model,
                       voice="alloy", output_format="mp3",
                       request_timeout=120, prompt_input=None):
        """
        Sends an audio generation request to OpenRouter.

        Handles two response shapes:
        - JSON envelope with choices[0].message.audio.data (base64)
        - Raw binary audio body (non-JSON Content-Type)

        Returns (audio_dict, stats_str, credits_str).
        """
        silent = self._silent_audio()

        # Resolve API key and prompt
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
        headers = shared.build_standard_headers(api_key)
        url = f"{shared.BASE_URL}/chat/completions"

        # OpenAI audio-in-chat-completions format:
        #   "modalities": ["text", "audio"]
        #   "audio": {"voice": "...", "format": "..."}
        # Note: top-level "response_format" is for JSON-mode text output and must
        # be an object ({type: ...}), not a string — don't use it for audio format.
        data = {
            "model": model,
            "modalities": ["text", "audio"],
            "audio": {
                "voice": voice,
                "format": output_format,
            },
            "messages": [{"role": "user", "content": effective_prompt}],
        }

        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=data, timeout=validated_timeout)
            response.raise_for_status()
            end_time = time.time()
            elapsed = end_time - start_time

            content_type = response.headers.get("Content-Type", "")

            if "application/json" in content_type:
                # JSON envelope — look for audio in choices[0].message.audio.data
                result = response.json()
                debug_str = json.dumps(result, default=str)
                print(f"[AudioNode] API response ({len(debug_str)} chars): {debug_str[:500]}")

                audio_b64 = None
                choices = result.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    audio_obj = msg.get("audio", {})
                    audio_b64 = audio_obj.get("data")

                if not audio_b64:
                    error_msg = "No audio data found in JSON response."
                    print(f"[AudioNode] {error_msg}")
                    return (silent, "Stats N/A", error_msg)

                try:
                    audio_bytes = base64.b64decode(audio_b64)
                except Exception as e:
                    return (silent, "Stats N/A", f"Error decoding audio base64: {e}")

                audio_dict = shared.decode_audio_bytes(audio_bytes, output_format)

                api_usage = result.get("usage", {})
                prompt_tokens = api_usage.get("prompt_tokens", 0)
                completion_tokens = api_usage.get("completion_tokens", 0)
                response_ms = result.get("response_ms")
                if response_ms and response_ms > 0:
                    tps = completion_tokens / (response_ms / 1000.0)
                elif elapsed > 0:
                    tps = completion_tokens / elapsed
                else:
                    tps = 0

                stats = shared.build_stats_text(
                    tps, prompt_tokens, completion_tokens, 0.0, model,
                    extra_parts=[f"Format: {output_format}", f"Voice: {voice}"]
                )

            else:
                # Raw binary audio response
                audio_bytes = response.content
                if not audio_bytes:
                    return (silent, "Stats N/A", "Empty audio response body.")

                audio_dict = shared.decode_audio_bytes(audio_bytes, output_format)
                stats = (
                    f"TPS: N/A, Prompt Tokens: N/A, Completion Tokens: N/A, "
                    f"Temp: 0.0, Model: {model}, "
                    f"Format: {output_format}, Voice: {voice}"
                )

            credits = shared.fetch_credits(api_key, validated_timeout)
            return (audio_dict, stats, credits)

        except requests.exceptions.RequestException as e:
            error_msg = f"API Request Error: {str(e)}"
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_msg += f" | Details: {e.response.json()}"
                except json.JSONDecodeError:
                    error_msg += f" | Status: {e.response.status_code}"
            else:
                error_msg += " (Network or connection issue)"
            print(f"[AudioNode] ERROR: {error_msg}")
            return (silent, "Stats N/A due to error", error_msg)
        except Exception as e:
            print(f"[AudioNode] ERROR: Node Error: {str(e)}")
            return (silent, "Stats N/A due to error", f"Node Error: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, api_key, prompt, model,
                   voice="alloy", output_format="mp3",
                   request_timeout=120, prompt_input=None):
        """Check if any input that affects the output has changed."""
        try:
            timeout_int = int(request_timeout)
            timeout_int = max(cls.min_request_timeout, min(cls.max_request_timeout, timeout_int))
        except (ValueError, TypeError):
            timeout_int = cls.default_request_timeout

        return (api_key, prompt, model, voice, output_format, timeout_int, prompt_input)


NODE_CLASS_MAPPINGS = {
    "OpenRouterAudioNode": OpenRouterAudioNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterAudioNode": "OpenRouter Audio Generation Node"
}
