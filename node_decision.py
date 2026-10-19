"""
node_decision.py — OpenRouter Decision Node

Asks one yes/no (noul) question about text through POST /api/alpha/decisions
and returns the probability of true.
"""

import json
import time
from urllib.parse import urlsplit

import requests

from . import openrouter_shared as shared

QUESTION_NAME = "decision"


class OpenRouterDecisionNode:
    """
    ComfyUI node for the OpenRouter Decisions API.

    Returns five outputs:
      1) "answer" : "true" or "false" by threshold
      2) "value"  : probability of true
      3) "passed" : value >= threshold
      4) "details": the answer object as JSON
      5) "cost"   : usage.cost as a float (0.0 if not reported)
    Errors raise RuntimeError so ComfyUI shows them and caches nothing.
    """

    models_cache = None
    model_input_modalities = {}
    last_fetch_time = 0
    cache_duration = shared.CACHE_DURATION
    default_request_timeout = shared.DEFAULT_REQUEST_TIMEOUT
    min_request_timeout = shared.MIN_REQUEST_TIMEOUT
    max_request_timeout = shared.MAX_REQUEST_TIMEOUT

    _fallback_models = [
        "perplexity/pplx-decider-v1.2",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "model": (cls.fetch_openrouter_models(),),
                "instructions": ("STRING", {"multiline": True, "default": ""}),
                "true_guidance": ("STRING", {"multiline": True, "default": ""}),
                "false_guidance": ("STRING", {"multiline": True, "default": ""}),
                "state_text": ("STRING", {"multiline": True, "default": ""}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "request_timeout": ("INT", {
                    "default": cls.default_request_timeout,
                    "min": cls.min_request_timeout,
                    "max": cls.max_request_timeout,
                    "step": 1,
                    "display": "number",
                }),
                "retries": ("INT", {"default": 1, "min": 0, "max": 5}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "BOOLEAN", "STRING", "FLOAT")
    RETURN_NAMES = ("answer", "value", "passed", "details", "cost")

    FUNCTION = "decide"
    CATEGORY = "LLM"

    @classmethod
    def fetch_openrouter_models(cls):
        """Fetches decision model IDs and their input modalities via GET /api/v1/models?output_modalities=decisions."""
        current_time = time.time()
        if cls.models_cache is None or (current_time - cls.last_fetch_time > cls.cache_duration):
            try:
                response = requests.get(
                    f"{shared.BASE_URL}/models",
                    params={"output_modalities": "decisions"},
                    timeout=shared.DEFAULT_REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json().get("data", [])
                cls.model_input_modalities = {m["id"]: m.get("architecture", {}).get("input_modalities", []) for m in data if m.get("id")}
                cls.models_cache = sorted(cls.model_input_modalities) or cls._fallback_models[:]
                cls.last_fetch_time = current_time
            except Exception as e:
                print(f"[DecisionNode] Error fetching decision models: {e}")
                if cls.models_cache is None:
                    cls.models_cache = cls._fallback_models[:]
        return cls.models_cache

    @staticmethod
    def build_payload(model, instructions, true_guidance, false_guidance, state_text):
        return {
            "model": model,
            "questions": {QUESTION_NAME: {
                "type": "noul",
                "instructions": instructions,
                "criteria": {"true": true_guidance, "false": false_guidance},
            }},
            "state": [state_text] if state_text else [],
        }

    @staticmethod
    def parse_answer(result, threshold):
        answer = result["answers"][QUESTION_NAME]
        value = float(answer["noul"])
        passed = value >= threshold
        cost = (result.get("usage") or {}).get("cost") or 0.0
        return ("true" if passed else "false", value, passed, json.dumps(answer), cost)

    async def decide(self, api_key, model, instructions, true_guidance, false_guidance, state_text,
                     threshold, request_timeout, retries):
        api_key = shared.get_api_key(api_key)
        if not api_key:
            raise RuntimeError("API Key not provided. Set LLM_KEY env var or use openrouter_api_key.json")
        timeout = shared.validate_request_timeout(
            request_timeout, self.min_request_timeout, self.max_request_timeout, self.default_request_timeout)
        base = urlsplit(shared.BASE_URL)
        url = f"{base.scheme}://{base.netloc}/api/alpha/decisions"
        payload = self.build_payload(model, instructions, true_guidance, false_guidance, state_text)
        result = await shared.post_with_retries(url, api_key, payload, timeout, retries, "Decision")
        return self.parse_answer(result, threshold)


NODE_CLASS_MAPPINGS = {
    "OpenRouterDecisionNode": OpenRouterDecisionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterDecisionNode": "OpenRouter Decision Node"
}
