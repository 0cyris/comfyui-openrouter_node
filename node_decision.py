"""
node_decision.py — OpenRouter Decision Node

Asks one noul, choice or score question about text and context images through
POST /api/alpha/decisions and returns the result as a 0-1 value.
"""

import asyncio
import base64
import io
import json
import math
import time
from urllib.parse import urlsplit

import requests
import torch
from PIL import Image

from . import openrouter_shared as shared

QUESTION_NAME = "decision"

# Accepted image range of every decision model (Jev's limits, the strictest listed)
MIN_SIDE, MAX_SIDE = 256, 1568
MIN_PIXELS, MAX_PIXELS = 78400, 2458624


def fit_size(width, height):
    """Scales into the accepted range keeping aspect ratio; sizes already inside it are unchanged."""
    scale = max(1, MIN_SIDE / min(width, height), math.sqrt(MIN_PIXELS / (width * height)))
    scale = min(scale, MAX_SIDE / max(width, height), math.sqrt(MAX_PIXELS / (width * height)))
    if scale == 1:
        return width, height
    round_size = math.ceil if scale > 1 else round
    return round_size(width * scale), round_size(height * scale)


def image_batch_to_jpeg_base64_list(images):
    """Converts every frame of a ComfyUI IMAGE batch (BHWC) to a base64 JPEG fitted to the accepted size range."""
    encoded = []
    for frame in images:
        pil_image = Image.fromarray((frame.clamp(0, 1) * 255).round().byte().cpu().numpy(), "RGB")
        size = fit_size(*pil_image.size)
        if size != pil_image.size:
            pil_image = pil_image.resize(size, Image.LANCZOS)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=95)
        encoded.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    return encoded


class OpenRouterDecisionNode:
    """
    ComfyUI node for the OpenRouter Decisions API.

    Each frame of the optional "candidates" batch is judged in its own concurrent request, after the text and
    context images. Without candidates there is one request.

    Returns seven outputs; the first three describe the candidate with the highest value (the lowest index on a tie):
      1) "answer"    : "true"/"false" by threshold (noul), the chosen option (choice) or the nearest level's legend text (score)
      2) "value"     : probability of true (noul), confidence (choice) or score / (levels - 1) (score)
      3) "passed"    : value >= threshold
      4) "details"   : the answer object as JSON, or a JSON array of every candidate's answer in order
      5) "cost"      : usage.cost summed over all requests (0.0 if not reported)
      6) "best_index": index of that candidate (0 without candidates)
      7) "best_image": that candidate's original frame (a black 64x64 placeholder without candidates)
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
                "question_type": (["noul", "choice", "score"], {"tooltip": "noul: yes/no probability. choice: pick one of the options in criteria. score: position on the ordered levels in criteria."}),
                "instructions": ("STRING", {"multiline": True, "default": "", "placeholder": "The question, e.g. Which direction does the character face?", "tooltip": "The question to ask about the state."}),
                "criteria": ("STRING", {"multiline": True, "default": "", "placeholder": "choice: one option per line as name: guidance\nscore: one level per line, lowest first", "tooltip": "choice: one option per line as 'name: guidance' (guidance is optional). score: one level per line, lowest first."}),
                "true_guidance": ("STRING", {"multiline": True, "default": "", "placeholder": "When the answer is true, e.g. faces viewer-left", "tooltip": "noul only: what makes the answer true."}),
                "false_guidance": ("STRING", {"multiline": True, "default": "", "placeholder": "When the answer is false, e.g. faces viewer-right or the camera", "tooltip": "noul only: what makes the answer false."}),
                "state_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Context in words, e.g. a profile view of the character", "tooltip": "Text context sent before the images."}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "value >= threshold sets passed, and the noul answer."}),
                "request_timeout": ("INT", {
                    "default": cls.default_request_timeout,
                    "min": cls.min_request_timeout,
                    "max": cls.max_request_timeout,
                    "step": 1,
                    "display": "number",
                }),
                "retries": ("INT", {"default": 1, "min": 0, "max": 5}),
            },
            "optional": {
                "candidates": ("IMAGE", {"tooltip": "Each frame is judged in its own request; the best one is returned as best_index and best_image."}),
            },
            # image_1..image_N are added dynamically by openrouter_dynamic_inputs.js
        }

    RETURN_TYPES = ("STRING", "FLOAT", "BOOLEAN", "STRING", "FLOAT", "INT", "IMAGE")
    RETURN_NAMES = ("answer", "value", "passed", "details", "cost", "best_index", "best_image")

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
    def build_payload(model, question_type, instructions, criteria, true_guidance, false_guidance, state_text, images):
        lines = [line.strip() for line in criteria.splitlines() if line.strip()]
        if question_type == "noul":
            criteria = {"true": true_guidance, "false": false_guidance}
        elif question_type == "choice":
            criteria = {}
            for line in lines:
                name, colon, guidance = line.partition(":")
                name = name.strip()
                if name in criteria:
                    raise RuntimeError(f"Duplicate option '{name}' in criteria: each choice option needs a unique name")
                criteria[name] = guidance.strip() if colon else None
            if not criteria:
                raise RuntimeError("question_type is 'choice' but criteria is empty: write one option per line as 'name: guidance', or set question_type to 'noul' for a yes/no question")
        else:
            criteria = lines
            if not criteria:
                raise RuntimeError("question_type is 'score' but criteria is empty: write one level per line, lowest first, or set question_type to 'noul' for a yes/no question")
        return {
            "model": model,
            "questions": {QUESTION_NAME: {"type": question_type, "instructions": instructions, "criteria": criteria}},
            "state": ([state_text] if state_text else []) + [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in images
            ],
        }

    @staticmethod
    def parse_answer(result, payload, threshold):
        question = payload["questions"][QUESTION_NAME]
        answer = result["answers"][QUESTION_NAME]
        if question["type"] == "noul":
            value = float(answer["noul"])
            text = "true" if value >= threshold else "false"
        elif question["type"] == "choice":
            text = answer["choice"]
            value = float(answer["confidence"] if "confidence" in answer else answer["probabilities"][text])
        else:
            top = len(question["criteria"]) - 1
            value = float(answer["score"]) / top if top else 0.0
            text = question["criteria"][min(max(int(answer["score"] + 0.5), 0), top)]
        cost = (result.get("usage") or {}).get("cost") or 0.0
        return (text, value, value >= threshold, answer, cost)

    async def decide(self, api_key, model, question_type, instructions, criteria, true_guidance, false_guidance,
                     state_text, threshold, request_timeout, retries, candidates=None, **kwargs):
        api_key = shared.get_api_key(api_key)
        if not api_key:
            raise RuntimeError("API key not provided: fill in api_key, set the LLM_KEY env var, or use openrouter_api_key.json")
        timeout = shared.validate_request_timeout(
            request_timeout, self.min_request_timeout, self.max_request_timeout, self.default_request_timeout)
        connected = [kwargs[k] for k in sorted((k for k in kwargs if k.startswith("image_")), key=lambda x: int(x.split("_")[1]))
                     if kwargs[k] is not None]
        if (connected or candidates is not None) and model in self.model_input_modalities and "image" not in self.model_input_modalities[model]:
            raise RuntimeError(f"{model} does not accept images, but image_N or candidates is connected: pick a model that accepts images or disconnect them")
        context = []
        for batch in connected:
            context.extend(await asyncio.to_thread(image_batch_to_jpeg_base64_list, batch))
        frames = [[]] if candidates is None else [[b64] for b64 in await asyncio.to_thread(image_batch_to_jpeg_base64_list, candidates)]
        base = urlsplit(shared.BASE_URL)
        url = f"{base.scheme}://{base.netloc}/api/alpha/decisions"
        payloads = [self.build_payload(model, question_type, instructions, criteria, true_guidance, false_guidance, state_text, context + frame)
                    for frame in frames]
        results = await asyncio.gather(*(shared.post_with_retries(url, api_key, payload, timeout, retries, "Decision") for payload in payloads))
        parsed = [self.parse_answer(result, payload, threshold) for result, payload in zip(results, payloads)]
        best = max(range(len(parsed)), key=lambda i: parsed[i][1])
        text, value, passed, _, _ = parsed[best]
        if candidates is None:
            details, best_image = parsed[0][3], torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        else:
            details, best_image = [p[3] for p in parsed], candidates[best:best + 1].clone()
        return (text, value, passed, json.dumps(details), sum(p[4] for p in parsed), best, best_image)


NODE_CLASS_MAPPINGS = {
    "OpenRouterDecisionNode": OpenRouterDecisionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterDecisionNode": "OpenRouter Decision Node"
}
