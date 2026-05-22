"""
node_rerank.py — OpenRouter Rerank Node

POST /api/v1/rerank

Uses ComfyUI's native list mechanism:
  INPUT_IS_LIST  = True         — documents arrives as list[str]; all other
                                  scalar inputs arrive as list[T] and are
                                  unwrapped with [0].
  OUTPUT_IS_LIST = (True, True, False, False)
                                — documents and scores go out as lists for
                                  sequential downstream processing; Stats and
                                  Credits are plain strings.

Request:
  model     — rerank model ID
  query     — search query string
  documents — list of document strings
  top_n     — optional; how many results to return (0 = all)

Response results[] (sorted by relevance, highest first):
  document.text    — original document text
  relevance_score  — float
"""

import requests
import json
import time
from . import openrouter_shared as shared


class OpenRouterRerankNode:
    """
    ComfyUI node for document reranking via OpenRouter's /v1/rerank endpoint.

    Returns four outputs:
      1) "documents" (list[STRING]) : re-ranked texts, highest relevance first
      2) "scores"    (list[FLOAT])  : corresponding relevance scores
      3) "Stats"     (STRING)       : timing, model, doc counts, cost
      4) "Credits"   (STRING)       : remaining OpenRouter account balance
    """

    INPUT_IS_LIST  = True
    OUTPUT_IS_LIST = (True, True, False, False)

    models_cache = None
    last_fetch_time = 0
    cache_duration = 3600
    default_request_timeout = shared.DEFAULT_REQUEST_TIMEOUT
    min_request_timeout = shared.MIN_REQUEST_TIMEOUT
    max_request_timeout = shared.MAX_REQUEST_TIMEOUT

    _fallback_models = [
        "cohere/rerank-v3.5",
        "cohere/rerank-english-v3.0",
        "cohere/rerank-multilingual-v3.0",
        "jina-ai/jina-reranker-v2-base-multilingual",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "query": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                # forceInput: documents must be wired from an upstream list node
                "documents": ("STRING", {"forceInput": True}),
                "model": (cls.fetch_openrouter_models(),),
                # 0 = return all results; >0 = return only the top N
                "top_n": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
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
        }

    RETURN_TYPES  = ("STRING", "FLOAT",  "STRING", "STRING")
    RETURN_NAMES  = ("documents", "scores", "Stats",  "Credits")
    FUNCTION      = "rerank"
    CATEGORY      = "LLM"

    @classmethod
    def fetch_openrouter_models(cls):
        """
        Fetches rerank model IDs via GET /api/v1/models?supported_parameters=rerank.
        Falls back to the hardcoded Cohere/Jina list on any error.
        """
        current_time = time.time()
        if cls.models_cache is None or (current_time - cls.last_fetch_time > cls.cache_duration):
            try:
                response = requests.get(
                    f"{shared.BASE_URL}/models",
                    params={"supported_parameters": "rerank"},
                    timeout=shared.DEFAULT_REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json().get("data", [])
                models = sorted(m["id"] for m in data if m.get("id"))
                cls.models_cache = models if models else cls._fallback_models[:]
                cls.last_fetch_time = current_time
            except Exception as e:
                print(f"[RerankNode] Error fetching rerank models: {e}")
                if cls.models_cache is None:
                    cls.models_cache = cls._fallback_models[:]
        return cls.models_cache

    def rerank(self, api_key, query, documents, model,
               top_n, request_timeout):
        """
        Calls POST /v1/rerank with the documents list and returns results
        sorted by descending relevance.

        All parameters arrive as lists (INPUT_IS_LIST = True).
        Scalars are unwrapped with [0]; `documents` is used as-is.
        """
        # Unwrap scalar inputs
        api_key_str      = api_key[0]       if isinstance(api_key, list)       else api_key
        query_str        = query[0]         if isinstance(query, list)         else query
        model_str        = model[0]         if isinstance(model, list)         else model
        top_n_int        = top_n[0]         if isinstance(top_n, list)         else top_n
        timeout_val      = request_timeout[0] if isinstance(request_timeout, list) else request_timeout

        # documents is the full list
        doc_list = [d for d in (documents if isinstance(documents, list) else [documents])
                    if d and d.strip()]

        error_lists = ([], [], "Stats N/A", "")   # empty-list sentinel for list outputs

        api_key_str = shared.get_api_key(api_key_str)
        if not api_key_str:
            return ([], [], "Stats N/A",
                    "Error: API Key not provided. Set LLM_KEY env var or use openrouter_api_key.json")

        if not query_str or not query_str.strip():
            return ([], [], "Stats N/A", "Error: query is empty.")

        if not doc_list:
            return ([], [], "Stats N/A", "Error: documents list is empty.")

        validated_timeout = shared.validate_request_timeout(
            timeout_val, self.min_request_timeout,
            self.max_request_timeout, self.default_request_timeout
        )

        headers = shared.build_standard_headers(api_key_str)
        url = f"{shared.BASE_URL}/rerank"

        data = {
            "model": model_str,
            "query": query_str.strip(),
            "documents": doc_list,
        }
        if top_n_int and top_n_int > 0:
            data["top_n"] = top_n_int

        try:
            start_time = time.time()
            response = requests.post(
                url, headers=headers, json=data, timeout=validated_timeout
            )
            response.raise_for_status()
            elapsed = time.time() - start_time

            result  = response.json()
            results = result.get("results", [])

            if not results:
                return ([], [], f"Model: {model_str}, Elapsed: {elapsed:.2f}s — no results", "")

            # results are already sorted by relevance (highest first)
            ranked_texts  = [r["document"]["text"]  for r in results]
            ranked_scores = [float(r["relevance_score"]) for r in results]

            usage        = result.get("usage") or {}
            total_tokens = usage.get("total_tokens", 0)
            search_units = usage.get("search_units", 0)
            cost         = usage.get("cost", 0.0)

            stats = (
                f"Model: {model_str}, "
                f"Docs in: {len(doc_list)}, Docs out: {len(results)}, "
                f"Total tokens: {total_tokens}, Search units: {search_units}, "
                f"Cost: ${cost:.4f}, Elapsed: {elapsed:.2f}s"
            )

            credits = shared.fetch_credits(api_key_str, validated_timeout)
            return (ranked_texts, ranked_scores, stats, credits)

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
            print(f"[RerankNode] ERROR: {error_msg}")
            return ([], [], "Stats N/A due to error", error_msg)
        except Exception as e:
            print(f"[RerankNode] ERROR: {str(e)}")
            return ([], [], "Stats N/A due to error", f"Node Error: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, api_key, query, documents, model, top_n, request_timeout):
        """IS_CHANGED also receives lists when INPUT_IS_LIST = True."""
        def u(v):
            return v[0] if isinstance(v, list) else v

        return (
            u(api_key), u(query),
            tuple(documents) if isinstance(documents, list) else documents,
            u(model), u(top_n), u(request_timeout),
        )


NODE_CLASS_MAPPINGS = {
    "OpenRouterRerankNode": OpenRouterRerankNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterRerankNode": "OpenRouter Rerank Node"
}
