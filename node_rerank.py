"""
node_rerank.py — OpenRouter Rerank Node

POST /api/v1/rerank

Request:
  model     — rerank model ID (e.g. cohere/rerank-v3.5)
  query     — search query string
  documents — list of document strings
  top_n     — optional; how many results to return (0 = all)

Response:
  results[].document.text    — original document text
  results[].index            — original position in input list
  results[].relevance_score  — float relevance score (higher = more relevant)

Documents are passed in / returned as separator-delimited strings so they
connect naturally to other ComfyUI text nodes.
"""

import requests
import json
import time
from . import openrouter_shared as shared


# Separator tokens used when splitting/joining document lists.
# Key = display name shown in the dropdown.
_SEPARATORS = {
    "newline":        "\n",
    "double newline": "\n\n",
    "---":            "\n---\n",
    "|||":            "|||",
}


class OpenRouterRerankNode:
    """
    ComfyUI node for document reranking via OpenRouter's /v1/rerank endpoint.

    Takes a query and a list of documents (as a delimited string), calls the
    rerank API, and returns the documents re-sorted by relevance.

    Returns four outputs:
      1) "documents"  : re-ranked document texts joined by the chosen separator
      2) "scores"     : relevance scores, one per line, matching document order
      3) "Stats"      : timing, model, usage (tokens, search units, cost)
      4) "Credits"    : remaining OpenRouter account balance
    """

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
                "documents": ("STRING", {
                    "multiline": True,
                    "default": "Document one\nDocument two\nDocument three",
                }),
                "model": (cls.fetch_openrouter_models(),),
                # 0 = return all results; >0 = return only the top N
                "top_n": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "display": "number",
                }),
                "separator": (list(_SEPARATORS.keys()), {"default": "newline"}),
                "request_timeout": ("INT", {
                    "default": cls.default_request_timeout,
                    "min": cls.min_request_timeout,
                    "max": cls.max_request_timeout,
                    "step": 1,
                    "display": "number",
                }),
            },
            "optional": {
                # Wire another node's STRING output here to override the text area
                "query_input":     ("STRING", {"forceInput": True}),
                "documents_input": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("documents", "scores", "Stats", "Credits")

    FUNCTION = "rerank"
    CATEGORY = "LLM"

    @classmethod
    def fetch_openrouter_models(cls):
        """
        Returns the list of rerank model IDs.

        Rerank models live on a separate endpoint (/api/v1/rerank) and don't
        appear in the output_modalities filter used by other nodes, so we
        maintain our own hardcoded list with 1-hour caching.
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
               top_n=0, separator="newline", request_timeout=120,
               query_input=None, documents_input=None):
        """
        Splits the documents string into a list, calls POST /v1/rerank,
        then returns the texts and scores re-sorted by relevance.

        Returns (ranked_docs_str, scores_str, stats_str, credits_str).
        """
        api_key = shared.get_api_key(api_key)
        if not api_key:
            return (
                "",
                "",
                "Stats N/A",
                "Error: API Key not provided. Set LLM_KEY env var or use openrouter_api_key.json",
            )

        effective_query = (
            query_input if query_input is not None and query_input.strip()
            else query
        )
        if not effective_query or not effective_query.strip():
            return ("", "", "Stats N/A", "Error: query is empty.")

        effective_docs_str = (
            documents_input if documents_input is not None and documents_input.strip()
            else documents
        )

        sep = _SEPARATORS.get(separator, "\n")
        doc_list = [d for d in effective_docs_str.split(sep) if d.strip()]
        if not doc_list:
            return ("", "", "Stats N/A", "Error: documents list is empty.")

        validated_timeout = shared.validate_request_timeout(
            request_timeout, self.min_request_timeout,
            self.max_request_timeout, self.default_request_timeout
        )

        headers = shared.build_standard_headers(api_key)
        url = f"{shared.BASE_URL}/rerank"

        data = {
            "model": model,
            "query": effective_query.strip(),
            "documents": doc_list,
        }
        if top_n and top_n > 0:
            data["top_n"] = top_n

        try:
            start_time = time.time()
            response = requests.post(
                url, headers=headers, json=data, timeout=validated_timeout
            )
            response.raise_for_status()
            elapsed = time.time() - start_time

            result = response.json()
            results = result.get("results", [])

            if not results:
                return ("", "", f"Model: {model}, Elapsed: {elapsed:.2f}s", "No results returned.")

            # results are already sorted by relevance (highest first)
            ranked_texts = [r["document"]["text"] for r in results]
            ranked_scores = [r["relevance_score"] for r in results]

            docs_out = sep.join(ranked_texts)
            scores_out = "\n".join(f"{s:.6f}" for s in ranked_scores)

            usage = result.get("usage") or {}
            total_tokens = usage.get("total_tokens", 0)
            search_units = usage.get("search_units", 0)
            cost = usage.get("cost", 0.0)

            stats = (
                f"Model: {model}, "
                f"Docs in: {len(doc_list)}, Docs out: {len(results)}, "
                f"Total tokens: {total_tokens}, Search units: {search_units}, "
                f"Cost: ${cost:.4f}, Elapsed: {elapsed:.2f}s"
            )

            credits = shared.fetch_credits(api_key, validated_timeout)
            return (docs_out, scores_out, stats, credits)

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
            return ("", "", "Stats N/A due to error", error_msg)
        except Exception as e:
            print(f"[RerankNode] ERROR: {str(e)}")
            return ("", "", "Stats N/A due to error", f"Node Error: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, api_key, query, documents, model,
                   top_n=0, separator="newline", request_timeout=120,
                   query_input=None, documents_input=None):
        """Cache key covers all inputs that affect the rerank result."""
        try:
            timeout_int = int(request_timeout)
            timeout_int = max(cls.min_request_timeout, min(cls.max_request_timeout, timeout_int))
        except (ValueError, TypeError):
            timeout_int = cls.default_request_timeout

        effective_query = query_input if (query_input and query_input.strip()) else query
        effective_docs  = documents_input if (documents_input and documents_input.strip()) else documents

        return (api_key, model, effective_query, effective_docs, top_n, separator, timeout_int)


NODE_CLASS_MAPPINGS = {
    "OpenRouterRerankNode": OpenRouterRerankNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterRerankNode": "OpenRouter Rerank Node"
}
