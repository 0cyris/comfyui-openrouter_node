from .node import NODE_CLASS_MAPPINGS as _T, NODE_DISPLAY_NAME_MAPPINGS as _TN
from .node_image_gen import NODE_CLASS_MAPPINGS as _I, NODE_DISPLAY_NAME_MAPPINGS as _IN
from .node_speech import NODE_CLASS_MAPPINGS as _S, NODE_DISPLAY_NAME_MAPPINGS as _SN
from .node_transcription import NODE_CLASS_MAPPINGS as _TR, NODE_DISPLAY_NAME_MAPPINGS as _TRN
from .node_rerank import NODE_CLASS_MAPPINGS as _RR, NODE_DISPLAY_NAME_MAPPINGS as _RRN
from .node_video_gen import NODE_CLASS_MAPPINGS as _V, NODE_DISPLAY_NAME_MAPPINGS as _VN
from .node_image_api import NODE_CLASS_MAPPINGS as _IA, NODE_DISPLAY_NAME_MAPPINGS as _IAN
from .node_decision import NODE_CLASS_MAPPINGS as _D, NODE_DISPLAY_NAME_MAPPINGS as _DN
from . import openrouter_shared as shared
import json

NODE_CLASS_MAPPINGS = {**_T, **_I, **_IA, **_D, **_S, **_TR, **_RR, **_V}
NODE_DISPLAY_NAME_MAPPINGS = {**_TN, **_IN, **_IAN, **_DN, **_SN, **_TRN, **_RRN, **_VN}

for node in NODE_CLASS_MAPPINGS.values():
    node.fetch_openrouter_models()

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Register web route for fetching supported voices for a model
def _register_routes():
    """Register web routes with ComfyUI's PromptServer."""
    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get("/openrouter/voices/{model}")
        async def get_model_voices(request):
            """Fetch supported voices for a given model."""
            model = request.match_info.get("model", "")
            if not model:
                return web.json_response({"error": "Model ID required"}, status=400)

            voices = shared.get_model_supported_voices(model, shared.get_api_key())

            print(f"[OpenRouter] Voices endpoint called for model: {model}, found {len(voices)} voices")
            return web.json_response({"voices": voices})

        PromptServer.instance.add_on_prompt_handler(lambda json_data: shared.blank_prompt_api_keys(json_data, NODE_CLASS_MAPPINGS))

    except ImportError as e:
        print(f"[OpenRouter] Warning: Could not register voice routes: {e}")
        pass

# Try to register routes immediately
_register_routes()
