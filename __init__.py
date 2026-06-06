from .node import NODE_CLASS_MAPPINGS as _T, NODE_DISPLAY_NAME_MAPPINGS as _TN
from .node_image_gen import NODE_CLASS_MAPPINGS as _I, NODE_DISPLAY_NAME_MAPPINGS as _IN
from .node_speech import NODE_CLASS_MAPPINGS as _S, NODE_DISPLAY_NAME_MAPPINGS as _SN
from .node_transcription import NODE_CLASS_MAPPINGS as _TR, NODE_DISPLAY_NAME_MAPPINGS as _TRN
from .node_rerank import NODE_CLASS_MAPPINGS as _RR, NODE_DISPLAY_NAME_MAPPINGS as _RRN
from .node_video_gen import NODE_CLASS_MAPPINGS as _V, NODE_DISPLAY_NAME_MAPPINGS as _VN
from . import openrouter_shared as shared
import json

NODE_CLASS_MAPPINGS = {**_T, **_I, **_S, **_TR, **_RR, **_V}
NODE_DISPLAY_NAME_MAPPINGS = {**_TN, **_IN, **_SN, **_TRN, **_RRN, **_VN}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Register web route for fetching supported voices for a model
try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/openrouter/voices/{model}")
    async def get_model_voices(request):
        """Fetch supported voices for a given model."""
        model = request.match_info.get("model", "")
        if not model:
            return web.json_response({"error": "Model ID required"}, status=400)

        # Get API key from header (optional - some models have public voice lists)
        api_key = request.headers.get("Authorization", "").replace("Bearer ", "")

        # Fetch voices for the model
        voices = shared.get_model_supported_voices(model, api_key if api_key else None)

        return web.json_response({"voices": voices})

except ImportError:
    # ComfyUI or aiohttp not available - skip route registration
    pass
