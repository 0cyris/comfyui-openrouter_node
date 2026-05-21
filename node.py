import requests
import json
import time
import base64
import hashlib
import torch
from . import openrouter_shared as shared
from .chat_manager import ChatSessionManager

# Define a placeholder type name for PDF data.
# The actual input connection will accept '*' but we check the structure.
# Expecting a dictionary: {"filename": str, "bytes": bytes}
PDF_DATA_TYPE = "*"  # Use '*' to accept any type, check structure later

class OpenRouterNode:
    """
    A node for interacting with OpenRouter's chat/completion API.
    Supports text, images, and PDFs as input.
    Returns four outputs:
      1) "Output": the text response from the LLM
      2) "image": an image tensor if the response contains an image, else empty tensor
      3) "Stats": a string detailing tokens per second, input tokens, and output tokens
      4) "Credits": a string showing your remaining OpenRouter account balance
    """

    models_cache = None
    last_fetch_time = 0
    cache_duration = 3600  # Cache duration in seconds (1 hour)
    default_request_timeout = 120
    min_request_timeout = 1
    max_request_timeout = 3600
    reasoning_effort_options = ("auto", "none", "minimal", "low", "medium", "high", "xhigh")
    default_reasoning_effort = "auto"

    def __init__(self):
        self.chat_manager = ChatSessionManager()

    @staticmethod
    def get_api_key(api_key_ui):
        """
        Resolves the API key from:
        1. UI input field (if not empty)
        2. Environment variable 'LLM_KEY'
        3. config file 'openrouter_api_key.json' in node directory
        """
        return shared.get_api_key(api_key_ui)

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input specification for this node.
        Includes optional inputs for image and PDF data.
        """
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant."
                }),
                "user_message_box": ("STRING", {
                    "multiline": True,
                    "default": "Hello, how are you?"
                }),
                "model": (cls.fetch_openrouter_models(),),
                "web_search": ("BOOLEAN", {"default": False}),
                "cheapest": ("BOOLEAN", {"default": True}),
                "fastest": ("BOOLEAN", {"default": False}),
                "aspect_ratio": (shared.ASPECT_RATIO_OPTIONS, {"default": "auto"}),
                "image_resolution": (shared.IMAGE_RESOLUTION_OPTIONS, {"default": "1K"}),
                "reasoning_effort": (list(cls.reasoning_effort_options), {"default": cls.default_reasoning_effort}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed"}),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider",
                    "round": 0.01,
                }),
                "pdf_engine": (["auto", "mistral-ocr", "pdf-text"], {"default": "auto"}),
                "chat_mode": ("BOOLEAN", {"default": False}),
                "request_timeout": ("INT", {
                    "default": cls.default_request_timeout,
                    "min": cls.min_request_timeout,
                    "max": cls.max_request_timeout,
                    "step": 1,
                    "display": "number",
                }),
            },
            "optional": {
                "pdf_data": (PDF_DATA_TYPE,),  # Use '*' and check structure in generate_response
                "user_message_input": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING",)
    RETURN_NAMES = ("Output", "image", "Stats", "Credits")

    FUNCTION = "generate_response"
    CATEGORY = "LLM"

    @classmethod
    def fetch_openrouter_models(cls):
        """
        Fetches text-output model IDs from the OpenRouter API, with caching.
        Filters to only models that support text output.
        """
        raw, was_refreshed = shared.fetch_all_models_raw()
        if cls.models_cache is None or was_refreshed:
            filtered = shared.filter_models_by_output(raw, "text")
            cls.models_cache = filtered if filtered else [
                "error_fetching_models", "google/gemma-3-27b-it",
                "openai/gpt-4o", "anthropic/claude-3.5-sonnet"
            ]
        return cls.models_cache

    def validate_temperature(self, temperature):
        """
        Validates and converts temperature value to float within acceptable range.
        """
        return shared.validate_temperature(temperature)

    def validate_request_timeout(self, request_timeout):
        """
        Validates and converts request timeout to seconds within an acceptable range.
        """
        return shared.validate_request_timeout(
            request_timeout,
            self.min_request_timeout,
            self.max_request_timeout,
            self.default_request_timeout,
        )

    @classmethod
    def validate_reasoning_effort(cls, reasoning_effort):
        """
        Validates OpenRouter reasoning effort. "auto" means do not send a
        reasoning override and let OpenRouter/model defaults apply.
        """
        return shared.validate_reasoning_effort(
            reasoning_effort,
            cls.reasoning_effort_options,
            cls.default_reasoning_effort,
        )

    def fetch_credits(self, api_key, timeout=None):
        """
        Fetches the user's credits information from the OpenRouter API.
        Returns a formatted string with remaining credits.
        """
        api_key = self.get_api_key(api_key)
        if not api_key:
            return "API Key not provided."
        return shared.fetch_credits(api_key, timeout)

    def generate_response(self, api_key, system_prompt, user_message_box, model,
                         web_search, cheapest, fastest, temperature, pdf_engine, chat_mode,
                         request_timeout=120, aspect_ratio="auto", image_resolution="1K", seed=0,
                         pdf_data=None, user_message_input=None, reasoning_effort="auto", **kwargs):
        """
        Sends a completion request to the OpenRouter chat completion endpoint.
        Handles text, optional image, and optional PDF inputs.

        Returns four outputs:
          (1) Output: the LLM's text response
          (2) image: an image tensor if the response contains an image, else empty tensor
          (3) Stats: a string with tokens per second, prompt tokens, completion tokens
          (4) Credits: a string with the user's credit information
        """
        # Create empty placeholder image
        placeholder_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

        # Resolve API key
        api_key = self.get_api_key(api_key)

        if not api_key:
            return ("Error: API Key not provided. Set LLM_KEY env var or use openrouter_api_key.json", placeholder_image, "Stats N/A", "Credits N/A")

        url = f"{shared.BASE_URL}/chat/completions"
        headers = shared.build_standard_headers(api_key)

        # Validate and convert temperature
        validated_temp = self.validate_temperature(temperature)
        validated_timeout = self.validate_request_timeout(request_timeout)
        validated_reasoning_effort = self.validate_reasoning_effort(reasoning_effort)

        # Decide whether to use user_message_input or user_message_box
        user_text = user_message_input if user_message_input is not None and user_message_input.strip() else user_message_box

        # Initialize session_path
        session_path = None

        # Handle chat mode
        if chat_mode:
            # Get or create a chat session
            session_path, messages = self.chat_manager.get_or_create_session(user_text, system_prompt)

            # Check if we need to update the system prompt (for existing sessions)
            if messages and messages[0]["role"] == "system" and messages[0]["content"] != system_prompt:
                # Update system prompt if it has changed
                messages[0]["content"] = system_prompt
        else:
            # Non-chat mode: Build the messages array, starting with a system prompt.
            messages = [
                {"role": "system", "content": system_prompt},
            ]

        # --- Build the user message content ---
        user_content_blocks = []

        # 1. Add Text part (always present)
        user_content_blocks.append({
            "type": "text",
            "text": user_text
        })

        # 2. Add Image parts (optional) - support multiple images from kwargs
        # Process all image_N inputs from kwargs
        image_keys = sorted([k for k in kwargs.keys() if k.startswith('image_')],
                           key=lambda x: int(x.split('_')[1]))

        for image_key in image_keys:
            if kwargs[image_key] is not None:
                try:
                    img_str = self.image_to_base64(kwargs[image_key])
                    user_content_blocks.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}"
                        }
                    })
                except Exception as e:
                    print(f"Error processing {image_key}: {e}")
                    return (f"Error processing {image_key}: {e}", placeholder_image, "Stats N/A", "Credits N/A")

        # 3. Add PDF part (optional)
        pdf_filename = "document.pdf"  # Default filename if not provided
        if pdf_data is not None:
            # Validate pdf_data structure (expecting dict with 'filename' and 'bytes')
            if isinstance(pdf_data, dict) and "bytes" in pdf_data and isinstance(pdf_data["bytes"], bytes):
                pdf_bytes = pdf_data["bytes"]
                # Use provided filename if available and valid, otherwise use default
                if "filename" in pdf_data and isinstance(pdf_data["filename"], str) and pdf_data["filename"].strip():
                    pdf_filename = pdf_data["filename"]

                try:
                    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                    data_url = f"data:application/pdf;base64,{base64_pdf}"
                    user_content_blocks.append({
                        "type": "file",
                        "file": {
                            "filename": pdf_filename,
                            "file_data": data_url
                        }
                    })
                except Exception as e:
                    print(f"Error encoding PDF: {e}")
                    return (f"Error encoding PDF: {e}", placeholder_image, "Stats N/A", "Credits N/A")
            else:
                # Handle case where pdf_data is not in the expected format
                print(f"Warning: pdf_data input is not in the expected format (dict with 'filename' and 'bytes'). PDF not included.")

        # Determine message format based on content type
        # Use simple string format for text-only requests to ensure compatibility
        # Use structured format only when we have multimodal content
        has_multimodal_content = len(user_content_blocks) > 1 or any(block.get("type") != "text" for block in user_content_blocks)

        if has_multimodal_content:
            # Use structured format for multimodal content
            new_user_message = {
                "role": "user",
                "content": user_content_blocks
            }
        else:
            # Use simple string format for text-only requests
            new_user_message = {
                "role": "user",
                "content": user_text
            }

        if chat_mode:
            # In chat mode, append to existing conversation (but don't save yet - wait for response)
            messages.append(new_user_message)
        else:
            # In non-chat mode, messages array already has system prompt, just append user message
            messages.append(new_user_message)

        # --- Apply model modifiers ---
        modified_model = shared.apply_model_modifiers(model, web_search, cheapest, fastest)

        # --- Construct the final payload ---
        data = {
            "model": modified_model,
            "messages": messages,
            "temperature": validated_temp,
            "seed": seed
        }
        if validated_reasoning_effort != "auto":
            data["reasoning"] = {"effort": validated_reasoning_effort}

        print(f"Payload: model={modified_model}")

        # Add plugins if a specific PDF engine is selected
        if pdf_engine != "auto":
            data["plugins"] = [
                {
                    "id": "file-parser",
                    "pdf": {
                        "engine": pdf_engine
                    }
                }
            ]

        # --- Pre-calculate text input tokens (rough estimate) ---
        text_token_estimate = 0
        try:
            text_token_estimate = self.count_tokens(system_prompt, model) + self.count_tokens(user_text, model)
        except Exception as e:
            print(f"Warning: Token counting failed - {e}")

        # --- Make API Call and Process Response ---
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=data, timeout=validated_timeout)
            response.raise_for_status()
            end_time = time.time()

            result = response.json()
            # Debug: print truncated response to see what OpenRouter returned
            debug_str = json.dumps(result, default=str)
            print(f"API response ({len(debug_str)} chars): {debug_str[:500]}")

            # --- Extract results and calculate stats ---
            if not result.get("choices") or not result["choices"][0].get("message"):
                raise ValueError("Invalid response format from API: 'choices' or 'message' missing.")

            # Parse response for text and image content
            message = result["choices"][0]["message"]
            text_output = message.get("content", "")
            image_tensor = placeholder_image

            # Check for images in the separate images field (OpenRouter format)
            if message.get("images"):
                print(f"Found {len(message['images'])} image(s) in API response")
                try:
                    # Get the first image from the images array
                    first_image = message["images"][0]
                    image_url = first_image["image_url"]["url"]

                    if image_url.startswith("data:image"):
                        base64_str = image_url.split(",", 1)[1]
                        try:
                            # Convert base64 to image tensor
                            image_tensor = self.base64_to_image(base64_str)
                            print(f"Successfully decoded image from API response")
                        except Exception as e:
                            print(f"Error decoding image: {e}")
                    else:
                        print(f"Image URL format not supported: {image_url[:50]}...")
                except Exception as e:
                    print(f"Error processing images from response: {e}")
            else:
                print("No images found in API response - this may be normal if the model doesn't support image generation or the prompt didn't request an image")

            # Also handle legacy multimodal content format as fallback
            if isinstance(text_output, list):
                text_parts = []
                for content in text_output:
                    if isinstance(content, dict):
                        if content.get("type") == "text":
                            text_parts.append(content.get("text", ""))
                        elif content.get("type") == "image_url":
                            # Extract base64 image data
                            image_url = content["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                base64_str = image_url.split(",", 1)[1]
                                try:
                                    # Convert base64 to image tensor
                                    image_tensor = self.base64_to_image(base64_str)
                                except Exception as e:
                                    print(f"Error decoding image: {e}")
                text_output = "\n".join(text_parts)

            response_ms = result.get("response_ms", None)
            api_usage = result.get("usage", {})
            prompt_tokens = api_usage.get("prompt_tokens", text_token_estimate)
            completion_tokens = api_usage.get("completion_tokens", 0)
            if completion_tokens == 0 and text_output:
                try:
                    completion_tokens = self.count_tokens(text_output, model)
                except Exception as e:
                    print(f"Warning: Completion token counting failed - {e}")

            # Calculate tokens per second (TPS)
            tps = 0
            elapsed_time = end_time - start_time
            if response_ms is not None:
                server_elapsed_time = response_ms / 1000.0
                if server_elapsed_time > 0:
                    tps = completion_tokens / server_elapsed_time
            elif elapsed_time > 0:
                tps = completion_tokens / elapsed_time

            extra_parts = []
            if pdf_engine != "auto":
                extra_parts.append(f"PDF Engine: {pdf_engine}")
            if validated_reasoning_effort != "auto":
                extra_parts.append(f"Reasoning: {validated_reasoning_effort}")

            stats_text = shared.build_stats_text(
                tps, prompt_tokens, completion_tokens, validated_temp, modified_model,
                extra_parts if extra_parts else None
            )

            # Fetch credits information AFTER the main request
            credits_text = self.fetch_credits(api_key, timeout=validated_timeout)

            # Save conversation in chat mode
            if chat_mode and session_path:
                # Append assistant's response to the conversation
                assistant_message = {
                    "role": "assistant",
                    "content": text_output
                }
                messages.append(assistant_message)

                # Save the updated conversation
                self.chat_manager.save_conversation(session_path, messages)

            return (text_output, image_tensor, stats_text, credits_text)

        except requests.exceptions.RequestException as e:
            error_message = f"API Request Error: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_message += f" | Details: {error_detail}"
                except json.JSONDecodeError:
                    error_message += f" | Status: {e.response.status_code} | Response: {e.response.text[:200]}"
            else:
                error_message += " (Network or connection issue)"
            print(f"ERROR: {error_message}")
            return (error_message, placeholder_image, "Stats N/A due to error", "Credits N/A due to error")
        except Exception as e:
            print(f"ERROR: Node Error: {str(e)}")
            return (f"Node Error: {str(e)}", placeholder_image, "Stats N/A due to error", "Credits N/A due to error")

    @staticmethod
    def image_to_base64(image):
        """
        Converts a ComfyUI IMAGE (torch.Tensor, BHWC, float 0-1)
        into a base64-encoded PNG string.
        """
        return shared.image_to_base64(image)

    @staticmethod
    def base64_to_image(base64_str: str) -> torch.Tensor:
        """
        Converts a base64 image string to a ComfyUI image tensor
        Returns tensor in [1, H, W, 3] format with values in [0, 1]
        """
        return shared.base64_to_image(base64_str)

    @staticmethod
    def count_tokens(text, model):
        """
        Count tokens for a given text using tiktoken.
        Uses model-specific encodings where possible, falls back to cl100k_base.
        """
        return shared.count_tokens(text, model)

    @classmethod
    def IS_CHANGED(cls, api_key, system_prompt, user_message_box, model,
                   web_search, cheapest, fastest, temperature, pdf_engine, chat_mode,
                   request_timeout=120, aspect_ratio="auto", image_resolution="1K", seed=0,
                   pdf_data=None, user_message_input=None, reasoning_effort="auto", **kwargs):
        """
        Check if any input that affects the output has changed.
        Includes hashing image and PDF data.
        """
        # Hash image data if present - handle multiple images from kwargs
        image_hashes = []
        image_keys = sorted([k for k in kwargs.keys() if k.startswith('image_')],
                           key=lambda x: int(x.split('_')[1]))

        for image_key in image_keys:
            if kwargs[image_key] is not None:
                image = kwargs[image_key]
                if isinstance(image, torch.Tensor):
                    try:
                        hasher = hashlib.sha256()
                        hasher.update(image.cpu().numpy().tobytes())
                        image_hashes.append(hasher.hexdigest())
                    except Exception as e:
                        print(f"Warning: Could not hash {image_key} data for IS_CHANGED: {e}")
                        image_hashes.append(f"{image_key}_hashing_error")
                else:
                    image_hashes.append(None)

        # Hash PDF data if present and valid
        pdf_hash = None
        if pdf_data is not None and isinstance(pdf_data, dict) and "bytes" in pdf_data and isinstance(pdf_data["bytes"], bytes):
            try:
                hasher = hashlib.sha256()
                hasher.update(pdf_data["bytes"])
                pdf_hash = hasher.hexdigest()
            except Exception as e:
                print(f"Warning: Could not hash pdf data for IS_CHANGED: {e}")
                pdf_hash = "pdf_hashing_error"
        elif pdf_data is not None:
            pdf_hash = "invalid_pdf_data_format"

        # Ensure temperature is consistently represented
        try:
            temp_float = float(temperature) if isinstance(temperature, (str, int, float)) else 1.0
            temp_float = max(0.0, min(2.0, temp_float))
        except (ValueError, TypeError):
            temp_float = 1.0

        try:
            timeout_int = int(request_timeout)
            timeout_int = max(cls.min_request_timeout, min(cls.max_request_timeout, timeout_int))
        except (ValueError, TypeError):
            timeout_int = cls.default_request_timeout

        validated_reasoning_effort = cls.validate_reasoning_effort(reasoning_effort)

        # Combine all relevant inputs into a tuple for comparison
        # Note: api_key here is the UI value only. Keys resolved from the LLM_KEY
        # env var or openrouter_api_key.json are intentionally NOT part of the
        # cache key — they're treated as user environment, not workflow inputs.
        return (api_key, system_prompt, user_message_box, model,
                web_search, cheapest, fastest, temp_float, pdf_engine, chat_mode,
                timeout_int, aspect_ratio, image_resolution, seed, validated_reasoning_effort,
                tuple(image_hashes), pdf_hash, user_message_input)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "OpenRouterNode": OpenRouterNode
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterNode": "OpenRouter LLM Node (Text/Multi-Image/PDF/Chat)"
}
