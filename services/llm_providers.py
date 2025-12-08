"""
LLM Provider Management for LearnForge.

Supports multiple LLM providers with search and thinking capabilities.
Add new models by adding to MODELS dict below.
"""

import os
from typing import Optional, Generator
from dotenv import load_dotenv

load_dotenv()

# ===================
# MODEL REGISTRY
# ===================
# Add new models here. Format:
# "display_name": {
#     "provider": "provider_name",
#     "model_id": "actual_model_id",
#     "description": "short description",
#     "supports_search": True/False,
#     "supports_thinking": True/False,
#     "search_always_on": True/False  # For models like compound where search can't be disabled
# }

MODELS = {
    "Gemini 2.5 Flash": {
        "provider": "gemini",
        "model_id": "gemini-flash-latest",
        "description": "Powerful Gemini reasoning model",
        "supports_search": True,
        "supports_thinking": True,
        "search_always_on": False
    },
    "Gemini 2.5 Flash Lite": {
        "provider": "gemini",
        "model_id": "gemini-flash-lite-latest",
        "description": "Fast, lightweight Gemini model",
        "supports_search": True,
        "supports_thinking": True,
        "search_always_on": False
    },
    "Mistral Medium": {
        "provider": "mistral",
        "model_id": "mistral-medium-latest",
        "description": "Mistral's powerful medium model with web search",
        "supports_search": True,
        "supports_thinking": False,
        "search_always_on": False
    },
    "Groq Compound": {
        "provider": "groq",
        "model_id": "compound-beta",
        "description": "Groq's compound AI with web search",
        "supports_search": True,
        "supports_thinking": False,
        "search_always_on": True  # Search is always enabled for compound
    },
    "Llama 4 Scout 17B": {
        "provider": "groq",
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "description": "Llama 4 Scout 17B on Groq",
        "supports_search": False,
        "supports_thinking": False,
        "search_always_on": False
    },
    "Llama 3.3 70B": {
        "provider": "cerebras",
        "model_id": "llama-3.3-70b",
        "description": "Llama 3.3 70B on Cerebras - ultra fast inference",
        "supports_search": False,
        "supports_thinking": False,
        "search_always_on": False
    },
    "Nova Canvas": {
        "provider": "nova",
        "model_id": "bedrock-amazon.nova-canvas-v1-0",
        "description": "Amazon Nova Canvas for image generation",
        "supports_search": False,
        "supports_thinking": False,
        "search_always_on": False,
        "type": "image",
        "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5"]
    },
}

# Default model
DEFAULT_MODEL = "Gemini 2.5 Flash"


# ===================
# PROVIDER CLIENTS
# ===================

class GeminiProvider:
    """Google Gemini provider with search and thinking support."""

    def __init__(self):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        self.client = genai.Client(api_key=api_key)

    def generate(
        self,
        model_id: str,
        prompt: str,
        use_search: bool = False,
        use_thinking: bool = False
    ) -> str:
        from google.genai import types

        # Build tools list
        tools = []
        if use_search:
            tools.append(types.Tool(google_search=types.GoogleSearch()))

        # Build config
        config_params = {}

        # Thinking config
        if use_thinking:
            config_params["thinking_config"] = types.ThinkingConfig(
                thinking_budget=8192
            )
        else:
            config_params["thinking_config"] = types.ThinkingConfig(
                thinking_budget=0
            )

        # Add tools if any
        if tools:
            config_params["tools"] = tools

        config = types.GenerateContentConfig(**config_params)

        # Generate
        response = self.client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=config
        )

        return response.text

    def generate_stream(
        self,
        model_id: str,
        prompt: str,
        use_search: bool = False,
        use_thinking: bool = False
    ) -> Generator[dict, None, None]:
        """
        Stream response with thinking support.

        Yields dicts:
            - {"type": "thinking", "content": "..."} for thinking chunks
            - {"type": "text", "content": "..."} for text chunks
        """
        from google.genai import types

        # Build tools list
        tools = []
        if use_search:
            tools.append(types.Tool(google_search=types.GoogleSearch()))

        # Build config
        config_params = {}

        # Thinking config - include_thoughts=True to get thinking content in response
        if use_thinking:
            config_params["thinking_config"] = types.ThinkingConfig(
                thinking_budget=8192,
                include_thoughts=True
            )

        # Add tools if any
        if tools:
            config_params["tools"] = tools

        config = types.GenerateContentConfig(**config_params)

        # Stream generate
        for chunk in self.client.models.generate_content_stream(
            model=model_id,
            contents=prompt,
            config=config
        ):
            for part in chunk.candidates[0].content.parts:
                if part.thought:
                    yield {"type": "thinking", "content": part.text}
                else:
                    if part.text:
                        yield {"type": "text", "content": part.text}

    def generate_image(
        self,
        model_id: str,
        prompt: str,
        aspect_ratio: str = "auto"
    ) -> dict:
        """
        Generate image using Gemini Flash Image model.

        Returns:
            dict with 'image_base64' and 'mime_type'
        """
        import base64
        from google.genai import types

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]

        # Build config
        config_params = {
            "response_modalities": ["IMAGE", "TEXT"],
        }

        # Add aspect ratio config if not auto
        if aspect_ratio and aspect_ratio != "auto":
            config_params["image_config"] = types.ImageConfig(
                aspect_ratio=aspect_ratio,
            )

        config = types.GenerateContentConfig(**config_params)

        # Stream and collect image data
        image_data = None
        mime_type = None

        for chunk in self.client.models.generate_content_stream(
            model=model_id,
            contents=contents,
            config=config,
        ):
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue

            for part in chunk.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    image_data = part.inline_data.data
                    mime_type = part.inline_data.mime_type
                    break

            if image_data:
                break

        if not image_data:
            raise ValueError("No image generated")

        # Convert to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        return {
            "image_base64": image_base64,
            "mime_type": mime_type or "image/png"
        }


class GroqProvider:
    """Groq provider. Compound models have built-in search."""

    def __init__(self):
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        self.client = Groq(api_key=api_key)

    def generate(
        self,
        model_id: str,
        prompt: str,
        use_search: bool = False,
        use_thinking: bool = False
    ) -> str:
        # Note: compound-beta has search built-in, can't be toggled
        # Regular models don't support search/thinking
        response = self.client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def generate_stream(
        self,
        model_id: str,
        prompt: str,
        use_search: bool = False,
        use_thinking: bool = False
    ) -> Generator[dict, None, None]:
        stream = self.client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {"type": "text", "content": chunk.choices[0].delta.content}


class MistralProvider:
    """Mistral provider with web search support via conversations API."""

    def __init__(self):
        from mistralai import Mistral
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set in .env")
        self.client = Mistral(api_key=api_key)

    def generate(
        self,
        model_id: str,
        prompt: str,
        use_search: bool = False,
        use_thinking: bool = False
    ) -> str:
        # Build tools list
        tools = [{"type": "web_search"}] if use_search else []

        response = self.client.beta.conversations.start(
            inputs=[{"role": "user", "content": prompt}],
            model=model_id,
            instructions="",
            completion_args={
                "temperature": 0.7,
                "max_tokens": 4096,
                "top_p": 1
            },
            tools=tools
        )

        # Extract text from conversation response
        if hasattr(response, 'outputs') and response.outputs:
            for output in response.outputs:
                # output.content might be a list of content blocks
                if hasattr(output, 'content'):
                    content = output.content
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        # Extract text from content blocks
                        text_parts = []
                        for block in content:
                            if hasattr(block, 'text'):
                                text_parts.append(block.text)
                            elif isinstance(block, dict) and 'text' in block:
                                text_parts.append(block['text'])
                            elif isinstance(block, str):
                                text_parts.append(block)
                        return ''.join(text_parts)
        return str(response)

    def generate_stream(
        self,
        model_id: str,
        prompt: str,
        use_search: bool = False,
        use_thinking: bool = False
    ) -> Generator[dict, None, None]:
        if use_search:
            # Conversations API doesn't support streaming well, fall back to non-streaming
            result = self.generate(model_id, prompt, use_search=True, use_thinking=use_thinking)
            yield {"type": "text", "content": result}
        else:
            # Use regular chat streaming without search
            stream = self.client.chat.stream(
                model=model_id,
                messages=[{"role": "user", "content": prompt}]
            )
            for event in stream:
                if event.data.choices[0].delta.content:
                    yield {"type": "text", "content": event.data.choices[0].delta.content}


class CerebrasProvider:
    """Cerebras provider for ultra-fast Llama inference."""

    def __init__(self):
        from cerebras.cloud.sdk import Cerebras
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY not set in .env")
        self.client = Cerebras(api_key=api_key)

    def generate(
        self,
        model_id: str,
        prompt: str,
        use_search: bool = False,
        use_thinking: bool = False
    ) -> str:
        response = self.client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4096,
            temperature=0.2,
            top_p=1
        )
        return response.choices[0].message.content

    def generate_stream(
        self,
        model_id: str,
        prompt: str,
        use_search: bool = False,
        use_thinking: bool = False
    ) -> Generator[dict, None, None]:
        stream = self.client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4096,
            temperature=0.2,
            top_p=1,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {"type": "text", "content": chunk.choices[0].delta.content}


class NovaCanvasProvider:
    """Amazon Nova Canvas provider via PartyRock API."""

    def __init__(self):
        import json
        self.base_url = "https://partyrock.aws"
        self.cookies = self._load_cookies()

    def _load_cookies(self) -> dict:
        """Load cookies from cookie.txt file."""
        import json
        import os

        cookie_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cookie.txt")

        if not os.path.exists(cookie_path):
            raise ValueError(f"cookie.txt not found at {cookie_path}. Please export PartyRock cookies.")

        with open(cookie_path, 'r') as f:
            cookies_data = json.load(f)

        cookies = {}
        for cookie in cookies_data:
            if cookie['name'] in ['aws-waf-token', 's_sq', 'pr_refresh_token', 'idToken', 's_fid']:
                cookies[cookie['name']] = cookie['value']

        return cookies

    def _get_dimensions(self, aspect_ratio: str) -> tuple:
        """Get width/height for aspect ratio."""
        aspect_map = {
            "16:9": (1280, 720),
            "9:16": (720, 1280),
            "2:3": (768, 1152),
            "3:2": (1152, 768),
            "1:1": (512, 512),
            "3:4": (768, 1024),
            "4:3": (1024, 768),
            "4:5": (768, 960),
            "5:4": (960, 768),
        }
        return aspect_map.get(aspect_ratio, (512, 512))

    def generate_image(
        self,
        model_id: str,
        prompt: str,
        aspect_ratio: str = "1:1"
    ) -> dict:
        """Generate image using Nova Canvas via PartyRock."""
        import httpx
        import re
        from fake_useragent import UserAgent

        image_url = f"{self.base_url}/image"
        post_url = f"{self.base_url}/api/generateImage"
        ua = UserAgent()

        with httpx.Client(cookies=self.cookies, timeout=60.0) as client:
            # Get CSRF token
            response = client.get(image_url)
            match = re.search(r'<meta name="anti-csrftoken-a2z" value="([^"]+)"', response.text)
            csrf_token = match.group(1) if match else None

            if not csrf_token:
                raise ValueError("Could not extract CSRF token from PartyRock")

            headers = {
                "Content-Type": "application/json",
                "Origin": self.base_url,
                "Referer": image_url,
                "User-Agent": ua.random,
                "anti-csrftoken-a2z": csrf_token,
            }

            width, height = self._get_dimensions(aspect_ratio)

            payload = {
                "prompt": prompt,
                "negativePrompt": "",
                "options": {
                    "model": model_id,
                    "cfgScale": 6,
                    "height": height,
                    "width": width
                },
                "context": {
                    "type": "image-playground"
                }
            }

            response = client.post(post_url, headers=headers, json=payload)
            response_json = response.json()

            if "result" not in response_json or "data" not in response_json["result"]:
                raise ValueError(f"Unexpected response: {response_json}")

            image_base64 = response_json["result"]["data"]["imageBase64"]

            # Strip data URL prefix if present
            if image_base64.startswith("data:image/"):
                image_base64 = image_base64.split(",")[1]

            return {
                "image_base64": image_base64,
                "mime_type": "image/jpeg"
            }


# Provider registry
PROVIDERS = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "mistral": MistralProvider,
    "cerebras": CerebrasProvider,
    "nova": NovaCanvasProvider,
}


# ===================
# LLM MANAGER
# ===================

class LLMManager:
    """
    Manages LLM providers and model switching with search/thinking support.

    Usage:
        manager = LLMManager()
        manager.set_model("Llama 3.1 8B")
        response = manager.generate("What is AI?", use_search=True)
    """

    def __init__(self):
        self.current_model_name = DEFAULT_MODEL
        self._providers = {}  # Lazy-loaded providers

    def _get_provider(self, provider_name: str):
        """Get or create provider instance."""
        if provider_name not in self._providers:
            if provider_name not in PROVIDERS:
                raise ValueError(f"Unknown provider: {provider_name}")
            self._providers[provider_name] = PROVIDERS[provider_name]()
        return self._providers[provider_name]

    def set_model(self, model_name: str) -> bool:
        """Switch to a different model."""
        if model_name not in MODELS:
            return False
        self.current_model_name = model_name
        return True

    def get_current_model(self) -> dict:
        """Get current model info with capabilities."""
        model_info = MODELS[self.current_model_name]
        return {
            "name": self.current_model_name,
            "provider": model_info["provider"],
            "model_id": model_info["model_id"],
            "description": model_info["description"],
            "supports_search": model_info.get("supports_search", False),
            "supports_thinking": model_info.get("supports_thinking", False),
            "search_always_on": model_info.get("search_always_on", False),
            "type": model_info.get("type", "text"),
            "aspect_ratios": model_info.get("aspect_ratios", [])
        }

    def list_models(self) -> list:
        """List all available models with capabilities."""
        return [
            {
                "name": name,
                "provider": info["provider"],
                "description": info["description"],
                "supports_search": info.get("supports_search", False),
                "supports_thinking": info.get("supports_thinking", False),
                "search_always_on": info.get("search_always_on", False),
                "type": info.get("type", "text"),
                "aspect_ratios": info.get("aspect_ratios", []),
                "is_current": name == self.current_model_name
            }
            for name, info in MODELS.items()
        ]

    def generate(
        self,
        prompt: str,
        use_search: bool = False,
        use_thinking: bool = False
    ) -> str:
        """Generate response using current model with optional features."""
        model_info = MODELS[self.current_model_name]
        provider = self._get_provider(model_info["provider"])

        # Only pass features if model supports them
        actual_search = use_search and model_info.get("supports_search", False)
        actual_thinking = use_thinking and model_info.get("supports_thinking", False)

        return provider.generate(
            model_info["model_id"],
            prompt,
            use_search=actual_search,
            use_thinking=actual_thinking
        )

    def generate_stream(
        self,
        prompt: str,
        use_search: bool = False,
        use_thinking: bool = False
    ) -> Generator[dict, None, None]:
        """
        Stream response using current model with optional features.

        Yields dicts:
            - {"type": "thinking", "content": "..."} for thinking chunks (Gemini only)
            - {"type": "text", "content": "..."} for text chunks
        """
        model_info = MODELS[self.current_model_name]
        provider = self._get_provider(model_info["provider"])

        # Only pass features if model supports them
        actual_search = use_search and model_info.get("supports_search", False)
        actual_thinking = use_thinking and model_info.get("supports_thinking", False)

        yield from provider.generate_stream(
            model_info["model_id"],
            prompt,
            use_search=actual_search,
            use_thinking=actual_thinking
        )

    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1"
    ) -> dict:
        """
        Generate image using current model (must be image type).

        Returns:
            dict with 'image_base64' and 'mime_type'
        """
        model_info = MODELS[self.current_model_name]

        if model_info.get("type") != "image":
            raise ValueError(f"Model {self.current_model_name} does not support image generation")

        provider = self._get_provider(model_info["provider"])

        if not hasattr(provider, 'generate_image'):
            raise ValueError(f"Provider {model_info['provider']} does not support image generation")

        return provider.generate_image(
            model_info["model_id"],
            prompt,
            aspect_ratio=aspect_ratio
        )


# Singleton instance
_manager: Optional[LLMManager] = None

def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance."""
    global _manager
    if _manager is None:
        _manager = LLMManager()
    return _manager
