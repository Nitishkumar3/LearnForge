"""
LLM Provider Management for LearnForge.

Supports multiple LLM providers with search and thinking capabilities.
Add new models by adding to MODELS dict below.
"""

import os
from typing import Optional
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
    "Llama 3.1 8B": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "description": "Fast Llama model on Groq",
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


# Provider registry
PROVIDERS = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "mistral": MistralProvider,
    "cerebras": CerebrasProvider,
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
            "search_always_on": model_info.get("search_always_on", False)
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


# Singleton instance
_manager: Optional[LLMManager] = None

def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance."""
    global _manager
    if _manager is None:
        _manager = LLMManager()
    return _manager
