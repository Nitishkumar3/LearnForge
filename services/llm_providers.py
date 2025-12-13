"""LLM provider utilities."""

import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

MODELS = {
    "Gemini 2.5 Flash": {
        "provider": "gemini",
        "model_id": "gemini-flash-latest",
        "description": "Powerful Gemini reasoning model",
        "supports_search": True,
        "supports_thinking": True
    },
    "Gemini 2.5 Flash Lite": {
        "provider": "gemini",
        "model_id": "gemini-flash-lite-latest",
        "description": "Fast, lightweight Gemini model",
        "supports_search": True,
        "supports_thinking": True
    },
    "Mistral Medium": {
        "provider": "mistral",
        "model_id": "mistral-medium-latest",
        "description": "Mistral's powerful medium model with web search",
        "supports_search": True,
        "supports_thinking": False
    },
    "Groq Compound": {
        "provider": "groq",
        "model_id": "compound-beta",
        "description": "Groq's compound AI with web search",
        "supports_search": True,
        "supports_thinking": False,
        "search_always_on": True
    },
    "Llama 4 Scout 17B": {
        "provider": "groq",
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "description": "Llama 4 Scout 17B on Groq",
        "supports_search": False,
        "supports_thinking": False
    },
    "Llama 3.3 70B": {
        "provider": "cerebras",
        "model_id": "llama-3.3-70b",
        "description": "Llama 3.3 70B on Cerebras - ultra fast inference",
        "supports_search": False,
        "supports_thinking": False
    },
    "GPT OSS 120B": {
        "provider": "cerebras",
        "model_id": "gpt-oss-120b",
        "description": "GPT OSS 120B on Cerebras - high reasoning capability",
        "supports_search": False,
        "supports_thinking": False
    },
    "Nova Canvas": {
        "provider": "nova",
        "model_id": "bedrock-amazon.nova-canvas-v1-0",
        "description": "Amazon Nova Canvas for image generation",
        "supports_search": False,
        "supports_thinking": False,
        "type": "image",
        "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5"]
    },
}

DEFAULT_MODEL = "Mistral Medium"
providers = {}
current_model = DEFAULT_MODEL


def get_provider(name):
    global providers
    if name not in providers:
        if name == "gemini":
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            providers[name] = {"client": genai.Client(api_key=api_key), "type": "gemini"}
        elif name == "groq":
            from groq import Groq
            providers[name] = {"client": Groq(api_key=os.getenv("GROQ_API_KEY")), "type": "groq"}
        elif name == "mistral":
            from mistralai import Mistral
            providers[name] = {"client": Mistral(api_key=os.getenv("MISTRAL_API_KEY")), "type": "mistral"}
        elif name == "cerebras":
            from cerebras.cloud.sdk import Cerebras
            providers[name] = {"client": Cerebras(api_key=os.getenv("CEREBRAS_API_KEY")), "type": "cerebras"}
        elif name == "nova":
            cookie_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cookie.txt")
            with open(cookie_path, 'r') as f:
                cookies_data = json.load(f)
            cookies = {}
            for c in cookies_data:
                if c['name'] in ['aws-waf-token', 's_sq', 'pr_refresh_token', 'idToken', 's_fid']:
                    cookies[c['name']] = c['value']
            providers[name] = {"cookies": cookies, "type": "nova", "base_url": "https://partyrock.aws"}
    return providers[name]


def set_model(model_name):
    global current_model
    if model_name not in MODELS:
        return False
    current_model = model_name
    return True


def get_current_model():
    info = MODELS[current_model]
    return {
        "name": current_model,
        "provider": info["provider"],
        "model_id": info["model_id"],
        "description": info["description"],
        "supports_search": info.get("supports_search", False),
        "supports_thinking": info.get("supports_thinking", False),
        "search_always_on": info.get("search_always_on", False),
        "type": info.get("type", "text"),
        "aspect_ratios": info.get("aspect_ratios", [])
    }


def list_models():
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
            "is_current": name == current_model
        }
        for name, info in MODELS.items()
    ]


def generate(prompt, use_search=False, use_thinking=False):
    model_info = MODELS[current_model]
    provider = get_provider(model_info["provider"])
    model_id = model_info["model_id"]

    search = use_search and model_info.get("supports_search", False)
    thinking = use_thinking and model_info.get("supports_thinking", False)

    if provider["type"] == "gemini":
        return gemini_generate(provider["client"], model_id, prompt, search, thinking)
    elif provider["type"] == "groq":
        return groq_generate(provider["client"], model_id, prompt)
    elif provider["type"] == "mistral":
        return mistral_generate(provider["client"], model_id, prompt, search)
    elif provider["type"] == "cerebras":
        return cerebras_generate(provider["client"], model_id, prompt)


def generate_stream(prompt, use_search=False, use_thinking=False):
    model_info = MODELS[current_model]
    provider = get_provider(model_info["provider"])
    model_id = model_info["model_id"]

    search = use_search and model_info.get("supports_search", False)
    thinking = use_thinking and model_info.get("supports_thinking", False)

    if provider["type"] == "gemini":
        yield from gemini_stream(provider["client"], model_id, prompt, search, thinking)
    elif provider["type"] == "groq":
        yield from groq_stream(provider["client"], model_id, prompt)
    elif provider["type"] == "mistral":
        yield from mistral_stream(provider["client"], model_id, prompt, search)
    elif provider["type"] == "cerebras":
        yield from cerebras_stream(provider["client"], model_id, prompt)


def generate_image(prompt, aspect_ratio="1:1"):
    model_info = MODELS[current_model]
    if model_info.get("type") != "image":
        raise ValueError(f"Model {current_model} does not support image generation")

    provider = get_provider(model_info["provider"])
    return nova_generate_image(provider, model_info["model_id"], prompt, aspect_ratio)


# Gemini
def gemini_generate(client, model_id, prompt, use_search, use_thinking):
    from google.genai import types
    tools = [types.Tool(google_search=types.GoogleSearch())] if use_search else []
    config = {"thinking_config": types.ThinkingConfig(thinking_budget=8192 if use_thinking else 0)}
    if tools:
        config["tools"] = tools
    response = client.models.generate_content(model=model_id, contents=prompt, config=types.GenerateContentConfig(**config))
    return response.text


def gemini_stream(client, model_id, prompt, use_search, use_thinking):
    from google.genai import types
    tools = [types.Tool(google_search=types.GoogleSearch())] if use_search else []
    config = {}
    if use_thinking:
        config["thinking_config"] = types.ThinkingConfig(thinking_budget=8192, include_thoughts=True)
    if tools:
        config["tools"] = tools
    for chunk in client.models.generate_content_stream(model=model_id, contents=prompt, config=types.GenerateContentConfig(**config)):
        for part in chunk.candidates[0].content.parts:
            if part.thought:
                yield {"type": "thinking", "content": part.text}
            elif part.text:
                yield {"type": "text", "content": part.text}


# Groq
def groq_generate(client, model_id, prompt):
    response = client.chat.completions.create(model=model_id, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content


def groq_stream(client, model_id, prompt):
    stream = client.chat.completions.create(model=model_id, messages=[{"role": "user", "content": prompt}], stream=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield {"type": "text", "content": chunk.choices[0].delta.content}


# Mistral
def mistral_generate(client, model_id, prompt, use_search):
    tools = [{"type": "web_search"}] if use_search else []
    response = client.beta.conversations.start(
        inputs=[{"role": "user", "content": prompt}],
        model=model_id,
        instructions="",
        completion_args={"temperature": 0.7, "max_tokens": 4096, "top_p": 1},
        tools=tools
    )
    if hasattr(response, 'outputs') and response.outputs:
        for output in response.outputs:
            if hasattr(output, 'content'):
                content = output.content
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    parts = []
                    for block in content:
                        if hasattr(block, 'text'):
                            parts.append(block.text)
                        elif isinstance(block, dict) and 'text' in block:
                            parts.append(block['text'])
                        elif isinstance(block, str):
                            parts.append(block)
                    return ''.join(parts)
    return str(response)


def mistral_stream(client, model_id, prompt, use_search):
    if use_search:
        result = mistral_generate(client, model_id, prompt, True)
        yield {"type": "text", "content": result}
    else:
        stream = client.chat.stream(model=model_id, messages=[{"role": "user", "content": prompt}])
        for event in stream:
            if event.data.choices[0].delta.content:
                yield {"type": "text", "content": event.data.choices[0].delta.content}


# Study Material Generation (always uses Cerebras qwen-3-235b)
def generate_study_material(prompt):
    """Generate study material using Cerebras qwen-3-235b (20k tokens)."""
    import time
    print(f"[STUDY-QWEN] Starting non-stream generation...")
    start = time.time()

    provider = get_provider("cerebras")
    response = provider["client"].chat.completions.create(
        model="qwen-3-235b-a22b-instruct-2507",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=20000,
        temperature=0.7,
        top_p=0.8
    )

    elapsed = time.time() - start
    content = response.choices[0].message.content
    print(f"[STUDY-QWEN] Completed in {elapsed:.2f}s, {len(content)} chars")
    return content


def generate_study_material_stream(prompt):
    """Stream study material generation using Cerebras qwen-3-235b (20k tokens)."""
    import time
    print(f"[STUDY-QWEN] Starting stream generation...")
    start = time.time()
    first_chunk_time = None
    chunk_count = 0
    total_chars = 0

    provider = get_provider("cerebras")
    print(f"[STUDY-QWEN] Got provider, calling API...")

    stream = provider["client"].chat.completions.create(
        model="qwen-3-235b-a22b-instruct-2507",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=20000,
        temperature=0.7,
        top_p=0.8,
        stream=True
    )
    print(f"[STUDY-QWEN] Stream created, iterating...")

    for chunk in stream:
        if chunk.choices[0].delta.content:
            if first_chunk_time is None:
                first_chunk_time = time.time() - start
                print(f"[STUDY-QWEN] First chunk after {first_chunk_time:.2f}s")

            chunk_count += 1
            total_chars += len(chunk.choices[0].delta.content)

            if chunk_count % 50 == 0:
                print(f"[STUDY-QWEN] Chunk {chunk_count}, {total_chars} chars so far...")

            yield {"type": "text", "content": chunk.choices[0].delta.content}

    elapsed = time.time() - start
    print(f"[STUDY-QWEN] Stream done: {chunk_count} chunks, {total_chars} chars, {elapsed:.2f}s total")


# Quiz Generation (always uses Cerebras gpt-oss-120b with JSON format)
def generate_quiz(prompt):
    """Generate quiz questions using Cerebras gpt-oss-120b with JSON response format (20k tokens)."""
    import time
    print(f"[QUIZ-GPT-OSS] Starting quiz generation...")
    start = time.time()

    provider = get_provider("cerebras")
    response = provider["client"].chat.completions.create(
        model="gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a quiz generator. Always respond with valid JSON only, no markdown code blocks."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=20000,
        temperature=0.7,
        top_p=1,
        response_format={"type": "json_object"}
    )

    elapsed = time.time() - start
    content = response.choices[0].message.content
    print(f"[QUIZ-GPT-OSS] Completed in {elapsed:.2f}s, {len(content)} chars")
    return content


# Cerebras
def cerebras_generate(client, model_id, prompt):
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=4096,
        temperature=0.2,
        top_p=1
    )
    return response.choices[0].message.content


def cerebras_stream(client, model_id, prompt):
    stream = client.chat.completions.create(
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


# Nova (image generation)
def nova_generate_image(provider, model_id, prompt, aspect_ratio):
    import httpx
    from fake_useragent import UserAgent

    sizes = {
        "16:9": (1280, 720), "9:16": (720, 1280), "2:3": (768, 1152),
        "3:2": (1152, 768), "1:1": (512, 512), "3:4": (768, 1024),
        "4:3": (1024, 768), "4:5": (768, 960), "5:4": (960, 768),
    }
    width, height = sizes.get(aspect_ratio, (512, 512))

    with httpx.Client(cookies=provider["cookies"], timeout=60.0) as client:
        response = client.get(f"{provider['base_url']}/image")
        match = re.search(r'<meta name="anti-csrftoken-a2z" value="([^"]+)"', response.text)
        csrf_token = match.group(1) if match else None

        if not csrf_token:
            raise ValueError("Could not extract CSRF token")

        ua = UserAgent()
        headers = {
            "Content-Type": "application/json",
            "Origin": provider["base_url"],
            "Referer": f"{provider['base_url']}/image",
            "User-Agent": ua.random,
            "anti-csrftoken-a2z": csrf_token,
        }

        payload = {
            "prompt": prompt,
            "negativePrompt": "",
            "options": {"model": model_id, "cfgScale": 6, "height": height, "width": width},
            "context": {"type": "image-playground"}
        }

        response = client.post(f"{provider['base_url']}/api/generateImage", headers=headers, json=payload)

        if response.status_code != 200:
            raise ValueError(f"Image generation failed: HTTP {response.status_code}")

        try:
            data = response.json()
        except Exception:
            raise ValueError("Image generation failed: Invalid response from server. Cookies may have expired - update cookie.txt")

        if "result" not in data or "data" not in data["result"]:
            raise ValueError(f"Unexpected response: {data}")

        image_base64 = data["result"]["data"]["imageBase64"]
        if image_base64.startswith("data:image/"):
            image_base64 = image_base64.split(",")[1]

        return {"image_base64": image_base64, "mime_type": "image/jpeg"}
