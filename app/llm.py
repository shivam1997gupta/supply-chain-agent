"""
The one place that knows which LLM provider we use.

Every agent calls get_llm() instead of importing a vendor SDK directly, so
switching providers is an environment-variable change, not a code change.

Config comes from .env (see .env.example):
    LLM_PROVIDER=google_genai  LLM_MODEL=gemini-3.5-flash   GOOGLE_API_KEY=...
    LLM_PROVIDER=openai        LLM_MODEL=gpt-4o             OPENAI_API_KEY=...
    LLM_PROVIDER=anthropic     LLM_MODEL=claude-sonnet-4-6  ANTHROPIC_API_KEY=...

NOTE on Gemini "thinking" (the cause of the big latency we debugged):
  - Gemini 3.x models (gemini-3*) reason with a `thinking_level` that DEFAULTS
    TO 'high'. That hidden reasoning pass made every call take 20-30s. We set
    it to 'low' for our simple SQL/JSON tasks to get fast responses.
    We also do NOT force temperature here for Gemini 3.x: Google recommends
    temperature 1.0, and values below 1.0 can cause looping/degraded output.
  - Gemini 2.5.x models use an integer `thinking_budget`; 0 disables thinking.
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "google_genai")
    model = os.getenv("LLM_MODEL", "gemini-3.5-flash")

    if provider == "google_genai" and model.startswith("gemini-3"):
        # Gemini 3.x: control reasoning with thinking_level; 'low' = fast.
        # Leave temperature unset (Google wants 1.0 on these models).
        kwargs = {"thinking_level": "low"}
    elif provider == "google_genai":
        # Gemini 2.5.x: integer thinking budget; 0 disables the thinking pass.
        kwargs = {"thinking_budget": 0, "temperature": 0}
    else:
        # OpenAI / Anthropic / etc: deterministic output for SQL & JSON.
        kwargs = {"temperature": 0}

    return init_chat_model(model, model_provider=provider, **kwargs)


def to_text(content) -> str:
    """Normalize an LLM response's `.content` to a plain string.

    Providers differ: OpenAI returns a string, but Gemini (and Anthropic for
    multi-part replies) return a LIST of content blocks, e.g.
        [{"type": "text", "text": "..."}]
    Both agents need this, so it lives here in the one shared LLM module.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)