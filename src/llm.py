"""Pluggable LLM provider so the same pipeline works for whichever API key
a given client/gig gives you access to. Add a provider by subclassing
LLMProvider and registering it in get_llm_provider().
"""
import os


class LLMProvider:
    def generate(self, system: str, user: str) -> str:
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str = None, model: str = None):
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (check your .env file).")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

    def generate(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str = None, model: str = None):
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set (check your .env file).")
        self.client = OpenAI(api_key=key)
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content


def get_llm_provider() -> LLMProvider:
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":
        return AnthropicProvider()
    if provider == "openai":
        return OpenAIProvider()
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'anthropic' or 'openai'.")
