"""LLM client wrapping OpenAI-compatible APIs (Groq, OpenAI, Together, etc.)."""

from __future__ import annotations

from openai import AsyncOpenAI

from recurse.config import ModelConfig
from recurse.engine.prompts import ROOT_SYSTEM_PROMPT, SUB_SYSTEM_PROMPT


class LLMClient:
    def __init__(self, model_config: ModelConfig) -> None:
        self.root_model = model_config.root
        self.sub_model = model_config.sub
        self.client = AsyncOpenAI(
            base_url=model_config.base_url,
            api_key=model_config.api_key,
        )
        self._root_tokens_used: int = 0
        self._sub_tokens_used: int = 0

    @property
    def tokens_used(self) -> int:
        return self._root_tokens_used + self._sub_tokens_used

    async def root_completion(self, system_prompt: str, messages: list[dict]) -> str:
        """Root LLM call — orchestrates the RLM loop."""
        response = await self.client.chat.completions.create(
            model=self.root_model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
        )
        usage = response.usage
        if usage:
            self._root_tokens_used += usage.total_tokens
        return response.choices[0].message.content or ""

    async def sub_completion(self, query: str, context: str) -> str:
        """Sub-LLM call — fast focused chunk analysis."""
        messages = [
            {"role": "system", "content": SUB_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        response = await self.client.chat.completions.create(
            model=self.sub_model,
            messages=messages,
            max_tokens=4096,
            temperature=0.3,
        )
        usage = response.usage
        if usage:
            self._sub_tokens_used += usage.total_tokens
        return response.choices[0].message.content or ""


# Keep old name as alias so any external code doesn't break
QwenClient = LLMClient
