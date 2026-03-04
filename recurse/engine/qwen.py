"""Qwen 3.5 client wrapping the OpenAI-compatible API (Ollama / vLLM / DashScope)."""

from __future__ import annotations

from openai import AsyncOpenAI

from recurse.config import ModelConfig
from recurse.engine.prompts import ROOT_SYSTEM_PROMPT, SUB_SYSTEM_PROMPT


class QwenClient:
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
        """Root LLM call. Thinking mode ON (default for Qwen 3.5). Used for orchestration."""
        response = await self.client.chat.completions.create(
            model=self.root_model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            max_tokens=16384,
            temperature=0.6,
            top_p=0.95,
            extra_body={"top_k": 20},
        )
        usage = response.usage
        if usage:
            self._root_tokens_used += usage.total_tokens
        content = response.choices[0].message.content or ""
        # Strip <think>...</think> blocks from final output (keep reasoning internal)
        return _strip_thinking(content)

    async def sub_completion(self, query: str, context: str) -> str:
        """Sub-LLM call. Thinking mode OFF for speed. Used for focused chunk analysis."""
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


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    import re
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
