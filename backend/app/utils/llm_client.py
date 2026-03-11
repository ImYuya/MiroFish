"""
LLM client wrapper
Unified calls using OpenAI format
"""

import json
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config


class LLMClient:
    """LLM client"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 16384,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Send a chat request

        Args:
            messages: List of messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            response_format: Response format (e.g. JSON mode)

        Returns:
            Model response text
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        content = msg.content or ""

        # Some thinking models (e.g. qwen3.5) put the answer in a "reasoning"
        # field and leave content empty.  Fall back to reasoning if content is
        # blank after cleaning.
        if not content.strip():
            reasoning = getattr(msg, 'reasoning', None) or ""
            if reasoning:
                content = reasoning

        # Remove <think> blocks from reasoning models (MiniMax, GLM, etc.)
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 16384
    ) -> Dict[str, Any]:
        """
        Send a chat request and return JSON

        Args:
            messages: List of messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens

        Returns:
            Parsed JSON object
        """
        # Try with response_format first; fall back to plain chat if it
        # returns empty (some Ollama models ignore response_format).
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

        if not response.strip():
            # Retry without response_format constraint
            response = self.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # Clean markdown code blocks
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        # Extract first JSON object from mixed text (e.g. thinking + JSON)
        if cleaned_response:
            # Try direct parse first
            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError:
                pass

            # Try to find JSON object in the text
            match = re.search(r'\{[\s\S]*\}', cleaned_response)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"Invalid JSON returned by LLM: {cleaned_response[:500]}")
