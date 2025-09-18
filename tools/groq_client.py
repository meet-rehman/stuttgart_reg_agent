# src/stuttgart_reg_agent/tools/groq_client.py

import os
import requests
import httpx
import asyncio
from typing import Any, Dict, Optional

class GroqClient:
    """
    Enhanced Groq client with async support for better FastAPI performance.
    """

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.api_url = api_url or os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
        self.model = model or "llama-3.1-8b-instant"  # Faster model

        if not self.api_key or not self.api_url:
            raise RuntimeError("GROQ_API_KEY and GROQ_API_URL must be set in environment or passed to GroqClient")

        # Sync session for backward compatibility
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        # Async client for new endpoints
        self.async_client = None

    async def get_async_client(self):
        """Get or create async HTTP client"""
        if self.async_client is None:
            self.async_client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self.async_client

    def build_payload(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Build payload compatible with Groq OpenAI-compatible chat API.
        """
        return {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Synchronous completion - for backward compatibility
        """
        payload = self.build_payload(prompt, max_tokens=max_tokens, temperature=temperature)
        resp = self.session.post(self.api_url, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()

    async def complete_async(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Asynchronous completion - for better FastAPI performance
        """
        client = await self.get_async_client()
        payload = self.build_payload(prompt, max_tokens=max_tokens, temperature=temperature)
        
        response = await client.post(self.api_url, json=payload)
        response.raise_for_status()
        return response.json()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.async_client:
            await self.async_client.aclose()