from __future__ import annotations

import json
from typing import Any

import requests

from config import AppConfig
from errors import ConfigError, ServiceError


class LLMClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.provider = self._resolve_provider()

    def _resolve_provider(self) -> str | None:
        if self.config.openai_api_key:
            return "openai"
        if self.config.anthropic_api_key:
            return "anthropic"
        return None

    @property
    def is_configured(self) -> bool:
        return self.provider is not None

    def chat(self, system_prompt: str, user_prompt: str, *, json_mode: bool = False) -> str:
        if not self.provider:
            raise ConfigError("未配置 OPENAI_API_KEY 或 ANTHROPIC_API_KEY")
        if self.provider == "openai":
            return self._chat_openai(system_prompt, user_prompt, json_mode=json_mode)
        return self._chat_anthropic(system_prompt, user_prompt)

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raw = self.chat(system_prompt, user_prompt, json_mode=True)
        parsed = self._safe_json(raw)
        if not isinstance(parsed, dict):
            raise ServiceError("LLM 返回的 JSON 格式不正确")
        return parsed

    def _chat_openai(self, system_prompt: str, user_prompt: str, *, json_mode: bool) -> str:
        url = f"{self.config.openai_base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.openai_api_key}"}
        payload: dict[str, Any] = {
            "model": self.config.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 800,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = requests.post(
            url, headers=headers, json=payload, timeout=self.config.request_timeout
        )
        if response.status_code >= 400:
            raise ServiceError(f"OpenAI 调用失败：{response.status_code} {response.text}")
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _chat_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.config.anthropic_base_url.rstrip('/')}/v1/messages"
        headers = {
            "x-api-key": self.config.anthropic_api_key or "",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.config.anthropic_model,
            "max_tokens": 800,
            "temperature": 0.2,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        response = requests.post(
            url, headers=headers, json=payload, timeout=self.config.request_timeout
        )
        if response.status_code >= 400:
            raise ServiceError(f"Anthropic 调用失败：{response.status_code} {response.text}")
        data = response.json()
        content = data.get("content", [])
        return "".join(block.get("text", "") for block in content)

    def _safe_json(self, content: str) -> Any:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            block = self._extract_json_block(content)
            try:
                return json.loads(block)
            except json.JSONDecodeError as exc:
                raise ServiceError("LLM 返回的 JSON 解析失败") from exc

    def _extract_json_block(self, content: str) -> str:
        start = content.find("{")
        if start == -1:
            raise ServiceError("LLM 未返回可解析的 JSON")
        depth = 0
        for index in range(start, len(content)):
            char = content[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return content[start : index + 1]
        raise ServiceError("LLM 未返回完整的 JSON 对象")
