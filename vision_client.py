from __future__ import annotations

import base64
import json
from typing import Any

import requests

from config import AppConfig
from errors import ConfigError, ServiceError


class VisionClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def parse_events(self, file_data: bytes) -> list[dict[str, str]]:
        if self.config.vision_provider == "anthropic":
            return self._parse_with_anthropic(file_data)
        return self._parse_with_openai(file_data)

    def _parse_with_openai(self, file_data: bytes) -> list[dict[str, str]]:
        if not self.config.openai_api_key:
            raise ConfigError("未配置 OPENAI_API_KEY")
        image_data, media_type = self._encode_image(file_data)
        url = f"{self.config.openai_base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.openai_api_key}"}
        prompt = (
            "你是活动信息识别助手。请从图片中提取活动信息，并输出 JSON："
            '{"events":[{"date":"YYYY-MM-DD","time":"HH:MM","title":"活动名称"}]}。'
            "无法识别的字段请填写“待确认日期/时间”。"
        )
        payload = {
            "model": self.config.openai_model,
            "messages": [
                {"role": "system", "content": "输出必须是 JSON。"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{image_data}"},
                        },
                    ],
                },
            ],
            "temperature": 0.1,
            "max_tokens": 600,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(
            url, headers=headers, json=payload, timeout=self.config.request_timeout
        )
        if response.status_code >= 400:
            raise ServiceError(f"视觉模型调用失败：{response.status_code} {response.text}")
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return self._extract_events(content)

    def _parse_with_anthropic(self, file_data: bytes) -> list[dict[str, str]]:
        if not self.config.anthropic_api_key:
            raise ConfigError("未配置 ANTHROPIC_API_KEY")
        image_data, media_type = self._encode_image(file_data)
        url = f"{self.config.anthropic_base_url.rstrip('/')}/v1/messages"
        headers = {
            "x-api-key": self.config.anthropic_api_key,
            "anthropic-version": "2023-06-01",
        }
        prompt = (
            "请从图片中提取活动信息并输出 JSON："
            '{"events":[{"date":"YYYY-MM-DD","time":"HH:MM","title":"活动名称"}]}。'
            "无法识别的字段请填写“待确认日期/时间”。"
        )
        payload = {
            "model": self.config.anthropic_model,
            "max_tokens": 600,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                    ],
                }
            ],
        }
        response = requests.post(
            url, headers=headers, json=payload, timeout=self.config.request_timeout
        )
        if response.status_code >= 400:
            raise ServiceError(f"视觉模型调用失败：{response.status_code} {response.text}")
        data = response.json()
        content = "".join(block.get("text", "") for block in data.get("content", []))
        return self._extract_events(content)

    def _encode_image(self, file_data: bytes) -> tuple[str, str]:
        if not file_data:
            raise ServiceError("未找到上传的图片内容")
        media_type = self._detect_media_type(file_data)
        encoded = base64.b64encode(file_data).decode("utf-8")
        return encoded, media_type

    def _detect_media_type(self, file_data: bytes) -> str:
        if file_data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if file_data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if file_data[:6] in {b"GIF87a", b"GIF89a"}:
            return "image/gif"
        if file_data.startswith(b"RIFF") and file_data[8:12] == b"WEBP":
            return "image/webp"
        return "image/png"

    def _extract_events(self, content: str) -> list[dict[str, str]]:
        parsed = self._safe_json(content)
        events = parsed.get("events", [])
        if not isinstance(events, list):
            raise ServiceError("视觉模型未返回 events 列表")
        return [
            {
                "date": str(event.get("date", "待确认日期")),
                "time": str(event.get("time", "待确认时间")),
                "title": str(event.get("title", "未命名活动")),
            }
            for event in events
        ]

    def _safe_json(self, content: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            block = self._extract_json_block(content)
            try:
                return json.loads(block)
            except json.JSONDecodeError as exc:
                raise ServiceError("视觉模型 JSON 解析失败") from exc

    def _extract_json_block(self, content: str) -> str:
        start = content.find("{")
        if start == -1:
            raise ServiceError("视觉模型未返回可解析的 JSON")
        depth = 0
        for index in range(start, len(content)):
            char = content[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return content[start : index + 1]
        raise ServiceError("视觉模型未返回完整的 JSON 对象")
