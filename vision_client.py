from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

import requests

from config import AppConfig
from errors import ConfigError, ServiceError


class VisionClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def parse_events(self, file_path: str) -> list[dict[str, str]]:
        if self.config.vision_provider == "anthropic":
            return self._parse_with_anthropic(file_path)
        return self._parse_with_openai(file_path)

    def _parse_with_openai(self, file_path: str) -> list[dict[str, str]]:
        if not self.config.openai_api_key:
            raise ConfigError("未配置 OPENAI_API_KEY")
        image_data, media_type = self._read_image(file_path)
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

    def _parse_with_anthropic(self, file_path: str) -> list[dict[str, str]]:
        if not self.config.anthropic_api_key:
            raise ConfigError("未配置 ANTHROPIC_API_KEY")
        image_data, media_type = self._read_image(file_path)
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

    def _read_image(self, file_path: str) -> tuple[str, str]:
        path = Path(file_path)
        if not path.exists():
            raise ServiceError("未找到上传的图片文件")
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return encoded, media_type

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
            match = re.search(r"\{.*\}", content, re.S)
            if not match:
                raise ServiceError("视觉模型未返回可解析的 JSON")
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise ServiceError("视觉模型 JSON 解析失败") from exc
