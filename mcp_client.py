from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from config import AppConfig
from errors import ConfigError, ServiceError


@dataclass(frozen=True)
class ExchangeRate:
    base: str
    target: str
    rate: float


@dataclass(frozen=True)
class PortTraffic:
    port: str
    queue_minutes: int


@dataclass(frozen=True)
class MTRSchedule:
    station: str
    interval_minutes: int


@dataclass(frozen=True)
class RouteOption:
    mode: str
    duration_min: int
    cost_hkd: float
    note: str


@dataclass(frozen=True)
class RoutePlan:
    origin: str
    destination: str
    routes: list[RouteOption]
    source: str


@dataclass(frozen=True)
class FileOpResult:
    action: str
    filename: str
    status: str
    data: Any | None = None
    files: list[dict] | None = None


class MCPClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def call_tool(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.config.mcp_base_url:
            raise ConfigError("未配置 MCP_BASE_URL")
        url = f"{self.config.mcp_base_url.rstrip('/')}/tools/{tool_name}"
        headers = {"Content-Type": "application/json"}
        if self.config.mcp_api_key:
            headers["Authorization"] = f"Bearer {self.config.mcp_api_key}"
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.config.request_timeout
            )
        except requests.exceptions.ConnectionError:
            raise ServiceError(f"MCP 服务不可达：{url}")
        except requests.exceptions.Timeout:
            raise ServiceError(f"MCP 服务超时：{url}")
        if response.status_code >= 400:
            raise ServiceError(f"MCP 工具调用失败：{response.status_code} {response.text}")
        data = response.json()
        return data.get("result", data)

    def get_exchange_rate(self, base: str = "HKD", target: str = "CNY") -> ExchangeRate:
        response = requests.get(
            self.config.exchange_rate_api_url, timeout=self.config.request_timeout
        )
        if response.status_code >= 400:
            raise ServiceError(
                f"汇率接口调用失败：{response.status_code} {response.text}"
            )
        data = response.json()
        rates = data.get("rates") or data.get("result", {}).get("rates", {})
        rate = rates.get(target)
        if rate is None:
            raise ServiceError("汇率接口未返回目标币种")
        return ExchangeRate(base=base, target=target, rate=float(rate))

    def get_port_traffic(self, port: str) -> PortTraffic:
        data = self.call_tool(self.config.mcp_port_tool, {"port": port})
        queue = data.get("queue_minutes") or data.get("queue") or data.get("wait_minutes")
        if queue is None:
            raise ServiceError("口岸工具未返回排队时长")
        return PortTraffic(port=port, queue_minutes=int(queue))

    def get_mtr_schedule(self, station: str) -> MTRSchedule:
        data = self.call_tool(self.config.mcp_mtr_tool, {"station": station})
        interval = data.get("interval_minutes") or data.get("interval") or data.get("wait_minutes")
        if interval is None:
            raise ServiceError("港铁工具未返回班次间隔")
        return MTRSchedule(station=station, interval_minutes=int(interval))

    def get_route(self, origin: str, destination: str) -> RoutePlan:
        """跨境路线规划，支持 Google Maps（需配置 API Key）或内置模拟数据。"""
        data = self.call_tool(
            self.config.mcp_route_tool,
            {"origin": origin, "destination": destination},
        )
        raw_routes = data.get("routes", [])
        routes = [
            RouteOption(
                mode=r.get("mode", "未知"),
                duration_min=int(r.get("duration_min", 0)),
                cost_hkd=float(r.get("cost_hkd", 0)),
                note=r.get("note", ""),
            )
            for r in raw_routes
        ]
        return RoutePlan(
            origin=data.get("origin", origin),
            destination=data.get("destination", destination),
            routes=routes,
            source=data.get("source", "mock"),
        )

    def file_save(self, filename: str, data: dict) -> FileOpResult:
        """保存用户数据到服务端 user_data/ 目录。"""
        result = self.call_tool(
            self.config.mcp_file_tool,
            {"action": "save", "filename": filename, "data": data},
        )
        return FileOpResult(
            action=result.get("action", "save"),
            filename=result.get("filename", filename),
            status=result.get("status", "error"),
        )

    def file_load(self, filename: str) -> FileOpResult:
        """从服务端 user_data/ 目录读取用户数据。"""
        result = self.call_tool(
            self.config.mcp_file_tool,
            {"action": "load", "filename": filename},
        )
        return FileOpResult(
            action=result.get("action", "load"),
            filename=result.get("filename", filename),
            status=result.get("status", "error"),
            data=result.get("data"),
        )

    def file_list(self) -> FileOpResult:
        """列出服务端 user_data/ 目录下所有 JSON 文件。"""
        result = self.call_tool(
            self.config.mcp_file_tool,
            {"action": "list"},
        )
        return FileOpResult(
            action=result.get("action", "list"),
            filename="",
            status="ok",
            files=result.get("files", []),
        )
