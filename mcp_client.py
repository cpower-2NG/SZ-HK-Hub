from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date
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
    today_total: int | None = None
    note: str = ""


@dataclass(frozen=True)
class MTRSchedule:
    station: str
    interval_minutes: int
    next_trains: list[dict[str, str]] | None = None


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
        """主汇率接口：open.er-api.com。失败时尝试 nxvav.cn 备用。"""
        last_error = None
        for url in [self.config.exchange_rate_api_url, self.config.exchange_rate_api_url_backup]:
            try:
                if "nxvav" in url:
                    # nxvav.cn 接口格式不同：?currency=HKD
                    full_url = f"{url}?currency={base}"
                    response = requests.get(full_url, timeout=self.config.request_timeout)
                    if response.status_code >= 400:
                        continue
                    data = response.json()
                    rates_data = data.get("data", {}).get("rates", [])
                    rate = None
                    for r in rates_data:
                        if r.get("currency") == target:
                            rate = r.get("rate")
                            break
                    if rate is None:
                        continue
                else:
                    response = requests.get(url, timeout=self.config.request_timeout)
                    if response.status_code >= 400:
                        continue
                    data = response.json()
                    rates = data.get("rates") or data.get("result", {}).get("rates", {})
                    rate = rates.get(target)
                    if rate is None:
                        continue
                return ExchangeRate(base=base, target=target, rate=float(rate))
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, Exception) as exc:
                last_error = exc
                continue
        raise ServiceError(f"所有汇率接口均不可用（最后错误：{last_error}")

    def get_port_traffic(self, port: str) -> PortTraffic:
        """口岸实时客流：优先调用 MCP 服务，否则从入境处 CSV 估算。"""
        # 先尝试 MCP 服务
        if self.config.mcp_base_url:
            try:
                data = self.call_tool(self.config.mcp_port_tool, {"port": port})
                queue = data.get("queue_minutes") or data.get("queue") or data.get("wait_minutes")
                if queue is not None:
                    return PortTraffic(port=port, queue_minutes=int(queue))
            except (ConfigError, ServiceError):
                pass

        # 回退：从入境处 CSV 估算
        return self._estimate_port_traffic(port)

    def _estimate_port_traffic(self, port: str) -> PortTraffic:
        """从香港入境处每日客流 CSV 估算口岸繁忙度。"""
        port_map = {
            "深圳湾": "Shenzhen Bay",
            "罗湖": "Lo Wu",
            "落马洲": "Lok Ma Chau Spur Line",
            "福田": "Lok Ma Chau Spur Line",
            "皇岗": "Lok Ma Chau",
            "西九龙": "Express Rail Link West Kowloon",
            "港珠澳": "Hong Kong-Zhuhai-Macao Bridge",
            "机场": "Airport",
        }
        csv_port = port_map.get(port, port)
        today_str = date.today().strftime("%d-%m-%Y")

        try:
            resp = requests.get(self.config.immigration_csv_url, timeout=15)
            resp.raise_for_status()
            reader = csv.DictReader(io.StringIO(resp.text))
            total = 0
            for row in reader:
                if row.get("Control Point") == csv_port and row.get("Date") == today_str:
                    total += int(row.get("Total", 0) or 0)

            if total > 150000:
                queue = 35
                note = "🔴 极繁忙"
            elif total > 80000:
                queue = 20
                note = "🟡 较繁忙"
            elif total > 30000:
                queue = 10
                note = "🟢 通畅"
            elif total > 0:
                queue = 5
                note = "🟢 畅通"
            else:
                queue = 15
                note = "暂无今日数据，显示为预估值"

            return PortTraffic(port=port, queue_minutes=queue, today_total=total, note=note)
        except Exception:
            return PortTraffic(
                port=port, queue_minutes=15, today_total=None, note="客流数据暂不可用，显示为预估值"
            )

    def get_mtr_schedule(self, station: str) -> MTRSchedule:
        """港铁实时到站信息：调用 data.gov.hk 开放 API。"""
        return self._fetch_mtr_realtime(station)

    def _fetch_mtr_realtime(self, station: str) -> MTRSchedule:
        """从 data.gov.hk 获取港铁实时列车信息。"""
        station_map = {
            "罗湖": ("EAL", "LOW"),
            "Lo Wu": ("EAL", "LOW"),
            "落马洲": ("EAL", "LMC"),
            "Lok Ma Chau": ("EAL", "LMC"),
            "西九龙": ("EAL", "HUH"),  # 最近为红磡
            "红磡": ("EAL", "HUH"),
            "Hung Hom": ("EAL", "HUH"),
            "金钟": ("EAL", "ADM"),
            "Admiralty": ("EAL", "ADM"),
            "旺角东": ("EAL", "MKK"),
            "Mong Kok East": ("EAL", "MKK"),
            "九龙塘": ("EAL", "KOT"),
            "Kowloon Tong": ("EAL", "KOT"),
            "尖沙咀": ("TWL", "TST"),
            "Tsim Sha Tsui": ("TWL", "TST"),
            "中环": ("TWL", "CEN"),
            "Central": ("TWL", "CEN"),
        }

        line_code, sta_code = station_map.get(station, ("EAL", "HUH"))

        try:
            url = f"{self.config.mtr_realtime_api_url}?line={line_code}&sta={sta_code}"
            resp = requests.get(url, timeout=15)
            if resp.status_code >= 400:
                raise ServiceError(f"MTR API 返回 {resp.status_code}")

            data = resp.json()
            sta_data = data.get("data", {}).get(f"{line_code}-{sta_code}", {})
            if not sta_data:
                return MTRSchedule(station=station, interval_minutes=8)

            # 解析上下行列车
            trains: list[dict[str, str]] = []
            for direction, label in [("UP", "上行"), ("DOWN", "下行")]:
                for t in sta_data.get(direction, [])[:4]:
                    if t.get("valid") != "Y":
                        continue
                    trains.append({
                        "direction": label,
                        "dest": t.get("dest", "?"),
                        "platform": t.get("plat", "?"),
                        "arrive_in": f"{t.get('ttnt', '?')}分钟",
                        "time": t.get("time", ""),
                    })

            # 计算班次间隔
            ttnt_values = []
            for direction in ["UP", "DOWN"]:
                valid_trains = [
                    t for t in sta_data.get(direction, [])[:4] if t.get("valid") == "Y"
                ]
                for i in range(len(valid_trains) - 1):
                    diff = abs(
                        int(valid_trains[i + 1].get("ttnt", "0") or "0")
                        - int(valid_trains[i].get("ttnt", "0") or "0")
                    )
                    if 1 <= diff <= 20:
                        ttnt_values.append(diff)

            interval = round(sum(ttnt_values) / len(ttnt_values)) if ttnt_values else 6

            return MTRSchedule(
                station=station, interval_minutes=interval, next_trains=trains[:4]
            )
        except Exception:
            return MTRSchedule(station=station, interval_minutes=8)

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
