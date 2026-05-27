from __future__ import annotations

from dataclasses import dataclass
import re
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


PORT_ALIASES = {
    "深圳湾": ["深圳灣", "深圳灣口岸", "Shenzhen Bay", "Shenzhen Bay Control Point"],
    "福田": ["福田口岸", "Futian", "Lok Ma Chau Spur Line", "落馬洲支線"],
    "罗湖": ["罗湖口岸", "羅湖口岸", "Lo Wu"],
    "皇岗": ["皇岗口岸", "皇崗口岸", "Lok Ma Chau", "Lok Ma Chau Control Point"],
    "莲塘": ["莲塘口岸", "蓮塘口岸", "香园围", "Heung Yuen Wai"],
}

MTR_STATION_ALIASES = {
    "西九龙": ("TML", "AUS"),
    "西九龍": ("TML", "AUS"),
    "西九龙站": ("TML", "AUS"),
    "西九龍站": ("TML", "AUS"),
    "西九龙高铁": ("TML", "AUS"),
    "西九龍高鐵": ("TML", "AUS"),
    "柯士甸": ("TML", "AUS"),
    "九龙": ("TCL", "KOW"),
    "香港": ("AEL", "HOK"),
    "金钟": ("ISL", "ADM"),
}


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
        response = requests.post(
            url, headers=headers, json=payload, timeout=self.config.request_timeout
        )
        if response.status_code >= 400:
            raise ServiceError(f"MCP 工具调用失败：{response.status_code} {response.text}")
        data = response.json()
        return data.get("result", data)

    def get_exchange_rate(self, base: str = "HKD", target: str = "CNY") -> ExchangeRate:
        if self.config.mcp_base_url:
            data = self.call_tool(self.config.mcp_exchange_tool, {"base": base, "target": target})
            rate = self._extract_exchange_rate(data, target)
            if rate is None:
                raise ServiceError("汇率 MCP 工具未返回目标币种")
            return ExchangeRate(base=base, target=target, rate=rate)
        return self._get_exchange_rate_from_api(base, target)

    def get_port_traffic(self, port: str) -> PortTraffic:
        if self.config.mcp_base_url:
            data = self.call_tool(self.config.mcp_port_tool, {"port": port})
            queue = data.get("queue_minutes") or data.get("queue") or data.get("wait_minutes")
            if queue is None:
                raise ServiceError("口岸工具未返回排队时长")
            return PortTraffic(port=port, queue_minutes=int(queue))
        return self._get_port_traffic_from_api(port)

    def get_mtr_schedule(self, station: str) -> MTRSchedule:
        if self.config.mcp_base_url:
            data = self.call_tool(self.config.mcp_mtr_tool, {"station": station})
            interval = data.get("interval_minutes") or data.get("interval") or data.get("wait_minutes")
            if interval is None:
                raise ServiceError("港铁工具未返回班次间隔")
            return MTRSchedule(station=station, interval_minutes=int(interval))
        return self._get_mtr_schedule_from_api(station)

    def _get_exchange_rate_from_api(self, base: str, target: str) -> ExchangeRate:
        url = self._format_exchange_rate_url(base)
        response = requests.get(url, timeout=self.config.request_timeout)
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

    def _format_exchange_rate_url(self, base: str) -> str:
        url = self.config.exchange_rate_api_url
        if "{base}" in url:
            return url.format(base=base)
        if "base=" in url:
            return re.sub(r"base=[A-Za-z]{3}", f"base={base}", url)
        if re.search(r"/latest/[A-Za-z]{3}", url):
            return re.sub(r"/latest/[A-Za-z]{3}", f"/latest/{base}", url)
        if base == "HKD":
            return url
        raise ServiceError("汇率接口 URL 未提供 base 占位符")

    def _extract_exchange_rate(self, data: dict[str, Any], target: str) -> float | None:
        if not data:
            return None
        if "rate" in data:
            return float(data["rate"])
        if "exchange_rate" in data:
            return float(data["exchange_rate"])
        rates = data.get("rates") or data.get("result", {}).get("rates", {})
        rate = rates.get(target)
        return float(rate) if rate is not None else None

    def _get_port_traffic_from_api(self, port: str) -> PortTraffic:
        response = requests.get(
            self.config.port_wait_time_api_url, timeout=self.config.request_timeout
        )
        if response.status_code >= 400:
            raise ServiceError(
                f"口岸接口调用失败：{response.status_code} {response.text}"
            )
        data = response.json()
        records = self._extract_port_records(data)
        record = self._find_port_record(port, records)
        if not record:
            raise ServiceError("口岸接口未找到匹配口岸")
        queue = self._extract_wait_minutes(record)
        if queue is None:
            raise ServiceError("口岸接口未返回排队时长")
        return PortTraffic(port=port, queue_minutes=queue)

    def _extract_port_records(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("data", "records", "result", "controlPoints", "control_points"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            for value in payload.values():
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        raise ServiceError("口岸接口返回格式异常")

    def _find_port_record(
        self, port: str, records: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        normalized = self._normalize_port_name(port)
        aliases = self._aliases_for_port(normalized)
        fallback = None
        for record in records:
            names = self._extract_port_names(record)
            if any(self._name_matches(normalized, name, aliases) for name in names):
                direction = str(record.get("direction") or record.get("inOut") or "")
                if any(flag in direction for flag in ["Departure", "出境", "离境"]):
                    return record
                fallback = fallback or record
        return fallback

    def _extract_port_names(self, record: dict[str, Any]) -> list[str]:
        names: list[str] = []
        for key in (
            "controlPoint",
            "control_point",
            "controlPointName",
            "control_point_name",
            "controlPointNameEn",
            "controlPointNameTc",
            "controlPointNameZh",
            "name",
        ):
            value = record.get(key)
            if value:
                names.append(str(value))
        if not names:
            for value in record.values():
                if isinstance(value, str) and ("口岸" in value or "control point" in value.lower()):
                    names.append(value)
        return names

    def _normalize_port_name(self, name: str) -> str:
        normalized = re.sub(r"[\s\-_()（）]", "", name).lower()
        return (
            normalized.replace("口岸", "")
            .replace("關口", "")
            .replace("controlpoint", "")
        )

    def _aliases_for_port(self, normalized: str) -> list[str]:
        aliases: list[str] = []
        for key, values in PORT_ALIASES.items():
            if self._normalize_port_name(key) == normalized:
                return [*values, key]
            for alias in values:
                if self._normalize_port_name(alias) == normalized:
                    return [*values, key]
        return aliases

    def _name_matches(self, target: str, candidate: str, aliases: list[str]) -> bool:
        candidate_norm = self._normalize_port_name(candidate)
        if not candidate_norm:
            return False
        if target in candidate_norm or candidate_norm in target:
            return True
        for alias in aliases:
            alias_norm = self._normalize_port_name(alias)
            if alias_norm and (alias_norm in candidate_norm or candidate_norm in alias_norm):
                return True
        return False

    def _extract_wait_minutes(self, record: dict[str, Any]) -> int | None:
        for key in (
            "waitingTime",
            "waitTime",
            "waiting_time",
            "queue",
            "queue_minutes",
            "wait_minutes",
            "estimatedWaitingTime",
            "estimated_waiting_time",
        ):
            if key in record:
                value = record.get(key)
                minutes = self._parse_waiting_time(value)
                if minutes is not None:
                    return minutes
        return None

    def _parse_waiting_time(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        if not text:
            return None
        lowered = text.lower()
        if "no waiting" in lowered or "不需等候" in text or "无须等候" in text:
            return 0
        matches = re.findall(
            r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|小時|小时|minutes?|mins?|分鐘|分)",
            lowered,
        )
        minutes = []
        for number, unit in matches:
            value_num = float(number)
            if unit.startswith("hour") or unit.startswith("hr") or "小時" in unit or "小时" in unit:
                value_num *= 60
            minutes.append(value_num)
        if minutes:
            if len(minutes) > 1 and re.search(r"[-~至]|to", lowered):
                # 多个区间值时取平均值作为估算
                return int(round(sum(minutes) / len(minutes)))
            return int(round(sum(minutes)))
        digits = re.findall(r"\d+", lowered)
        if digits:
            return int(digits[0])
        return None

    def _get_mtr_schedule_from_api(self, station: str) -> MTRSchedule:
        line, station_code = self._resolve_mtr_codes(station)
        url, params = self._format_mtr_schedule_request(line, station_code)
        response = requests.get(url, params=params, timeout=self.config.request_timeout)
        if response.status_code >= 400:
            raise ServiceError(
                f"港铁接口调用失败：{response.status_code} {response.text}"
            )
        data = response.json()
        if data.get("status") not in (None, 1, "1", "SUCCESS", "success"):
            raise ServiceError(f"港铁接口返回异常：{data.get('message', 'unknown')}")
        station_data = self._extract_mtr_station_data(data, line, station_code)
        interval = self._extract_mtr_interval_minutes(station_data)
        if interval is None:
            raise ServiceError("港铁接口未返回班次间隔")
        return MTRSchedule(station=station, interval_minutes=interval)

    def _resolve_mtr_codes(self, station: str) -> tuple[str, str]:
        normalized = self._normalize_station_name(station)
        for key, value in MTR_STATION_ALIASES.items():
            if normalized == self._normalize_station_name(key):
                return value
        return self.config.mtr_default_line, self.config.mtr_default_station

    def _normalize_station_name(self, station: str) -> str:
        normalized = re.sub(r"[\s\-_()（）]", "", station).lower()
        return (
            normalized.replace("站", "")
            .replace("車站", "")
            .replace("高铁", "")
            .replace("高鐵", "")
        )

    def _format_mtr_schedule_request(self, line: str, station_code: str) -> tuple[str, dict[str, str] | None]:
        url = self.config.mtr_schedule_api_url
        if "{line}" in url or "{sta}" in url or "{station}" in url:
            return (
                url.format(line=line.lower(), sta=station_code.upper(), station=station_code.upper()),
                None,
            )
        return url, {"line": line.lower(), "sta": station_code.upper()}

    def _extract_mtr_station_data(
        self, payload: dict[str, Any], line: str, station_code: str
    ) -> dict[str, Any]:
        data = payload.get("data") or payload.get("result") or payload
        line_data = None
        for key in ("lines", "line", "data"):
            if isinstance(data, dict) and key in data:
                data = data[key]
                break
        if isinstance(data, dict):
            for candidate in {line.upper(), line.lower(), line.capitalize()}:
                if candidate in data:
                    line_data = data[candidate]
                    break
        if line_data is None:
            line_data = data
        station_data = None
        if isinstance(line_data, dict) and "stations" in line_data:
            line_data = line_data["stations"]
        if isinstance(line_data, dict):
            for candidate in {station_code.upper(), station_code.lower()}:
                if candidate in line_data:
                    station_data = line_data[candidate]
                    break
        if station_data is None or not isinstance(station_data, dict):
            raise ServiceError("港铁接口未找到站点数据")
        return station_data

    def _extract_mtr_interval_minutes(self, station_data: dict[str, Any]) -> int | None:
        for direction in ("UP", "DOWN", "up", "down"):
            trains = station_data.get(direction)
            interval = self._interval_from_trains(trains)
            if interval is not None:
                return interval
        for key in ("interval_minutes", "interval", "wait_minutes"):
            value = station_data.get(key)
            if value is not None:
                return int(value)
        return None

    def _interval_from_trains(self, trains: Any) -> int | None:
        if not isinstance(trains, list) or not trains:
            return None
        for train in trains:
            if not isinstance(train, dict):
                continue
            ttnt = train.get("ttnt") or train.get("time_to_next")
            if ttnt is not None:
                try:
                    return int(ttnt)
                except (TypeError, ValueError):
                    pass
        times = []
        for train in trains:
            if not isinstance(train, dict):
                continue
            time_value = train.get("time") or train.get("dep_time") or train.get("departure_time")
            minutes = self._parse_time_to_minutes(time_value)
            if minutes is not None:
                times.append(minutes)
        if len(times) >= 2:
            diff = times[1] - times[0]
            if diff < 0:
                # 跨越午夜时，将负差值平移到次日区间
                diff += 24 * 60
            return diff if diff else None
        return None

    def _parse_time_to_minutes(self, value: Any) -> int | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        match = re.match(r"^(\d{1,2}):(\d{2})$", text)
        if match:
            return int(match.group(1)) * 60 + int(match.group(2))
        if re.match(r"^\d{3,4}$", text):
            hours = int(text[:-2])
            minutes = int(text[-2:])
            return hours * 60 + minutes
        return None
