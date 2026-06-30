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


@dataclass(frozen=True)
class GeocodedPlace:
    """Google Geocoding 验证结果。"""
    query: str          # 原始查询
    found: bool          # 是否找到匹配
    formatted_address: str | None = None  # 标准化地址
    lat: float | None = None
    lng: float | None = None
    place_type: str | None = None  # restaurant, transit_station, university, etc.


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
        """跨境路线规划：优先 Google Maps，其次 MCP 服务，最后预设路线。"""
        # 1) 尝试高德地图
        if self.config.amap_api_key:
            try:
                routes = self._call_amap_directions(origin, destination)
                if routes:
                    return RoutePlan(origin=origin, destination=destination, routes=routes, source="amap")
            except Exception:
                pass

        # 2) 尝试 MCP 服务
        if self.config.mcp_base_url:
            try:
                data = self.call_tool(self.config.mcp_route_tool, {"origin": origin, "destination": destination})
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
                if routes:
                    return RoutePlan(
                        origin=data.get("origin", origin),
                        destination=data.get("destination", destination),
                        routes=routes,
                        source=data.get("source", "mcp"),
                    )
            except (ConfigError, ServiceError):
                pass

        # 3) 回退：预设路线
        return self._preset_route(origin, destination)

    def _call_amap_directions(self, origin: str, destination: str) -> list[RouteOption]:
        """调用高德地图公交路径规划 API。"""
        # 先 geocode 获取起终点坐标
        origin_geo = self.geocode_place(origin)
        dest_geo = self.geocode_place(destination)

        if not origin_geo.found or not dest_geo.found:
            return []

        url = "https://restapi.amap.com/v3/direction/transit/integrated"
        params = {
            "key": self.config.amap_api_key,
            "origin": f"{origin_geo.lng},{origin_geo.lat}",
            "destination": f"{dest_geo.lng},{dest_geo.lat}",
            "city": "香港",
            "cityd": "香港",
            "extensions": "all",
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1" or not data.get("route"):
            return []

        routes: list[RouteOption] = []
        transit = data["route"].get("transits", [])
        for t in transit[:2]:
            duration_min = round(int(t.get("duration", "0")) / 60)
            cost = float(t.get("cost", "0"))
            distance_m = int(t.get("distance", "0"))

            # 提取交通方式
            segments = t.get("segments", [])
            modes: list[str] = []
            for seg in segments[:5]:
                bus_info = seg.get("bus", {})
                bus_name = bus_info.get("buslines", [{}])[0].get("name", "") if bus_info.get("buslines") else ""
                if bus_name:
                    modes.append(bus_name)
                elif seg.get("walking"):
                    modes.append(f"步行{seg['walking'].get('distance','0')}m")

            mode_str = " → ".join(modes) if modes else "公交/地铁"
            cost_hkd = cost if cost > 0 else round(distance_m / 1000 * 2, 0)

            routes.append(RouteOption(
                mode=mode_str,
                duration_min=duration_min,
                cost_hkd=cost_hkd,
                note=f"高德地图实时路线，约 {distance_m/1000:.1f} km",
            ))

        return routes

    @staticmethod
    def _preset_route(origin: str, destination: str) -> RoutePlan:
        """内置深港预设路线库。"""
        key = f"{origin}→{destination}"
        presets: dict[str, list[RouteOption]] = {
            "福田→西九龙": [
                RouteOption("港铁东铁线→屯马线", 50, 40, "落马洲站出发"),
                RouteOption("高铁", 14, 80, "福田站→西九龙站，最快"),
            ],
            "深圳湾→西九龙": [
                RouteOption("巴士B2P→港铁屯马线", 55, 25, "天水围站转车"),
            ],
            "罗湖→尖沙咀": [
                RouteOption("港铁东铁线直达", 42, 40, "罗湖站→尖东站"),
            ],
            "福田→中环": [
                RouteOption("东铁线→金钟转港岛线", 65, 50, "落马洲站出发"),
                RouteOption("高铁→港铁", 35, 90, "西九龙站换乘"),
            ],
        }
        if key in presets:
            return RoutePlan(origin=origin, destination=destination, routes=presets[key], source="preset")
        return RoutePlan(
            origin=origin, destination=destination,
            routes=[RouteOption("港铁（推荐）", 55, 40, "请指定具体口岸以获取精确路线")],
            source="preset",
        )

    def geocode_place(self, query: str) -> GeocodedPlace:
        """通过高德地图地理编码 API 验证地点名称。"""
        if not self.config.amap_api_key:
            return GeocodedPlace(query=query, found=False)

        def _try_geocode(q: str, city: str | None = "香港") -> dict | None:
            params: dict = {"key": self.config.amap_api_key, "address": q}
            if city:
                params["city"] = city
            resp = requests.get("https://restapi.amap.com/v3/geocode/geo", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "1" or not data.get("geocodes"):
                return None
            return data

        def _is_hk_or_border(addr: str, prov: str) -> bool:
            # 严格匹配：地址或省份必须含"香港"或"深圳"
            return ("香港" in addr or "深圳" in addr or
                    "香港" in prov or "广东省深圳市" in addr)

        try:
            # 先尝试 city=香港
            data = _try_geocode(query, "香港")
            if data:
                for geo in data["geocodes"]:
                    addr = geo.get("formatted_address", "")
                    prov = geo.get("province", "")
                    if _is_hk_or_border(addr, prov):
                        return self._build_geo_result(query, geo, addr)

            # 再尝试不加 city 限制
            data = _try_geocode(query, None)
            if data:
                for geo in data["geocodes"]:
                    addr = geo.get("formatted_address", "")
                    prov = geo.get("province", "")
                    if _is_hk_or_border(addr, prov):
                        return self._build_geo_result(query, geo, addr)

            return GeocodedPlace(query=query, found=False)
        except Exception:
            return GeocodedPlace(query=query, found=False)

    @staticmethod
    def _build_geo_result(query: str, geo: dict, addr: str) -> GeocodedPlace:
        loc_str = geo.get("location", "")
        lat = None
        lng = None
        if loc_str and "," in loc_str:
            parts = loc_str.split(",")
            try:
                lng = float(parts[0])
                lat = float(parts[1]) if len(parts) >= 2 else None
            except (ValueError, IndexError):
                pass
        return GeocodedPlace(
            query=query, found=True,
            formatted_address=addr,
            lat=lat,
            lng=lng,
            place_type=geo.get("level", "未知"),
        )

    def batch_geocode_places(self, plan_steps: list[str]) -> dict[str, GeocodedPlace]:
        """从规划步骤中提取知名地名并批量验证（白名单匹配，避免正则贪心）。"""
        import re
        if not self.config.amap_api_key:
            return {}

        known = re.compile(
            r"(香港科技大学|香港中文大学|香港大学|太平山顶|维多利亚港|维多利亚公园"
            r"|星光大道|庙街夜市|庙街|兰桂坊|海洋公园|迪士尼乐园|迪士尼"
            r"|天星小轮|天星码头|西九龙文化区|西九龙站|西九龙"
            r"|铜锣湾|尖沙咀|旺角|中环|湾仔|上环|金钟|红磡|九龙塘|彩虹站|彩虹"
            r"|将军澳|西贡|元朗|屯门|沙田|大埔|上水|粉岭|落马洲|罗湖|福田|皇岗|深圳湾"
            r"|清水湾|南丫岛|昂坪|大澳|东涌|欣澳|青衣|荃湾|葵涌|深水埗|观塘|黄大仙"
            r"|罗湖口岸|福田口岸|深圳湾口岸|皇岗口岸|港珠澳大桥口岸|文锦渡口岸|沙头角口岸"
            r"|西九龙高铁站|九龙站|香港站|九龙塘站|红磡站|落马洲站|罗湖站|上水站|粉岭站"
            r"|太和站|大埔墟站|大学站|火炭站|沙田站|大围站|旺角东站|金钟站)"
        )
        places: dict[str, GeocodedPlace] = {}
        seen: set[str] = set()
        for step in plan_steps:
            for match in known.finditer(step):
                name = match.group(1)
                if name not in seen:
                    seen.add(name)
                    places[name] = self.geocode_place(name)
        return places

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
