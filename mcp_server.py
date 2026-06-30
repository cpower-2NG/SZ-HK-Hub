"""SZ-HK Hub MCP Server —— 口岸 / 港铁 / 路线规划 / 文件操作。

基于 FastAPI（Gradio 自带依赖），无需额外安装。
启动: python mcp_server.py
默认监听 http://localhost:3333
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request
import uvicorn

app = FastAPI(title="SZ-HK Hub MCP Server", version="1.0.0")

# ── 用户数据目录 ──────────────────────────────────────────
USER_DATA_DIR = Path(os.getenv("USER_DATA_PATH", "./user_data")).resolve()


def _ensure_user_data_dir() -> None:
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── 模拟数据：口岸排队 ────────────────────────────────────

PORT_TRAFFIC: dict[str, tuple[int, int]] = {
    "深圳湾": (10, 30),
    "福田": (15, 40),
    "罗湖": (20, 50),
    "皇岗": (5, 20),
    "港珠澳大桥": (5, 15),
    "文锦渡": (8, 25),
    "沙头角": (5, 15),
}

# ── 模拟数据：港铁班次 ────────────────────────────────────

MTR_SCHEDULE: dict[str, tuple[int, int]] = {
    "西九龙": (3, 8),
    "罗湖": (3, 8),
    "落马洲": (5, 12),
    "红磡": (2, 6),
    "尖沙咀": (2, 5),
    "旺角东": (2, 5),
    "九龙塘": (2, 5),
    "上水": (4, 10),
    "金钟": (2, 4),
    "中环": (2, 4),
    "铜锣湾": (2, 5),
}

# ── 模拟数据：深港跨境路线 ────────────────────────────────
# 预设路线优先匹配；未命中时若配置了 GOOGLE_MAPS_API_KEY 则调用真实 API

SZ_HK_ROUTES: dict[str, list[dict]] = {
    "深圳湾→西九龙": [
        {"mode": "巴士 B2P → 港铁屯马线", "duration_min": 55, "cost_hkd": 25,
         "note": "深圳湾口岸过关后乘 B2P 到天水围站转屯马线"},
        {"mode": "的士", "duration_min": 40, "cost_hkd": 350,
         "note": "直接打车经深圳湾大桥"},
    ],
    "福田→西九龙": [
        {"mode": "港铁东铁线 → 屯马线", "duration_min": 50, "cost_hkd": 40,
         "note": "福田口岸过关后落马洲站出发"},
        {"mode": "高铁", "duration_min": 14, "cost_hkd": 80,
         "note": "福田站 → 西九龙站，最快"},
    ],
    "深圳湾→中环": [
        {"mode": "巴士 B2P → 港铁", "duration_min": 75, "cost_hkd": 35,
         "note": "天水围站转屯马线到南昌再转东涌线"},
        {"mode": "的士", "duration_min": 50, "cost_hkd": 400,
         "note": "经西隧直达中环"},
    ],
    "福田→中环": [
        {"mode": "港铁东铁线 → 金钟转港岛线", "duration_min": 65, "cost_hkd": 50,
         "note": "落马洲站直达金钟换乘"},
        {"mode": "高铁 → 港铁", "duration_min": 35, "cost_hkd": 90,
         "note": "西九龙站出站后步行至柯士甸站换乘"},
    ],
    "罗湖→尖沙咀": [
        {"mode": "港铁东铁线", "duration_min": 42, "cost_hkd": 40,
         "note": "罗湖站直达尖东站"},
        {"mode": "港铁东铁线 → 屯马线", "duration_min": 45, "cost_hkd": 40,
         "note": "红磡站换乘屯马线到尖东"},
    ],
    "皇岗→旺角": [
        {"mode": "皇巴 → 港铁", "duration_min": 50, "cost_hkd": 30,
         "note": "皇岗口岸乘皇巴到落马洲再转港铁"},
        {"mode": "跨境巴士", "duration_min": 45, "cost_hkd": 45,
         "note": "直达旺角太子"},
    ],
    "港珠澳大桥→东涌": [
        {"mode": "金巴 → 巴士", "duration_min": 60, "cost_hkd": 70,
         "note": "港珠澳大桥口岸乘金巴到香港口岸再转B6"},
        {"mode": "跨境巴士", "duration_min": 50, "cost_hkd": 120,
         "note": "直达东涌"},
    ],
}

_DEFAULT_ROUTES: list[dict] = [
    {"mode": "港铁（推荐）", "duration_min": 55, "cost_hkd": 40,
     "note": "请指定具体起终点以获取精确路线"},
    {"mode": "的士", "duration_min": 45, "cost_hkd": 350,
     "note": "跨境的士较贵但最方便"},
]


def _random_range(key: str, mapping: dict[str, tuple[int, int]]) -> int:
    if key in mapping:
        lo, hi = mapping[key]
        return random.randint(lo, hi)
    return random.randint(10, 30)


def _lookup_route(origin: str, destination: str) -> list[dict]:
    """查找预设路线，支持模糊匹配。"""
    key = f"{origin}→{destination}"
    if key in SZ_HK_ROUTES:
        return SZ_HK_ROUTES[key]
    for route_key, routes in SZ_HK_ROUTES.items():
        if origin in route_key and destination in route_key:
            return routes
    return _DEFAULT_ROUTES


# ── 端点：口岸排队 ────────────────────────────────────────


@app.post("/tools/port_traffic")
async def port_traffic(request: Request):
    body = await request.json()
    port = body.get("port", "深圳湾")
    queue_minutes = _random_range(port, PORT_TRAFFIC)
    return {
        "result": {
            "port": port,
            "queue_minutes": queue_minutes,
            "updated_at": datetime.now().isoformat(),
        }
    }


# ── 端点：港铁班次 ────────────────────────────────────────


@app.post("/tools/mtr_schedule")
async def mtr_schedule(request: Request):
    body = await request.json()
    station = body.get("station", "西九龙")
    interval_minutes = _random_range(station, MTR_SCHEDULE)
    return {
        "result": {
            "station": station,
            "interval_minutes": interval_minutes,
            "updated_at": datetime.now().isoformat(),
        }
    }


# ── 端点：路线规划 ────────────────────────────────────────


@app.post("/tools/route_planner")
async def route_planner(request: Request):
    """跨境路线规划。预设路线优先；未命中时调用 Google Maps Directions API。"""
    body = await request.json()
    origin = body.get("origin", "深圳湾")
    destination = body.get("destination", "西九龙")
    google_key = os.getenv("GOOGLE_MAPS_API_KEY")

    # 1) 优先匹配预设路线（零 API 消耗）
    preset = _lookup_route(origin, destination)
    source = "preset"
    if preset is not _DEFAULT_ROUTES:
        return {
            "result": {
                "origin": origin,
                "destination": destination,
                "routes": preset,
                "source": source,
                "updated_at": datetime.now().isoformat(),
            }
        }

    # 2) 预设未命中，有 Google Key 则调用真实 API
    if google_key:
        try:
            gm_routes = _call_google_maps(google_key, origin, destination)
            if gm_routes:
                source = "google_maps"
                return {
                    "result": {
                        "origin": origin,
                        "destination": destination,
                        "routes": gm_routes,
                        "source": source,
                        "updated_at": datetime.now().isoformat(),
                    }
                }
        except Exception:
            pass  # API 失败则回退到默认路线

    # 3) 回退：返回默认路线
    return {
        "result": {
            "origin": origin,
            "destination": destination,
            "routes": _DEFAULT_ROUTES,
            "source": "mock",
            "updated_at": datetime.now().isoformat(),
        }
    }


def _call_google_maps(api_key: str, origin: str, destination: str) -> list[dict]:
    """调用 Google Maps Directions API，返回与本项目统一格式的路线列表。"""
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": destination,
        "key": api_key,
        "mode": "transit",            # 公交/地铁优先
        "alternatives": "true",
        "language": "zh-CN",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "OK":
        return []

    routes: list[dict] = []
    for route in data.get("routes", [])[:2]:  # 最多取 2 条
        legs = route.get("legs", [{}])
        leg = legs[0] if legs else {}
        duration_min = round(leg.get("duration", {}).get("value", 0) / 60)
        distance_km = leg.get("distance", {}).get("value", 0) / 1000

        # 构建交通方式描述
        steps = leg.get("steps", [])
        modes: list[str] = []
        for step in steps[:4]:
            mode = step.get("travel_mode", "")
            if mode == "TRANSIT":
                detail = step.get("transit_details", {})
                line = detail.get("line", {}).get("short_name", "")
                modes.append(line if line else "公交")
            elif mode:
                modes.append({"WALKING": "步行", "DRIVING": "驾车"}.get(mode, mode))

        mode_str = " → ".join(modes) if modes else "公交/地铁"

        routes.append({
            "mode": mode_str,
            "duration_min": duration_min,
            "cost_hkd": round(distance_km * 1.5, 0),  # 粗略估算 ~1.5 HKD/km
            "note": f"Google Maps 实时路线，全程约 {distance_km:.1f} km",
        })

    return routes


# ── 端点：文件操作 ────────────────────────────────────────


@app.post("/tools/file_ops")
async def file_ops(request: Request):
    """本地文件读写。支持 save / load / list 三种操作。"""
    _ensure_user_data_dir()
    body = await request.json()
    action = body.get("action", "load")
    filename = body.get("filename", "default.json")

    # 安全检查：防止路径穿越
    safe_name = Path(filename).name
    filepath = USER_DATA_DIR / safe_name

    if action == "save":
        data = body.get("data")
        if data is None:
            raise HTTPException(status_code=400, detail="缺少 data 字段")
        filepath.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            "result": {
                "action": "save",
                "filename": safe_name,
                "status": "ok",
                "timestamp": datetime.now().isoformat(),
            }
        }

    elif action == "load":
        if not filepath.exists():
            return {
                "result": {
                    "action": "load",
                    "filename": safe_name,
                    "status": "not_found",
                    "data": None,
                }
            }
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="文件格式错误，无法解析 JSON")
        return {
            "result": {
                "action": "load",
                "filename": safe_name,
                "status": "ok",
                "data": data,
                "timestamp": datetime.now().isoformat(),
            }
        }

    elif action == "list":
        files = [
            {
                "name": f.name,
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            }
            for f in USER_DATA_DIR.glob("*.json")
        ]
        return {"result": {"action": "list", "files": files}}

    else:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的操作: {action}，支持: save / load / list",
        )


# ── 健康检查 ──────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "tools": ["port_traffic", "mtr_schedule", "route_planner", "file_ops"],
        "timestamp": datetime.now().isoformat(),
    }


# ── 入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=3333, log_level="info")
