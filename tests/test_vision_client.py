from __future__ import annotations

import pytest

from errors import ServiceError
from vision_client import VisionClient


PNG_HEADER = b"\x89PNG\r\n\x1a\n"
JPG_HEADER = b"\xff\xd8\xff\xe0"
GIF_HEADER = b"GIF89a"
WEBP_HEADER = b"RIFFxxxxWEBP"


def test_detect_media_type_headers(make_config) -> None:
    client = VisionClient(make_config())
    assert client._detect_media_type(PNG_HEADER + b"x") == "image/png"
    assert client._detect_media_type(JPG_HEADER + b"x") == "image/jpeg"
    assert client._detect_media_type(GIF_HEADER + b"x") == "image/gif"
    assert client._detect_media_type(WEBP_HEADER + b"x") == "image/webp"


def test_encode_image_empty_raises(make_config) -> None:
    client = VisionClient(make_config())
    with pytest.raises(ServiceError, match="未找到上传的图片内容"):
        client._encode_image(b"")


def test_extract_events_from_json_text(make_config) -> None:
    client = VisionClient(make_config())
    events = client._extract_events('{"events":[{"date":"2026-05-20","time":"10:00","title":"开户"}]}')
    assert events == [{"date": "2026-05-20", "time": "10:00", "title": "开户"}]


def test_extract_events_from_wrapped_json_block(make_config) -> None:
    client = VisionClient(make_config())
    text = '模型输出如下：```json\\n{"events":[{"date":"待确认日期","time":"09:30","title":"活动"}]}\\n```'
    events = client._extract_events(text)
    assert events[0]["time"] == "09:30"
    assert events[0]["title"] == "活动"


def test_extract_events_requires_list(make_config) -> None:
    client = VisionClient(make_config())
    with pytest.raises(ServiceError, match="events 列表"):
        client._extract_events('{"events":{}}')


def test_parse_events_routes_by_provider(make_config, monkeypatch) -> None:
    client = VisionClient(make_config(vision_provider="anthropic"))

    monkeypatch.setattr(client, "_parse_with_anthropic", lambda data: [{"date": "d", "time": "t", "title": "a"}])
    monkeypatch.setattr(client, "_parse_with_openai", lambda data: [{"date": "x", "time": "y", "title": "z"}])

    events = client.parse_events(PNG_HEADER + b"demo")
    assert events[0]["title"] == "a"
