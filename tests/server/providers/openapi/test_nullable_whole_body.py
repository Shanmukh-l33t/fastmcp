"""An explicit whole-body JSON null must reach the upstream API as `null`."""

from typing import Any

import httpx2
import pytest

from fastmcp import Client, FastMCP

BASE_URL = "https://example.test"


def _spec(
    body_schema: dict[str, Any],
    *,
    version: str = "3.1.0",
    required: bool = True,
    media_type: str = "application/json",
) -> dict[str, Any]:
    return {
        "openapi": version,
        "info": {"title": "Nullable body", "version": "1.0"},
        "paths": {
            "/value": {
                "post": {
                    "operationId": "set_value",
                    "requestBody": {
                        "required": required,
                        "content": {media_type: {"schema": body_schema}},
                    },
                    "responses": {"200": {"description": "OK"}},
                }
            }
        },
    }


async def _call(
    spec: dict[str, Any], arguments: dict[str, Any], mode: str = "auto"
) -> list[httpx2.Request]:
    requests: list[httpx2.Request] = []

    def capture(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"ok": True})

    async with httpx2.AsyncClient(
        base_url=BASE_URL, transport=httpx2.MockTransport(capture)
    ) as http:
        server = FastMCP.from_openapi(openapi_spec=spec, client=http)
        async with Client(server, mode=mode) as client:
            await client.call_tool("set_value", arguments)
    return requests


@pytest.mark.parametrize("mode", ["auto", "legacy"])
@pytest.mark.parametrize(
    ("version", "body_schema"),
    [
        ("3.1.0", {"type": ["string", "null"]}),
        ("3.0.3", {"type": "string", "nullable": True}),
        ("3.1.0", {"anyOf": [{"type": "string"}, {"type": "null"}]}),
        ("3.1.0", {"oneOf": [{"type": "string"}, {"type": "null"}]}),
    ],
)
async def test_explicit_whole_body_null_is_sent_upstream(
    version: str, body_schema: dict[str, Any], mode: str
) -> None:
    [request] = await _call(_spec(body_schema, version=version), {"body": None}, mode)

    assert request.content == b"null"
    assert request.headers["content-type"] == "application/json"


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_omitted_optional_body_sends_no_body(mode: str) -> None:
    spec = _spec({"type": ["string", "null"]}, required=False)

    [request] = await _call(spec, {}, mode)

    assert request.content == b""
    assert "content-type" not in request.headers


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_non_null_whole_body_is_unchanged(mode: str) -> None:
    spec = _spec({"type": ["string", "null"]})

    [request] = await _call(spec, {"body": "hello"}, mode)

    assert request.content == b'"hello"'
    assert request.headers["content-type"] == "application/json"
