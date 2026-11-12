"""OpenAPI component argument handling: undeclared arguments, strict input
validation of advertised schemas, and configured headers."""

import json
from typing import Any

import httpx2
import pytest
from jsonschema_path import SchemaPath

from fastmcp import Client, FastMCP
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError
from fastmcp.server.providers.openapi.components import OpenAPITool
from fastmcp.server.providers.openapi.routing import MCPType, RouteMap
from fastmcp.utilities.openapi import director, parser
from fastmcp.utilities.openapi.director import RequestDirector
from fastmcp.utilities.openapi.models import (
    HTTPRoute,
    ParameterInfo,
    RequestBodyInfo,
)
from fastmcp.utilities.tests import run_server_async

BASE_URL = "https://api.example.com"
CONFIGURED_HEADERS = {
    "Authorization": "Bearer configured",
    "X-Api-Key": "configured-key",
    "X-Roles": "readonly",
}
CONFIGURED_COOKIES = {"session": "configured-session"}

UNDECLARED_ARGUMENTS = [
    pytest.param({"Authorization__header": "Bearer other"}, id="authorization-header"),
    pytest.param({"X-Roles__header": "admin"}, id="custom-header"),
    pytest.param({"Host__header": "other.invalid"}, id="host-header"),
    pytest.param({"session__cookie": "other-session"}, id="cookie"),
    pytest.param({"include_private__query": "true"}, id="query"),
    pytest.param({"extra": {"nested": "value"}}, id="body"),
    pytest.param({"X-Api-Key__header": "other-key"}, id="configured-api-key"),
    pytest.param({"note__header": "hello"}, id="note-header"),
    pytest.param({"theme__cookie": "dark"}, id="theme-cookie"),
    pytest.param({"extra__query": "true"}, id="extra-query"),
    pytest.param({"debug": True}, id="debug-body"),
]


def _spec(path: str, method: str, operation: dict[str, Any]) -> dict[str, Any]:
    return {
        "openapi": "3.1.0",
        "info": {"title": "Test API", "version": "1.0"},
        "paths": {
            path: {
                method: {
                    "operationId": "operation",
                    "responses": {"200": {"description": "OK"}},
                    **operation,
                }
            }
        },
    }


def _upstream_client(requests: list[httpx2.Request]) -> httpx2.AsyncClient:
    def capture(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"ok": True})

    return httpx2.AsyncClient(
        base_url=BASE_URL,
        headers=CONFIGURED_HEADERS,
        cookies=CONFIGURED_COOKIES,
        transport=httpx2.MockTransport(capture),
    )


async def _call_tool(
    spec: dict[str, Any], arguments: dict[str, Any]
) -> list[httpx2.Request]:
    requests: list[httpx2.Request] = []
    async with _upstream_client(requests) as http_client:
        server = FastMCP.from_openapi(openapi_spec=spec, client=http_client)
        async with Client(server) as client:
            await client.call_tool("operation", arguments)
    return requests


def _assert_sent_as_configured(request: httpx2.Request) -> None:
    assert request.headers["Authorization"] == "Bearer configured"
    assert request.headers["X-Api-Key"] == "configured-key"
    assert request.headers["X-Roles"] == "readonly"
    assert request.headers["Host"] == "api.example.com"
    assert request.headers["Cookie"] == "session=configured-session"
    assert "note" not in request.headers


@pytest.mark.parametrize("method", ["get", "post"])
@pytest.mark.parametrize("arguments", UNDECLARED_ARGUMENTS)
async def test_undeclared_arguments_are_ignored(
    method: str, arguments: dict[str, Any]
) -> None:
    spec = _spec("/health", method, {})

    [request] = await _call_tool(spec, arguments)

    _assert_sent_as_configured(request)
    assert str(request.url) == f"{BASE_URL}/health"
    assert request.content == b""


async def test_strict_input_validation_rejects_undeclared_arguments() -> None:
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    request_body = {"content": {"application/json": {"schema": schema}}}
    spec = _spec("/agents", "post", {"requestBody": request_body})

    requests: list[httpx2.Request] = []
    async with _upstream_client(requests) as http_client:
        server = FastMCP.from_openapi(
            openapi_spec=spec, client=http_client, strict_input_validation=True
        )
        async with Client(server) as client:
            with pytest.raises(
                ToolError, match="Unknown arguments: usage_limits. Accepted: name"
            ):
                await client.call_tool(
                    "operation", {"name": "agent", "usage_limits": {"requests": 5}}
                )

    assert requests == []


async def test_undeclared_arguments_are_ignored_after_schema_precalculation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("schema pre-calculation failed")

    monkeypatch.setattr(parser, "_combine_schemas_and_map_params", fail)
    limit = {"name": "limit", "in": "query", "schema": {"type": "integer"}}
    spec = _spec("/items", "get", {"parameters": [limit]})

    [request] = await _call_tool(
        spec,
        {
            "limit": 5,
            "Authorization__header": "Bearer other",
            "note__header": "hello",
            "debug": True,
        },
    )

    _assert_sent_as_configured(request)
    assert dict(request.url.params) == {"limit": "5"}
    assert request.content == b""


async def test_tool_call_fails_when_parameter_map_cannot_be_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("schema pre-calculation failed")

    monkeypatch.setattr(parser, "_combine_schemas_and_map_params", fail)
    monkeypatch.setattr(director, "_combine_schemas_and_map_params", fail)
    limit = {"name": "limit", "in": "query", "schema": {"type": "integer"}}
    spec = _spec("/items", "get", {"parameters": [limit]})

    requests: list[httpx2.Request] = []
    async with _upstream_client(requests) as http_client:
        server = FastMCP.from_openapi(openapi_spec=spec, client=http_client)
        async with Client(server) as client:
            with pytest.raises(ToolError, match="schema pre-calculation failed"):
                await client.call_tool(
                    "operation", {"limit": 5, "Authorization__header": "Bearer other"}
                )

    assert requests == []


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param({"type": "object"}, id="default-additional-properties"),
        pytest.param(
            {"type": "object", "additionalProperties": True},
            id="additional-properties",
        ),
    ],
)
async def test_whole_body_is_sent_unchanged(
    schema: dict[str, Any],
) -> None:
    body_schema = {"$ref": "#/components/schemas/Thing"}
    request_body = {
        "required": True,
        "content": {"application/json": {"schema": body_schema}},
    }
    spec = _spec("/things", "post", {"requestBody": request_body})
    spec["components"] = {"schemas": {"Thing": schema}}
    body = {
        "Authorization__header": "body data",
        "note__header": "body data",
        "debug": True,
    }

    [request] = await _call_tool(
        spec,
        {
            "body": body,
            "Authorization__header": "Bearer other",
            "note__header": "hello",
        },
    )

    _assert_sent_as_configured(request)
    assert json.loads(request.content) == body


@pytest.mark.parametrize(
    "component",
    ["tool", "resource", "resource_template"],
)
async def test_configured_headers_keep_their_values(
    component: str,
) -> None:
    ok = {"200": {"description": "OK"}}
    item_id = {"name": "item_id", "in": "path", "required": True, "schema": {}}
    spec = {
        "openapi": "3.1.0",
        "info": {"title": "Test API", "version": "1.0"},
        "paths": {
            "/actions": {"post": {"operationId": "act", "responses": ok}},
            "/status": {"get": {"operationId": "status", "responses": ok}},
            "/items/{item_id}": {
                "get": {"operationId": "item", "parameters": [item_id], "responses": ok}
            },
        },
    }
    route_maps = [
        RouteMap(pattern=r"\{", mcp_type=MCPType.RESOURCE_TEMPLATE),
        RouteMap(methods=["GET"], mcp_type=MCPType.RESOURCE),
    ]
    requests: list[httpx2.Request] = []

    async with _upstream_client(requests) as http_client:
        server = FastMCP.from_openapi(
            openapi_spec=spec, client=http_client, route_maps=route_maps
        )
        async with run_server_async(server) as url:
            transport = StreamableHttpTransport(
                url, headers={"X-Api-Key": "client-key", "X-Client-Tag": "forwarded"}
            )
            async with Client(transport) as client:
                if component == "tool":
                    await client.call_tool("act", {})
                elif component == "resource":
                    await client.read_resource("resource://status")
                else:
                    await client.read_resource("resource://item/7")

    [request] = requests
    _assert_sent_as_configured(request)
    assert request.headers["X-Client-Tag"] == "forwarded"


def test_manual_route_uses_canonical_parameter_names() -> None:
    route = HTTPRoute(
        path="/things/{id}",
        method="GET",
        parameters=[
            ParameterInfo(
                name="id", location="path", required=True, schema={"type": "string"}
            ),
            ParameterInfo(name="id", location="query", schema={"type": "string"}),
            ParameterInfo(name="id", location="header", schema={"type": "string"}),
        ],
    )
    request_director = RequestDirector(SchemaPath.from_dict({}))

    request = request_director.build(
        route,
        {
            "id__path": "path-value",
            "id__query": "query-value",
            "id__header": "header-value",
            "Authorization__header": "Bearer other",
            "note__header": "hello",
            "debug": True,
        },
        BASE_URL,
    )

    assert request.url.path == "/things/path-value"
    assert dict(request.url.params) == {"id": "query-value"}
    assert request.headers["id"] == "header-value"
    assert "Authorization" not in request.headers
    assert "note" not in request.headers
    assert request.content == b""


def test_manual_route_accepts_location_suffix_for_declared_parameter() -> None:
    route = HTTPRoute(
        path="/users/{id}",
        method="GET",
        parameters=[
            ParameterInfo(
                name="id", location="path", required=True, schema={"type": "integer"}
            )
        ],
    )
    request_director = RequestDirector(SchemaPath.from_dict({}))

    request = request_director.build(
        route,
        {
            "id__path": 123,
            "id__header": "header-value",
            "id__query": "query-value",
            "Authorization__header": "Bearer other",
        },
        BASE_URL,
    )

    assert str(request.url) == f"{BASE_URL}/users/123"
    assert "id" not in request.headers
    assert "Authorization" not in request.headers
    assert request.content == b""


def test_manual_route_with_allof_body_sends_declared_properties() -> None:
    content_schema = {
        "application/json": {"allOf": [{"$ref": "#/components/schemas/Thing"}]}
    }
    route = HTTPRoute(
        path="/things",
        method="POST",
        request_body=RequestBodyInfo(content_schema=content_schema),
        request_schemas={
            "Thing": {"type": "object", "properties": {"name": {"type": "string"}}}
        },
    )
    request_director = RequestDirector(SchemaPath.from_dict({}))

    request = request_director.build(
        route,
        {
            "name": "thing",
            "Authorization__header": "Bearer other",
            "note__header": "hello",
        },
        BASE_URL,
    )

    assert json.loads(request.content) == {"name": "thing"}
    assert "Authorization" not in request.headers
    assert "note" not in request.headers
    assert route.request_body is not None
    assert route.request_body.content_schema == {
        "application/json": {"allOf": [{"$ref": "#/components/schemas/Thing"}]}
    }


async def test_strict_validation_accepts_suffix_on_manual_route() -> None:
    route = HTTPRoute(
        path="/users/{id}",
        method="GET",
        parameters=[
            ParameterInfo(
                name="id", location="path", required=True, schema={"type": "integer"}
            )
        ],
    )
    requests: list[httpx2.Request] = []
    async with _upstream_client(requests) as http_client:
        server = FastMCP(strict_input_validation=True)
        server.add_tool(
            OpenAPITool(
                client=http_client,
                route=route,
                director=RequestDirector(SchemaPath.from_dict({})),
                name="get_user",
                description="",
                parameters={
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                },
            )
        )
        async with Client(server) as client:
            await client.call_tool("get_user", {"id__path": 123})
            with pytest.raises(ToolError, match="Unknown arguments: debug"):
                await client.call_tool("get_user", {"id": 1, "debug": True})

    assert [str(r.url) for r in requests] == [f"{BASE_URL}/users/123"]


NESTED_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "options": {
            "type": "object",
            "properties": {"level": {"type": "integer"}},
            "required": ["level"],
            "additionalProperties": False,
        }
    },
    "required": ["options"],
}


def _json_body_spec(schema: dict[str, Any]) -> dict[str, Any]:
    request_body = {
        "required": True,
        "content": {"application/json": {"schema": schema}},
    }
    return _spec("/run", "post", {"requestBody": request_body})


async def _call_tool_strict(
    spec: dict[str, Any],
    arguments: dict[str, Any],
    requests: list[httpx2.Request] | None = None,
) -> list[httpx2.Request]:
    """Call a strict server's ``operation`` tool; return dispatched requests.

    Pass a pre-created ``requests`` list to observe dispatches even when the
    call is expected to raise.
    """
    dispatched: list[httpx2.Request] = [] if requests is None else requests
    async with _upstream_client(dispatched) as http_client:
        server = FastMCP.from_openapi(
            openapi_spec=spec, client=http_client, strict_input_validation=True
        )
        async with Client(server) as client:
            await client.call_tool("operation", arguments)
    return dispatched


async def test_strict_validation_rejects_invalid_nested_values() -> None:
    # The reported scenario: options.level must be an integer and options
    # forbids extra keys, yet the invalid dict reached the transport.
    spec = _json_body_spec(NESTED_OBJECT_SCHEMA)
    dispatched: list[httpx2.Request] = []
    with pytest.raises(ToolError) as exc_info:
        await _call_tool_strict(
            spec, {"options": {"level": "one", "extra": 1}}, dispatched
        )

    assert dispatched == []
    message = str(exc_info.value)
    assert "does not match its schema" in message
    assert "'one' is not of type 'integer'" in message
    assert "Additional properties are not allowed" in message


async def test_strict_validation_rejects_missing_nested_required_property() -> None:
    spec = _json_body_spec(NESTED_OBJECT_SCHEMA)
    dispatched: list[httpx2.Request] = []
    with pytest.raises(ToolError) as exc_info:
        await _call_tool_strict(spec, {"options": {}}, dispatched)

    assert dispatched == []
    assert "'level' is a required property" in str(exc_info.value)


async def test_strict_validation_rejects_invalid_values_behind_refs() -> None:
    schema = {
        "type": "object",
        "properties": {"options": {"$ref": "#/components/schemas/Options"}},
        "required": ["options"],
    }
    spec = _json_body_spec(schema)
    spec["components"] = {
        "schemas": {
            "Options": {
                "type": "object",
                "properties": {"level": {"type": "integer"}},
                "required": ["level"],
                "additionalProperties": False,
            }
        }
    }
    dispatched: list[httpx2.Request] = []
    with pytest.raises(ToolError) as exc_info:
        await _call_tool_strict(spec, {"options": {"level": "one"}}, dispatched)

    assert dispatched == []
    assert "'one' is not of type 'integer'" in str(exc_info.value)


async def test_strict_validation_rejects_invalid_array_item_values() -> None:
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
                    "additionalProperties": False,
                },
            }
        },
    }
    spec = _json_body_spec(schema)
    dispatched: list[httpx2.Request] = []
    with pytest.raises(ToolError) as exc_info:
        await _call_tool_strict(
            spec, {"rows": [{"count": 1}, {"count": "two"}]}, dispatched
        )

    assert dispatched == []
    assert "'two' is not of type 'integer'" in str(exc_info.value)


async def test_strict_validation_accepts_valid_nested_values() -> None:
    closed = await _call_tool_strict(
        _json_body_spec(NESTED_OBJECT_SCHEMA), {"options": {"level": 2}}
    )
    assert [json.loads(request.content) for request in closed] == [
        {"options": {"level": 2}}
    ]

    open_schema = {
        "type": "object",
        "properties": {
            "options": {"type": "object", "properties": {"level": {"type": "integer"}}}
        },
    }
    opened = await _call_tool_strict(
        _json_body_spec(open_schema), {"options": {"level": 2, "extra": 1}}
    )
    assert [json.loads(request.content) for request in opened] == [
        {"options": {"level": 2, "extra": 1}}
    ]


async def test_nested_schema_violations_dispatch_without_strict_validation() -> None:
    spec = _json_body_spec(NESTED_OBJECT_SCHEMA)
    [request] = await _call_tool(spec, {"options": {"level": "one", "extra": 1}})
    assert json.loads(request.content) == {"options": {"level": "one", "extra": 1}}
