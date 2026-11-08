"""Success responses that declare different JSON bodies (e.g. 200 vs 202).

See https://github.com/PrefectHQ/fastmcp/issues/6265
"""

from typing import Any

import httpx2
import pytest

from fastmcp import Client, FastMCP


def json_response(field: str) -> dict[str, Any]:
    return {
        "description": "Successful response",
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {field: {"type": "string"}},
                    "required": [field],
                }
            }
        },
    }


SPEC: dict[str, Any] = {
    "openapi": "3.1.0",
    "info": {"title": "Reports", "version": "1.0"},
    "paths": {
        "/reports": {
            "post": {
                "operationId": "generate_report",
                "responses": {
                    "200": json_response("report_id"),
                    "202": json_response("job_id"),
                },
            }
        }
    },
}


def upstream_returning_202(request: httpx2.Request) -> httpx2.Response:
    assert request.method == "POST"
    return httpx2.Response(202, json={"job_id": "job-42"})


def upstream_returning_200(request: httpx2.Request) -> httpx2.Response:
    assert request.method == "POST"
    return httpx2.Response(200, json={"report_id": "report-7"})


def upstream_returning_an_undeclared_body(request: httpx2.Request) -> httpx2.Response:
    assert request.method == "POST"
    return httpx2.Response(202, json={"nope": 1})


async def call_with(upstream: Any, arguments: dict[str, Any] | None = None):
    async with httpx2.AsyncClient(
        base_url="https://reports.example.test",
        transport=httpx2.MockTransport(upstream),
    ) as http:
        server = FastMCP.from_openapi(openapi_spec=SPEC, client=http)
        async with Client(server) as client:
            return await client.call_tool("generate_report", arguments or {})


async def test_declared_202_success_reaches_the_caller():
    """The declared 202 body must not be validated against the 200 schema."""
    result = await call_with(upstream_returning_202)

    assert result.is_error is False
    assert result.structured_content == {"job_id": "job-42"}


async def test_declared_200_success_still_reaches_the_caller():
    result = await call_with(upstream_returning_200)

    assert result.is_error is False
    assert result.structured_content == {"report_id": "report-7"}


async def test_body_matching_no_declared_success_response_is_still_rejected():
    with pytest.raises(RuntimeError, match="Invalid structured content"):
        await call_with(upstream_returning_an_undeclared_body)


async def test_advertised_output_schema_accepts_both_declared_bodies():
    async with httpx2.AsyncClient(
        base_url="https://reports.example.test",
        transport=httpx2.MockTransport(upstream_returning_202),
    ) as http:
        server = FastMCP.from_openapi(openapi_spec=SPEC, client=http)
        async with Client(server) as client:
            (tool,) = await client.list_tools()

    assert tool.output_schema is not None
    from jsonschema import validators

    validator = validators.validator_for(tool.output_schema)(tool.output_schema)
    validator.validate({"report_id": "report-7"})
    validator.validate({"job_id": "job-42"})
