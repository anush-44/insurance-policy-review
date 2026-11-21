"""Regression checks for the live Gemini structured request."""

from __future__ import annotations

import json

from app.core.llm import GeminiLLMClient
from app.core.schemas import GroundedAnswer


def test_structured_request_uses_json_schema_and_header(monkeypatch) -> None:
    captured = {}

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "answer": "42",
                                            "citations": [],
                                            "confidence": 1.0,
                                            "is_abstention": False,
                                            "caveats": [],
                                            "retrieved_chunk_ids": [],
                                        }
                                    )
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 12},
                "modelVersion": "gemini-test",
            }

    class Client:
        def __init__(self, timeout: float) -> None:
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def post(self, url: str, json: dict, headers: dict) -> Response:
            captured.update(url=url, payload=json, headers=headers)
            return Response()

    monkeypatch.setattr("app.core.llm.httpx.Client", Client)
    client = GeminiLLMClient(api_key="test-secret", model_name="gemini-test")
    answer = client.generate_structured("What is the answer?", GroundedAnswer)

    assert answer.answer == "42"
    assert "test-secret" not in captured["url"]
    assert captured["headers"] == {"x-goog-api-key": "test-secret"}
    config = captured["payload"]["generationConfig"]
    assert "responseJsonSchema" in config and "responseSchema" not in config
    assert client.last_usage_metadata == {"promptTokenCount": 12}
    assert client.last_model_version == "gemini-test"
