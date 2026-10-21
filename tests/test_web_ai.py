"""AI cannot execute code or bypass deterministic data contracts."""

import json

import httpx
import pytest

from pylustrator.web.ai import ProviderError, list_providers, propose_plan
from pylustrator.web.datasets import ValidationError, parse_dataset


@pytest.fixture
def dataset():
    table = parse_dataset("sample,value\nsecret-name,1\nother-name,3\nmissing,\n")
    table["id"] = "test-dataset"
    return table


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("PYL_FIGURE_AI_API_KEY", "private-test-key")
    monkeypatch.setenv("PYL_FIGURE_AI_MODEL", "test-model")
    monkeypatch.setenv("PYL_FIGURE_AI_BASE_URL", "https://provider.example/v1")


def _client(result, status=200):
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        status, json={"choices": [{"message": {"content": json.dumps(result)}}]})))


def test_local_helper_is_labeled_and_produces_exact_previews(dataset):
    plan = propose_plan(dataset, "Drop missing value and normalize value", "local")
    assert plan["provider"] == "local"
    assert plan["model"] == "deterministic-rules"
    assert plan["dataShared"] == "none"
    assert plan["inputHash"] == dataset["hash"]
    assert plan["previewSummary"]["inputRows"] == 3
    assert plan["previewSummary"]["outputRows"] == 2
    assert plan["steps"][1]["op"] == "derive"


def test_external_request_contains_metadata_but_no_rows_or_public_secret(dataset, configured):
    seen = []
    def respond(request):
        seen.append(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer private-test-key"
        result = {"steps": [{"op": "drop_missing", "columns": ["value"]}], "explanation": "Remove missing values."}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}}]})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        plan = propose_plan(dataset, "Clean value", "openai-compatible", client=client)
    assert plan["previewSummary"]["outputRows"] == 2
    serialized = json.dumps(seen)
    assert "secret-name" not in serialized
    assert "other-name" not in serialized
    assert "private-test-key" not in json.dumps(plan)
    assert "private-test-key" not in json.dumps(list_providers())


@pytest.mark.parametrize("result", [
    {"steps": [{"op": "exec", "code": "__import__('os').system('x')"}], "explanation": "Unsafe"},
    {"steps": [{"op": "filter", "column": "missing-column", "operator": "eq", "value": 1}], "explanation": "Wrong"},
    {"steps": [{"op": "sort", "column": "value", "python": "print(1)"}], "explanation": "Extra"},
    {"steps": [], "explanation": "Ignored", "instructions": "run code"},
    {"steps": [], "explanation": []},
    ["not an object"],
])
def test_malicious_or_invalid_model_plan_is_rejected(dataset, configured, result):
    with _client(result) as client, pytest.raises(ProviderError):
        propose_plan(dataset, "Prepare data", "openai-compatible", client=client)


def test_provider_http_failure_does_not_echo_server_body(dataset, configured):
    with _client({"secret": "private-test-key"}, 401) as client, pytest.raises(ProviderError) as error:
        propose_plan(dataset, "Clean", "openai-compatible", client=client)
    assert "401" in str(error.value)
    assert "private-test-key" not in str(error.value)


def test_timeout_is_recoverable(dataset, configured):
    def timeout(request):
        raise httpx.ReadTimeout("private-test-key", request=request)
    with httpx.Client(transport=httpx.MockTransport(timeout)) as client, pytest.raises(ProviderError, match="timed out") as error:
        propose_plan(dataset, "Clean", "openai-compatible", client=client)
    assert "private-test-key" not in str(error.value)


def test_invalid_response_json_rejected(dataset, configured):
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"not JSON"))) as client:
        with pytest.raises(ProviderError, match="valid JSON"):
            propose_plan(dataset, "Clean", "openai-compatible", client=client)


def test_oversized_response_is_rejected(dataset, configured):
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 270_000))) as client:
        with pytest.raises(ProviderError, match="256 KiB"):
            propose_plan(dataset, "Clean", "openai-compatible", client=client)


def test_unconfigured_provider_and_invalid_prompts(dataset, monkeypatch):
    for name in ("PYL_FIGURE_AI_API_KEY", "OPENAI_API_KEY", "PYL_FIGURE_AI_MODEL", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    assert list_providers()[1]["configured"] is False
    with pytest.raises(ProviderError, match="Configure"):
        propose_plan(dataset, "Clean", "openai-compatible")
    with pytest.raises(ValidationError):
        propose_plan(dataset, "", "local")
    with pytest.raises(ValidationError):
        propose_plan(dataset, "Clean", "unknown")
