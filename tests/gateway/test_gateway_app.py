from unittest import mock

import pytest
from fastapi.testclient import TestClient

from mlflow.exceptions import MlflowException
from mlflow.gateway.app import create_app_from_config, create_app_from_env
from mlflow.gateway.config import GatewayConfig
from mlflow.gateway.constants import (
    MLFLOW_GATEWAY_CRUD_ENDPOINT_V3_BASE,
    MLFLOW_GATEWAY_CRUD_ROUTE_BASE,
    MLFLOW_GATEWAY_CRUD_ROUTE_V3_BASE,
    MLFLOW_GATEWAY_ROUTE_BASE,
)

from tests.gateway.tools import MockAsyncResponse


@pytest.fixture
def client() -> TestClient:
    config = GatewayConfig(
        **{
            "endpoints": [
                {
                    "name": "completions-gpt4",
                    "endpoint_type": "llm/v1/completions",
                    "model": {
                        "name": "gpt-4",
                        "provider": "openai",
                        "config": {
                            "openai_api_key": "mykey",
                            "openai_api_base": "https://api.openai.com/v1",
                            "openai_api_version": "2023-05-10",
                            "openai_api_type": "openai",
                        },
                    },
                },
                {
                    "name": "chat-gpt4",
                    "endpoint_type": "llm/v1/chat",
                    "model": {
                        "name": "gpt-4",
                        "provider": "openai",
                        "config": {
                            "openai_api_key": "MY_API_KEY",
                        },
                    },
                },
                {
                    "name": "chat-gpt5",
                    "endpoint_type": "llm/v1/chat",
                    "model": {
                        "name": "gpt-5",
                        "provider": "openai",
                        "config": {
                            "openai_api_key": "MY_API_KEY",
                        },
                    },
                },
            ],
            "routes": [
                {
                    "name": "traffic_route1",
                    "task_type": "llm/v1/chat",
                    "destinations": [
                        {
                            "name": "chat-gpt4",
                            "traffic_percentage": 80,
                        },
                        {
                            "name": "chat-gpt5",
                            "traffic_percentage": 20,
                        },
                    ],
                },
            ],
        }
    )
    app = create_app_from_config(config)
    return TestClient(app)


def test_index(client: TestClient):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["Location"] == "/docs"


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_favicon(client: TestClient):
    response = client.get("/favicon.ico")
    assert response.status_code == 200


def test_docs(client: TestClient):
    response = client.get("/docs")
    assert response.status_code == 200


def test_search_routes(client: TestClient):
    response = client.get(MLFLOW_GATEWAY_CRUD_ROUTE_BASE)
    assert response.status_code == 200
    assert response.json()["routes"] == [
        {
            "name": "completions-gpt4",
            "route_type": "llm/v1/completions",
            "route_url": "/gateway/completions-gpt4/invocations",
            "model": {
                "name": "gpt-4",
                "provider": "openai",
            },
            "limit": None,
        },
        {
            "name": "chat-gpt4",
            "route_type": "llm/v1/chat",
            "route_url": "/gateway/chat-gpt4/invocations",
            "model": {
                "name": "gpt-4",
                "provider": "openai",
            },
            "limit": None,
        },
        {
            "name": "chat-gpt5",
            "route_type": "llm/v1/chat",
            "route_url": "/gateway/chat-gpt5/invocations",
            "model": {
                "name": "gpt-5",
                "provider": "openai",
            },
            "limit": None,
        },
    ]


def test_get_route(client: TestClient):
    response = client.get(f"{MLFLOW_GATEWAY_CRUD_ROUTE_BASE}chat-gpt4")
    assert response.status_code == 200
    assert response.json() == {
        "name": "chat-gpt4",
        "route_type": "llm/v1/chat",
        "route_url": "/gateway/chat-gpt4/invocations",
        "model": {
            "name": "gpt-4",
            "provider": "openai",
        },
        "limit": None,
    }


def test_get_endpoint_v3(client: TestClient):
    response = client.get(f"{MLFLOW_GATEWAY_CRUD_ENDPOINT_V3_BASE}chat-gpt4")
    assert response.status_code == 200
    assert response.json() == {
        "name": "chat-gpt4",
        "endpoint_type": "llm/v1/chat",
        "model": {"name": "gpt-4", "provider": "openai"},
        "endpoint_url": "/gateway/chat-gpt4/invocations",
        "limit": None,
    }


def test_get_route_v3(client: TestClient):
    response = client.get(f"{MLFLOW_GATEWAY_CRUD_ROUTE_V3_BASE}traffic_route1")
    assert response.status_code == 200
    assert response.json() == {
        "name": "traffic_route1",
        "task_type": "llm/v1/chat",
        "destinations": [
            {"name": "chat-gpt4", "traffic_percentage": 80},
            {"name": "chat-gpt5", "traffic_percentage": 20},
        ],
        "routing_strategy": "TRAFFIC_SPLIT",
    }


def test_dynamic_route():
    config = GatewayConfig(
        **{
            "endpoints": [
                {
                    "name": "chat",
                    "endpoint_type": "llm/v1/chat",
                    "model": {
                        "name": "gpt-4",
                        "provider": "openai",
                        "config": {
                            "openai_api_key": "mykey",
                            "openai_api_base": "https://api.openai.com/v1",
                        },
                    },
                    "limit": None,
                }
            ],
            "routes": [
                {
                    "name": "traffic_route",
                    "task_type": "llm/v1/chat",
                    "destinations": [
                        {
                            "name": "chat",
                            "traffic_percentage": 100,
                        }
                    ],
                }
            ],
        }
    )
    app = create_app_from_config(config)
    client = TestClient(app)

    resp = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4o-mini",
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20,
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "\n\nThis is a test!",
                    "refusal": None,
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "headers": {"Content-Type": "application/json"},
    }
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        for name in ["chat", "traffic_route"]:
            resp = client.post(
                f"{MLFLOW_GATEWAY_ROUTE_BASE}{name}/invocations",
                json={"messages": [{"role": "user", "content": "Tell me a joke"}]},
            )
            mock_post.assert_called_once()
            assert resp.status_code == 200
            assert resp.json() == {
                "id": "chatcmpl-abc123",
                "object": "chat.completion",
                "created": 1677858242,
                "model": "gpt-4o-mini",
                "usage": {
                    "prompt_tokens": 13,
                    "completion_tokens": 7,
                    "total_tokens": 20,
                },
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "\n\nThis is a test!",
                            "tool_calls": None,
                            "refusal": None,
                        },
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
            }

            mock_post.reset_mock()


def test_create_app_from_env_fails_if_MLFLOW_GATEWAY_CONFIG_is_not_set(monkeypatch):
    monkeypatch.delenv("MLFLOW_GATEWAY_CONFIG", raising=False)
    with pytest.raises(MlflowException, match="'MLFLOW_GATEWAY_CONFIG' is not set"):
        create_app_from_env()


@pytest.fixture
def client_for_endpoint_creation(tmp_path) -> TestClient:
    """Fixture for testing endpoint creation with a temporary config file."""

    config = GatewayConfig(
        **{
            "endpoints": [
                {
                    "name": "existing-endpoint",
                    "endpoint_type": "llm/v1/chat",
                    "model": {
                        "name": "gpt-4",
                        "provider": "openai",
                        "config": {
                            "openai_api_key": "test-key",
                        },
                    },
                }
            ]
        }
    )
    config_path = tmp_path / "test_config.yaml"
    app = create_app_from_config(config, config_path=str(config_path))
    return TestClient(app)


def test_create_endpoint_success(client_for_endpoint_creation: TestClient):
    from mlflow.deployments.server.constants import MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE

    endpoint_data = {
        "name": "new-endpoint",
        "endpoint_type": "llm/v1/chat",
        "model": {
            "name": "gpt-3.5-turbo",
            "provider": "openai",
            "config": {
                "openai_api_key": "test-key",
            },
        },
    }

    response = client_for_endpoint_creation.post(
        MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE, json=endpoint_data
    )

    assert response.status_code == 200
    result = response.json()
    assert result["name"] == "new-endpoint"
    assert result["endpoint_type"] == "llm/v1/chat"
    assert result["model"]["name"] == "gpt-3.5-turbo"
    assert result["model"]["provider"] == "openai"

    # Verify the endpoint was added and can be retrieved
    get_response = client_for_endpoint_creation.get(
        f"{MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE}new-endpoint"
    )
    assert get_response.status_code == 200
    assert get_response.json()["name"] == "new-endpoint"


def test_create_endpoint_duplicate_name_rejection(client_for_endpoint_creation: TestClient):
    from mlflow.deployments.server.constants import MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE

    endpoint_data = {
        "name": "existing-endpoint",  # This name already exists
        "endpoint_type": "llm/v1/chat",
        "model": {
            "name": "gpt-3.5-turbo",
            "provider": "openai",
            "config": {
                "openai_api_key": "test-key",
            },
        },
    }

    response = client_for_endpoint_creation.post(
        MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE, json=endpoint_data
    )

    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]
    assert "existing-endpoint" in response.json()["detail"]


def test_create_endpoint_invalid_git_location(client_for_endpoint_creation: TestClient):
    from mlflow.deployments.server.constants import MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE

    async def mock_validate_false(url):
        return False

    endpoint_data = {
        "name": "endpoint-with-git",
        "endpoint_type": "llm/v1/chat",
        "model": {
            "name": "custom-model",
            "provider": "openai",
            "git_location": "https://invalid-git-url.com/model.git",
            "config": {
                "openai_api_key": "test-key",
            },
        },
    }

    # Mock validate_git_location to return False (invalid URL)
    with mock.patch("mlflow.gateway.app.validate_git_location", side_effect=mock_validate_false):
        response = client_for_endpoint_creation.post(
            MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE, json=endpoint_data
        )

    assert response.status_code == 400
    assert "not a valid URL" in response.json()["detail"]
    assert "https://invalid-git-url.com/model.git" in response.json()["detail"]


def test_create_endpoint_missing_model_name(client_for_endpoint_creation: TestClient):
    from mlflow.deployments.server.constants import MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE

    endpoint_data = {
        "name": "endpoint-no-model-name",
        "endpoint_type": "llm/v1/chat",
        "model": {
            "name": None,  # Missing model name
            "provider": "openai",
            "config": {
                "openai_api_key": "test-key",
            },
        },
    }

    response = client_for_endpoint_creation.post(
        MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE, json=endpoint_data
    )

    assert response.status_code == 400
    assert "The model name must be provided" in response.json()["detail"]


def test_create_endpoint_config_persistence(tmp_path):
    from mlflow.deployments.server.constants import MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE
    from mlflow.gateway.config import _load_gateway_config

    config = GatewayConfig(
        **{
            "endpoints": [
                {
                    "name": "initial-endpoint",
                    "endpoint_type": "llm/v1/chat",
                    "model": {
                        "name": "gpt-4",
                        "provider": "openai",
                        "config": {
                            "openai_api_key": "test-key",
                        },
                    },
                }
            ]
        }
    )
    config_path = tmp_path / "test_config.yaml"
    app = create_app_from_config(config, config_path=str(config_path))
    client = TestClient(app)

    # Create a new endpoint
    endpoint_data = {
        "name": "new-persisted-endpoint",
        "endpoint_type": "llm/v1/completions",
        "model": {
            "name": "gpt-3.5-turbo",
            "provider": "openai",
            "config": {
                "openai_api_key": "test-key",
            },
        },
    }

    response = client.post(MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE, json=endpoint_data)
    assert response.status_code == 200

    # Verify the config file was updated
    assert config_path.exists()
    loaded_config = _load_gateway_config(str(config_path))

    # Should have both the initial endpoint and the new one
    assert len(loaded_config.endpoints) == 2
    endpoint_names = {ep.name for ep in loaded_config.endpoints}
    assert "initial-endpoint" in endpoint_names
    assert "new-persisted-endpoint" in endpoint_names

    # Verify the new endpoint details
    new_endpoint = next(ep for ep in loaded_config.endpoints if ep.name == "new-persisted-endpoint")
    assert new_endpoint.endpoint_type == "llm/v1/completions"
    assert new_endpoint.model.name == "gpt-3.5-turbo"
    assert new_endpoint.model.provider == "openai"


def test_create_endpoint_with_valid_git_location(client_for_endpoint_creation: TestClient):
    from mlflow.deployments.server.constants import MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE

    async def mock_validate_true(url):
        return True

    endpoint_data = {
        "name": "endpoint-with-valid-git",
        "endpoint_type": "llm/v1/chat",
        "model": {
            "name": "custom-model",
            "provider": "openai",
            "git_location": "https://github.com/user/model.git",
            "config": {
                "openai_api_key": "test-key",
            },
        },
    }

    # Mock validate_git_location to return True (valid URL)
    with mock.patch("mlflow.gateway.app.validate_git_location", side_effect=mock_validate_true):
        response = client_for_endpoint_creation.post(
            MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE, json=endpoint_data
        )

    assert response.status_code == 200
    result = response.json()
    assert result["name"] == "endpoint-with-valid-git"
    assert result["model"]["name"] == "custom-model"
    assert result["model"]["provider"] == "openai"
