"""Tests for health endpoint."""

import pytest


def test_api_response_success():
    """Test ApiResponse.success factory method."""
    from backend.domain.models import ApiResponse, ApiStatus

    response = ApiResponse.success(data={"key": "value"}, message="Test message")

    assert response.status == ApiStatus.SUCCESS
    assert response.data == {"key": "value"}
    assert response.message == "Test message"


def test_api_response_error():
    """Test ApiResponse.error factory method."""
    from backend.domain.models import ApiResponse, ApiStatus

    response = ApiResponse.error(message="Error occurred")

    assert response.status == ApiStatus.ERROR
    assert response.message == "Error occurred"
    assert response.data is None


def test_settings_defaults():
    """Test Settings default values."""
    from backend.config import Settings

    settings = Settings()

    assert settings.api_port == 8000
    assert settings.ollama_model == "llama3.1:8b"
    assert settings.chroma_port == 8100


def test_port_interfaces_are_abstract():
    """Test that port interfaces cannot be instantiated directly."""
    from backend.domain.ports import (
        DocumentParserPort,
        EmbeddingPort,
        LlmPort,
        VectorStorePort,
    )

    with pytest.raises(TypeError):
        LlmPort()

    with pytest.raises(TypeError):
        VectorStorePort()

    with pytest.raises(TypeError):
        EmbeddingPort()

    with pytest.raises(TypeError):
        DocumentParserPort()
