"""
Basic tests for the sidecar server API.
"""
import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def test_process_s3_unauthorized():
    """Test that the endpoint requires authentication."""
    response = client.post(
        "/process-s3",
        json={
            "object_key": "test.webm",
            "secret": "wrong-secret"
        }
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized"


def test_process_s3_missing_secret():
    """Test that the endpoint requires a secret."""
    response = client.post(
        "/process-s3",
        json={
            "object_key": "test.webm"
        }
    )
    assert response.status_code == 401


def test_process_s3_with_valid_secret_but_no_s3_access():
    """
    Test that the endpoint accepts the correct secret but fails on S3 access.
    This is expected since we don't have real S3 credentials in the test environment.
    """
    response = client.post(
        "/process-s3",
        json={
            "object_key": "test.webm",
            "secret": "change-me"  # Default secret from server.py
        }
    )
    # We expect a 500 error because S3 credentials are not configured
    assert response.status_code == 500
    assert "failed to download object" in response.json()["detail"]
