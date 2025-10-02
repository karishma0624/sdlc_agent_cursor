import os
import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
	return TestClient(app)


def test_health(client):
	r = client.get("/health")
	assert r.status_code == 200
	assert r.json().get("status") == "ok"


