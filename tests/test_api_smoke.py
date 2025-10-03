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


def test_crud_students(client):
	# create
	r = client.post("/students", json={"name": "Alice", "email": "alice@example.com"})
	assert r.status_code == 200
	id_ = r.json()["id"]

	# read
	r = client.get(f"/students/{id_}")
	assert r.status_code == 200
	assert r.json()["name"] == "Alice"

	# update
	r = client.put(f"/students/{id_}", json={"name": "Alice Doe", "email": "alice@example.com"})
	assert r.status_code == 200 and r.json()["success"] is True

	# list
	r = client.get("/students")
	assert r.status_code == 200
	assert isinstance(r.json().get("items"), list)

	# delete
	r = client.delete(f"/students/{id_}")
	assert r.status_code == 200 and r.json()["success"] is True


def test_crud_events(client):
	# create
	r = client.post("/events", json={"title": "Hackathon", "description": "College event", "date": "2025-10-10"})
	assert r.status_code == 200
	id_ = r.json()["id"]

	# read
	r = client.get(f"/events/{id_}")
	assert r.status_code == 200
	assert r.json()["title"] == "Hackathon"

	# update
	r = client.put(f"/events/{id_}", json={"title": "Hackathon 2025", "description": "College event", "date": "2025-10-10"})
	assert r.status_code == 200 and r.json()["success"] is True

	# list
	r = client.get("/events")
	assert r.status_code == 200
	assert isinstance(r.json().get("items"), list)

	# delete
	r = client.delete(f"/events/{id_}")
	assert r.status_code == 200 and r.json()["success"] is True


def test_requirements_api(client):
	# create
	r = client.post("/requirements", json={"title": "Events CRUD", "content": {"entities": ["event"], "ac": ["create", "list"]}})
	assert r.status_code == 200
	req_id = r.json()["id"]

	# read
	r = client.get(f"/requirements/{req_id}")
	assert r.status_code == 200
	assert r.json()["title"] == "Events CRUD"

	# update
	r = client.put(f"/requirements/{req_id}", json={"title": "Events CRUD v2", "content": {"entities": ["event"], "ac": ["create", "list", "update"]}})
	assert r.status_code == 200 and r.json()["success"] is True

	# list
	r = client.get("/requirements")
	assert r.status_code == 200
	assert isinstance(r.json().get("items"), list)

	# delete
	r = client.delete(f"/requirements/{req_id}")
	assert r.status_code == 200 and r.json()["success"] is True


