from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction():
    response = client.post(
        "/predict",
        json={"age": 35, "income": 50000, "tenure": 4},
    )
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "confidence" in body
