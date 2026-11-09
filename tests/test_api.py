"""API and predictor checks (run:  pytest)."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.main import app  # noqa: E402

client = TestClient(app)


@pytest.mark.parametrize("url", ["/api/health", "/api/overview", "/api/km", "/api/effects", "/api/models",
                                 "/api/heterogeneity", "/api/figures", "/api/config", "/"])
def test_endpoints_respond(url):
    assert client.get(url).status_code == 200


def test_actg_prediction_is_valid_and_curves_decrease():
    r = client.post("/api/predict/actg175", json={}).json()
    assert set(r["arms"]) == {"ZDV", "ZDV+ddI", "ZDV+zal", "ddI"}
    for a in r["arms"].values():
        assert 0 <= a["event_free_prob"] <= 1
    for c in r["curves"].values():
        assert c[0] == 1.0 and all(x >= y for x, y in zip(c, c[1:]))
    # the model reproduces the main finding: combination arms beat ZDV alone
    assert all(r["arms"][k]["event_free_prob"] > r["arms"]["ZDV"]["event_free_prob"] for k in ("ZDV+ddI", "ZDV+zal", "ddI"))


def test_worse_cd4_lowers_predicted_survival():
    hi = client.post("/api/predict/actg175", json={"cd40": 600}).json()["arms"]["ZDV"]["event_free_prob"]
    lo = client.post("/api/predict/actg175", json={"cd40": 100}).json()["arms"]["ZDV"]["event_free_prob"]
    assert lo < hi


def test_gbsg_prediction_therapy_helps_and_more_nodes_hurt():
    a = client.post("/api/predict/gbsg2", json={}).json()["arms"]
    assert a["Hormonal therapy"]["event_free_prob"] > a["No hormonal therapy"]["event_free_prob"]
    few = client.post("/api/predict/gbsg2", json={"pnodes": 1}).json()["arms"]["No hormonal therapy"]["event_free_prob"]
    many = client.post("/api/predict/gbsg2", json={"pnodes": 20}).json()["arms"]["No hormonal therapy"]["event_free_prob"]
    assert many < few


def test_invalid_input_is_rejected():
    assert client.post("/api/predict/actg175", json={"karnof": 85}).status_code == 422
    assert client.post("/api/predict/actg175", json={"age": 500}).status_code == 422
    assert client.post("/api/predict/gbsg2", json={"grade": 7}).status_code == 422
