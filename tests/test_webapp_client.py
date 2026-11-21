"""Tests of the web app logic (client + file parsing). Streamlit itself is not started."""

import io
import sys
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "webapp"))
import client as webclient  # noqa: E402

from fraud_detection.api import dummy_model, main  # noqa: E402


@pytest.fixture
def api(tmp_path, monkeypatch):
    import mlflow

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    dummy_model.register()
    monkeypatch.delenv("MODEL_URI", raising=False)
    with TestClient(main.app) as c:
        # route the `requests` calls of the web client to the in-process API
        monkeypatch.setattr(requests, "post", lambda url, **kw: c.post(url.removeprefix("http://t"), **_kw(kw)))
        monkeypatch.setattr(requests, "get", lambda url, **kw: c.get(url.removeprefix("http://t")))
        yield "http://t"
    mlflow.set_tracking_uri(None)


def _kw(kw):
    return {k: v for k, v in kw.items() if k == "json"}


def test_demo_ride_has_the_six_channels():
    df = webclient.demo_ride(200)
    assert list(df.columns) == webclient.CHANNELS and len(df) == 200


def test_read_ride_accepts_csv_and_rejects_missing_columns():
    csv = webclient.demo_ride(150).to_csv(index=False).encode()
    assert len(webclient.read_ride("ride.csv", csv)) == 150
    with pytest.raises(ValueError, match="Missing column"):
        webclient.read_ride("bad.csv", b"ax,ay\n1,2\n")


def test_read_ride_accepts_parquet():
    buf = io.BytesIO()
    webclient.demo_ride(120).to_parquet(buf)
    assert len(webclient.read_ride("ride.parquet", buf.getvalue())) == 120


def test_to_payload_sorts_by_time_and_turns_nan_into_null():
    df = webclient.demo_ride(100)
    df["time_index"] = range(99, -1, -1)  # reversed
    df.loc[df.index[-1], "ax"] = float("nan")  # last row = first in time
    payload = webclient.to_payload(df, "r")
    assert payload["samples"][0]["ax"] is None and payload["ride_id"] == "r"


def test_predict_through_the_api(api):
    out = webclient.predict(api, webclient.to_payload(webclient.demo_ride(300), "demo"))
    assert out["predicted_class"] in out["probabilities"] and out["ride_id"] == "demo"


def test_predict_raises_readable_error_on_api_rejection(api):
    with pytest.raises(webclient.ApiError, match="422"):
        webclient.predict(api, {"ride_id": "x", "samples": []})


def test_feedback_and_health(api):
    assert webclient.health(api)["status"] == "ok"
    webclient.send_feedback(api, "Safe", "Tandem")
    with pytest.raises(webclient.ApiError):
        webclient.send_feedback(api, "Safe", "Banana")


def test_unreachable_api_is_reported(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(webclient.ApiError, match="unreachable"):
        webclient.predict("http://nowhere", {"samples": []})
