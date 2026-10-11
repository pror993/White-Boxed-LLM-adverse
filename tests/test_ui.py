"""Unit and integration tests for FastAPI research dashboard server and endpoints.

Verifies HTML serving, token decomposition, comparative autoregressive generation,
checkpoint persistence, sensitivity profiling, and optimization job lifecycles.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.ui.server import app, CHECKPOINTS_DIR


@pytest.fixture(scope="module")
def client():
    """TestClient fixture for FastAPI dashboard."""
    with TestClient(app) as test_client:
        yield test_client


class TestDashboardEndpoints:
    """Validate core HTTP endpoints for research dashboard."""

    def test_root_serves_html(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "GCG Discrete Coordinate Lab" in response.text

    def test_get_models(self, client: TestClient):
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        models = data["models"] if isinstance(data, dict) else data
        assert any("synthetic" in m for m in models)
        assert any("Llama-3.2" in m for m in models)

    def test_decompose_tokens(self, client: TestClient):
        payload = {
            "prompt": "Explain the fundamental theorem of calculus",
            "target": "The fundamental theorem connects differentiation and integration.",
            "optim_str": "! ! ! ! ! ! ! !",
            "model": "synthetic",
        }
        response = client.post("/api/decompose", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "tokens" in data
        assert "slices" in data
        opt_span = data["slices"]["opt"]
        assert (opt_span[1] - opt_span[0]) > 0
        target_span = data["slices"]["target"]
        assert (target_span[1] - target_span[0]) > 0
        assert len(data["tokens"]) == data["total_tokens"]

    def test_generate_comparative(self, client: TestClient):
        payload = {
            "prompt": "Explain the fundamental theorem of calculus",
            "suffix": "! ! ! !",
            "target": "differentiation",
            "model": "synthetic",
            "max_new_tokens": 15,
            "temperature": 0.0,
        }
        response = client.post("/api/generate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "baseline_output" in data
        assert "conditioned_output" in data
        assert "generation_time" in data
        assert isinstance(data["target_found_baseline"], bool)
        assert isinstance(data["target_found_conditioned"], bool)

    def test_checkpoints_lifecycle(self, client: TestClient):
        # 1. Save checkpoint
        save_payload = {
            "name": "test_calculus_checkpoint",
            "prompt": "Explain calculus",
            "target": "differentiation and integration",
            "suffix": "opt_tok_1 opt_tok_2",
            "model": "synthetic",
            "metrics": {
                "initial_loss": 3.5,
                "best_loss": 2.1,
                "steps": 10,
            },
        }
        save_res = client.post("/api/checkpoints/save", json=save_payload)
        assert save_res.status_code == 200
        save_data = save_res.json()
        assert save_data["status"] == "ok"
        checkpoint_id = save_data["checkpoint_id"]

        # 2. List checkpoints
        list_res = client.get("/api/checkpoints")
        assert list_res.status_code == 200
        checkpoints_data = list_res.json()
        checkpoints_list = checkpoints_data if isinstance(checkpoints_data, list) else checkpoints_data.get("checkpoints", [])
        assert any(cp.get("checkpoint_id") == checkpoint_id for cp in checkpoints_list)

        # 3. Load checkpoint
        load_res = client.post("/api/checkpoints/load", json={"checkpoint_id": checkpoint_id})
        assert load_res.status_code == 200
        loaded_data = load_res.json()
        assert loaded_data["suffix"] == "opt_tok_1 opt_tok_2"
        assert loaded_data["metrics"]["best_loss"] == 2.1

        # Cleanup test file
        target_file = CHECKPOINTS_DIR / f"{checkpoint_id}.json"
        if target_file.exists():
            target_file.unlink()

    def test_sensitivity_endpoint(self, client: TestClient):
        payload = {
            "prompt": "Explain calculus",
            "target": "differentiation",
            "suffix": "! ! ! !",
            "model": "synthetic",
            "noise_levels": [0.05, 0.10],
            "samples_per_level": 2,
        }
        response = client.post("/api/sensitivity", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "sharpness_index" in data
        assert "evaluations" in data
        assert len(data["evaluations"]) == 2

    def test_start_optimization_job(self, client: TestClient):
        payload = {
            "prompt": "Explain calculus",
            "target": "differentiation",
            "optim_init": "! ! ! !",
            "model": "synthetic",
            "num_steps": 2,
            "batch_size": 8,
            "top_k": 4,
            "micro_batch_size": 4,
            "objective_type": "cross_entropy",
        }
        response = client.post("/api/optimize", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["job_id"].startswith("job_")
