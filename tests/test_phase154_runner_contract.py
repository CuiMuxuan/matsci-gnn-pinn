from pathlib import Path


def test_phase154_runner_is_no_training_route_coverage_audit():
    script = Path("scripts/server/run_phase154_route_coverage_and_remaining_scheme_audit.sh").read_text(
        encoding="utf-8"
    )

    assert "build_phase154_route_coverage_and_remaining_scheme_audit.py" in script
    assert "phase154_route_coverage_and_remaining_scheme_audit_manifest.json" in script
    assert "route_coverage_phase154_gate" in script
    assert "currently_executable_model_routes_verified" in script
    assert "all_possible_future_schemes_exhausted" in script
    assert "new_model_claim_ready" in script
    assert "phase154_model_mechanism_allowed" in script
    assert "phase154_model_training_allowed" in script
    assert "operator_training_allowed_now" in script
    assert "a100_training_allowed_now" in script
    assert "a100_80gb_request_now" in script
    assert "data/raw/nist_ammt" not in script
    assert "data/raw/ambench" not in script
    assert "data/raw/external" not in script
    assert "train_macro_pinn" not in script
    assert "torchrun" not in script
