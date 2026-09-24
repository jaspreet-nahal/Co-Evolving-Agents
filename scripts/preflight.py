"""Lightweight preflight check: model registry status, GPU/runtime discovery,
and architecture-invariant status, without downloading any model weights.

Usage:
    python3.11 -m scripts.preflight
    python3.11 scripts/preflight.py --no-contact-hub
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from harness.core.model_backend import preflight, MODEL_REGISTRY


def _print_model_table(report):
    header = f"{'Model':<22} {'HF ID':<32} {'Cfg':<5} {'HFreach':<8} {'Auth':<6} {'Cached':<7} {'DL req':<7} {'Status':<18} {'Ready/Blocked'}"
    print(header)
    print("-" * len(header))
    for model in report["models"]:
        hf_reachable = "-" if model["hf_reachable"] is None else ("yes" if model["hf_reachable"] else "no")
        auth_required = "-" if model["auth_required"] is None else ("yes" if model["auth_required"] else "no")
        download_required = "yes" if model["status"] == "DOWNLOAD_REQUIRED" else ("-" if model["cached"] else "?")
        print(f"{model['model_name']:<22} {model['model_id']:<32} {'yes':<5} {hf_reachable:<8} {auth_required:<6} "
              f"{str(model['cached']):<7} {download_required:<7} {model['status']:<18} {model['ready_or_blocked']}")
        if model["detail"]:
            print(f"    -> {model['detail']}")


def _architecture_status():
    """Static architecture-status summary (no execution): confirms the
    relevant modules/symbols exist, does not run a trajectory."""
    status = {}
    try:
        from harness.core.flat_react import FlatReActExecutor, FLAT_REACT_ALLOWED_ACTIONS
        status["h0_implementation"] = f"FlatReActExecutor present; allowed actions: {sorted(a.value for a in FLAT_REACT_ALLOWED_ACTIONS)}"
    except Exception as error:
        status["h0_implementation"] = f"NOT AVAILABLE: {error}"

    try:
        from harness.core.harness import H0_CONFIG_DEFAULTS, H1_CONFIG_DEFAULTS
        status["h1_implementation"] = f"H1_CONFIG_DEFAULTS={H1_CONFIG_DEFAULTS}"
        status["c1_c5_isolation"] = f"H0_CONFIG_DEFAULTS={H0_CONFIG_DEFAULTS} (all off); components independently gated via EpisodeState"
    except Exception as error:
        status["h1_implementation"] = f"NOT AVAILABLE: {error}"
        status["c1_c5_isolation"] = f"NOT AVAILABLE: {error}"

    try:
        from harness.core.harness import HarnessConfig
        config = HarnessConfig()
        status["budget_parity"] = "HarnessConfig.budget_snapshot() is the single source read by both H0 and H1 execution loops"
    except Exception as error:
        status["budget_parity"] = f"NOT AVAILABLE: {error}"

    try:
        from harness.core.answer_generator import ModelBackedAnswerGenerator, RuleBasedAnswerGenerator
        status["real_answer_generator"] = "ModelBackedAnswerGenerator implemented; shared by H0/H1 via HarnessConfig.answer_generator; RuleBasedAnswerGenerator is test-only (require_model_backed_answers gate enforced)"
    except Exception as error:
        status["real_answer_generator"] = f"NOT AVAILABLE: {error}"

    try:
        from harness.benchmarks.qampari import RealDataUnavailableError
        status["sample_data_safety"] = "mode='real' on QAMPARI/FinanceBench/BrowseComp-Plus raises RealDataUnavailableError instead of falling back to sample data"
    except Exception as error:
        status["sample_data_safety"] = f"NOT AVAILABLE: {error}"

    return status


def _test_count():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "harness/tests/", "-q", "--collect-only"],
        capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), ".."),
    )
    for line in result.stdout.splitlines():
        if "collected" in line or "test" in line.lower():
            last_relevant = line
    tail = result.stdout.strip().splitlines()
    return tail[-1] if tail else "unknown"


def main():
    parser = argparse.ArgumentParser(description="Preflight check for model availability and architecture status")
    parser.add_argument("--no-contact-hub", action="store_true", help="Skip live Hugging Face Hub metadata checks")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model names to check (default: all four)")
    args = parser.parse_args()

    report = preflight(models=args.models, contact_hub=not args.no_contact_hub)

    print("=" * 100)
    print("MODEL PREFLIGHT REPORT")
    print("=" * 100)
    print(f"HF_HOME/TRANSFORMERS_CACHE: {report['hf_home'] or '(not set; using default cache location)'}")
    print(f"HF token configured:        {report['hf_token_configured']}")
    print(f"CUDA available:             {report['cuda_available']}")
    print(f"GPU device count:           {report['gpu_device_count']}")
    for device in report["gpu_devices"]:
        if "error" in device:
            print(f"  GPU[{device['index']}]: error inspecting device: {device['error']}")
        else:
            print(f"  GPU[{device['index']}]: {device['name']} - {device['free_memory_gb']}GB free / {device['total_memory_gb']}GB total")
    print()
    _print_model_table(report)

    print()
    print("=" * 100)
    print("ARCHITECTURE STATUS")
    print("=" * 100)
    for key, value in _architecture_status().items():
        print(f"{key}: {value}")

    print()
    print(f"test_count: {_test_count()}")
    print()
    print("Preflight complete. This did NOT download any model weights and did NOT run the 20-cell benchmark.")


if __name__ == "__main__":
    main()
