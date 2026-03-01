#!/usr/bin/env python3
"""Iterative synthetic-data generation loop with automatic validation feedback."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.generate import build_parser as build_generate_parser
from scripts.run_vectorized_pipeline import run_vectorized_pipeline

ARTIFACT_SUFFIX = ".json"


def build_loop_parser() -> argparse.ArgumentParser:
    parser = build_parse_with_defaults()
    return parser


def build_parse_with_defaults() -> argparse.ArgumentParser:
    parser = build_generate_parser()
    parser.prog = "scripts/iterative_synthetic_quality_loop.py"
    parser.description = (
        "Automate generate -> validate -> adjust -> regenerate for synthetic data quality targets."
    )

    parser.add_argument(
        "--workspace",
        default="outputs/iterative_quality_loop",
        help="Directory for experiment artifacts",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum optimization iterations (default: 10)",
    )
    parser.add_argument(
        "--target-overall-score",
        type=float,
        default=0.80,
        help="Minimum overall quality score target (default: 0.80)",
    )
    parser.add_argument(
        "--target-clinical-score",
        type=float,
        default=0.90,
        help="Minimum clinical score target (default: 0.90)",
    )
    parser.add_argument(
        "--target-statistical-score",
        type=float,
        default=0.80,
        help="Minimum statistical score target (default: 0.80)",
    )
    parser.add_argument(
        "--target-temporal-score",
        type=float,
        default=0.75,
        help="Minimum temporal consistency score target (default: 0.75)",
    )
    parser.add_argument(
        "--target-correlation-score",
        type=float,
        default=0.75,
        help="Minimum correlation score target (default: 0.75)",
    )
    parser.add_argument(
        "--target-capacity-score",
        type=float,
        default=0.85,
        help="Minimum capacity score target (default: 0.85)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=20240101,
        help="Random seed seed for the first attempt when not provided via --random-seed (default: 20240101)",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=17,
        help="Seed increment used when re-running with a new sample (default: 17)",
    )
    parser.add_argument(
        "--min-total-records",
        type=int,
        default=100000,
        help="Lower bound for total records when iteratively increasing sample size",
    )
    parser.add_argument(
        "--max-total-records",
        type=int,
        default=600000,
        help="Upper bound for total records during iterative search",
    )
    parser.add_argument(
        "--record-step",
        type=int,
        default=25000,
        help="Increment to add when increasing sample size (default: 25000)",
    )
    parser.add_argument(
        "--record-growth-factor",
        type=float,
        default=1.2,
        help="Multiplicative growth factor when statistical score is unstable (default: 1.2)",
    )
    parser.add_argument(
        "--conditional-context-min-count-step",
        type=int,
        default=8,
        help="How much to reduce conditional context minimum count when correlation is low (default: 8)",
    )
    parser.add_argument(
        "--conditional-global-mix-step",
        type=float,
        default=-0.05,
        help="Adjust conditional/global mix weight toward conditional (default: -0.05)",
    )
    parser.add_argument(
        "--conditional-smoothing-step",
        type=float,
        default=-0.005,
        help="Adjust conditional smoothing (default: -0.005)",
    )
    parser.add_argument(
        "--capacity-multiplier-step",
        type=float,
        default=1.12,
        help="Multiplier increase for base_capacity_multiplier when capacity score is below target",
    )
    parser.add_argument(
        "--capacity-margin-step",
        type=float,
        default=0.08,
        help="Safety margin increase when capacity score is below target",
    )
    parser.add_argument(
        "--capacity-ratio-step",
        type=float,
        default=0.05,
        help="Weekend/holiday multiplier increase when capacity score is below target",
    )
    parser.add_argument(
        "--max-capacity-multiplier",
        type=float,
        default=2.5,
        help="Ceiling for base_capacity_multiplier search",
    )
    parser.add_argument(
        "--max-capacity-margin",
        type=float,
        default=2.0,
        help="Ceiling for safety_margin search",
    )
    return parser


def _to_json(value: Any) -> Any:
    if isinstance(value, (Path, datetime)):
        return value.isoformat()
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (dict, list, tuple, set)):
        try:
            if isinstance(value, set):
                return [_to_json(v) for v in sorted(value)]
            if isinstance(value, (list, tuple)):
                return [_to_json(v) for v in value]
            if isinstance(value, dict):
                return {str(k): _to_json(v) for k, v in value.items()}
        except Exception:
            return str(value)
    return str(value)


def _serialize_payload(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = {k: _to_json(v) for k, v in payload.items()}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(safe_payload, handle, indent=2, ensure_ascii=False)


def _load_quality(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("quality"), dict):
        return payload["quality"]
    return payload if isinstance(payload, dict) else None


def _extract_scores(quality: Dict[str, Any]) -> Dict[str, Optional[float]]:
    component_scores = quality.get("component_scores") or quality.get("summary", {}).get("components", {})
    return {
        "overall": _to_float(quality.get("overall_score") or quality.get("summary", {}).get("overall_score")),
        "clinical": _to_float(component_scores.get("clinical")),
        "statistical": _to_float(component_scores.get("statistical")),
        "temporal": _to_float(component_scores.get("temporal")),
        "correlation": _to_float(component_scores.get("correlation")),
        "capacity": _to_float(component_scores.get("capacity")),
    }


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _is_below(score: Optional[float], target: float) -> bool:
    if score is None:
        return False
    return score < target


def _build_attempt_args(base_args: argparse.Namespace, state: Dict[str, Any], attempt_idx: int, workspace: Path) -> argparse.Namespace:
    attempt_dir = workspace / f"attempt_{attempt_idx:02d}"
    attempt_db = attempt_dir / "synthetic.duckdb"
    quality_path = attempt_dir / "quality_report.json"
    run_args = argparse.Namespace(**vars(base_args))
    run_args.database = str(attempt_db)
    run_args.quality_report_path = str(quality_path)
    run_args.random_seed = state["random_seed"]
    run_args.total_records = state["total_records"]
    run_args.time_resolution = state["time_resolution"]
    run_args.preserve_seasonality = state["preserve_seasonality"]
    run_args.preserve_weekly_pattern = state["preserve_weekly_pattern"]
    run_args.preserve_holiday_effects = state["preserve_holiday_effects"]
    run_args.enable_overflow_redistribution = state["enable_overflow_redistribution"]
    run_args.base_capacity_multiplier = state["base_capacity_multiplier"]
    run_args.weekend_capacity_multiplier = state["weekend_capacity_multiplier"]
    run_args.holiday_capacity_multiplier = state["holiday_capacity_multiplier"]
    run_args.safety_margin = state["safety_margin"]
    run_args.overflow_redistribution_method = state["overflow_redistribution_method"]
    run_args.disable_conditional_hour_patterns = not state["enable_conditional_hour_patterns"]
    run_args.conditional_context_min_count = state["conditional_context_min_count"]
    run_args.conditional_global_mix_weight = state["conditional_global_mix_weight"]
    run_args.conditional_smoothing_alpha = state["conditional_smoothing_alpha"]
    run_args.clean_existing_data = True
    run_args.quality_gate_threshold = 0.0
    run_args.config_path = base_args.config_path
    return run_args


def _targets_met(scores: Dict[str, Optional[float]], args: argparse.Namespace) -> bool:
    checks = [
        (scores.get("overall"), args.target_overall_score),
        (scores.get("clinical"), args.target_clinical_score),
        (scores.get("statistical"), args.target_statistical_score),
        (scores.get("temporal"), args.target_temporal_score),
        (scores.get("correlation"), args.target_correlation_score),
    ]
    if args.target_capacity_score > 0:
        checks.append((scores.get("capacity"), args.target_capacity_score))

    for score, target in checks:
        if score is None:
            continue
        if score < target:
            return False
    return True


def _propose_next_state(
    state: Dict[str, Any],
    quality: Optional[Dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[Dict[str, Any], List[str]]:
    updated = dict(state)
    actions: List[str] = []
    scores = _extract_scores(quality or {})

    if quality is None:
        updated["random_seed"] += args.seed_step
        return updated, ["retry with new seed (pipeline error)"]

    if _is_below(scores.get("clinical"), args.target_clinical_score):
        updated["random_seed"] += args.seed_step
        actions.append("rerun with new seed for clinical stability")

    if _is_below(scores.get("statistical"), args.target_statistical_score):
        if not args.use_original_size:
            next_records = int(max(updated["total_records"] + args.record_step, updated["total_records"] * args.record_growth_factor))
            next_records = max(args.min_total_records, next_records)
            if next_records != updated["total_records"]:
                updated["total_records"] = min(args.max_total_records, next_records)
                actions.append(f"increase total_records to {updated['total_records']}")
        updated["random_seed"] += args.seed_step
        actions.append("rerun with new seed for statistical stability")

    if _is_below(scores.get("temporal"), args.target_temporal_score):
        updated["preserve_seasonality"] = True
        updated["preserve_weekly_pattern"] = True
        updated["preserve_holiday_effects"] = True
        updated["time_resolution"] = "hourly"
        updated["random_seed"] += args.seed_step
        actions.append("strengthen temporal constraints and rerun")

    if _is_below(scores.get("correlation"), args.target_correlation_score):
        updated["enable_conditional_hour_patterns"] = True
        updated["conditional_context_min_count"] = max(
            5,
            updated["conditional_context_min_count"] - args.conditional_context_min_count_step,
        )
        updated["conditional_global_mix_weight"] = max(
            0.0,
            min(
                1.0,
                updated["conditional_global_mix_weight"] + args.conditional_global_mix_step,
            ),
        )
        updated["conditional_smoothing_alpha"] = max(
            0.0,
            min(
                1.0,
                updated["conditional_smoothing_alpha"] + args.conditional_smoothing_step,
            ),
        )
        updated["random_seed"] += args.seed_step
        actions.append("tighten conditional-hour settings for correlation")

    if args.target_capacity_score > 0 and _is_below(scores.get("capacity"), args.target_capacity_score):
        if not updated["enable_overflow_redistribution"]:
            updated["enable_overflow_redistribution"] = True
            actions.append("enable overflow redistribution")

        updated["base_capacity_multiplier"] = min(
            args.max_capacity_multiplier,
            updated["base_capacity_multiplier"] * args.capacity_multiplier_step,
        )
        updated["weekend_capacity_multiplier"] = min(
            1.0,
            updated["weekend_capacity_multiplier"] + args.capacity_ratio_step,
        )
        updated["holiday_capacity_multiplier"] = min(
            1.0,
            updated["holiday_capacity_multiplier"] + args.capacity_ratio_step,
        )
        updated["safety_margin"] = min(
            args.max_capacity_margin,
            updated["safety_margin"] + args.capacity_margin_step,
        )
        if updated["base_capacity_multiplier"] == args.max_capacity_multiplier and updated["safety_margin"] == args.max_capacity_margin:
            updated["overflow_redistribution_method"] = "second_choice_probability"
        actions.append("loosen capacity constraints and rerun")

    if not actions:
        updated["random_seed"] += args.seed_step
        actions.append("fallback reseed")

    # Avoid repeated no-op loops by forcing at least one change.
    if updated == state:
        updated["random_seed"] += args.seed_step
        actions.append("force reseed because no effective change")

    return updated, actions


def _print_progress(iteration: int, status: str, scores: Dict[str, Optional[float]], actions: List[str], attempt_dir: Path):
    score_text = ", ".join(f"{k}={v:.4f}" if v is not None else f"{k}=N/A" for k, v in scores.items())
    message = (
        f"[{iteration:02d}] status={status} | scores: {score_text} | "
        f"actions: {', '.join(actions) if actions else 'none'} | dir={attempt_dir}"
    )
    print(message)


def _initial_state(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "random_seed": args.random_seed if args.random_seed is not None else args.seed_start,
        "total_records": max(args.total_records, args.min_total_records),
        "time_resolution": args.time_resolution,
        "preserve_seasonality": args.preserve_seasonality,
        "preserve_weekly_pattern": args.preserve_weekly_pattern,
        "preserve_holiday_effects": args.preserve_holiday_effects,
        "enable_overflow_redistribution": args.enable_overflow_redistribution,
        "base_capacity_multiplier": args.base_capacity_multiplier,
        "weekend_capacity_multiplier": args.weekend_capacity_multiplier,
        "holiday_capacity_multiplier": args.holiday_capacity_multiplier,
        "safety_margin": args.safety_margin,
        "overflow_redistribution_method": args.overflow_redistribution_method,
        "enable_conditional_hour_patterns": not args.disable_conditional_hour_patterns,
        "conditional_context_min_count": args.conditional_context_min_count,
        "conditional_global_mix_weight": args.conditional_global_mix_weight,
        "conditional_smoothing_alpha": args.conditional_smoothing_alpha,
    }


def main() -> int:
    parser = build_loop_parser()
    args = parser.parse_args()

    workspace = Path(args.workspace)
    if not workspace.is_absolute():
        workspace = PROJECT_ROOT / workspace
    workspace.mkdir(parents=True, exist_ok=True)

    state = _initial_state(args)
    history: List[Dict[str, Any]] = []
    best = {"overall": -1.0, "iteration": None, "quality": None}
    stop_reason = "max_iterations_reached"

    for iteration in range(1, args.max_iterations + 1):
        attempt_dir = workspace / f"attempt_{iteration:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        quality_path = attempt_dir / "quality_report.json"

        run_args = _build_attempt_args(args, state, iteration, workspace)
        run_args.quality_report_path = str(quality_path)

        _serialize_payload(vars(run_args), attempt_dir / "run_args.json")
        ok = run_vectorized_pipeline(run_args)

        quality = _load_quality(quality_path)
        scores = _extract_scores(quality or {})
        if quality is None:
            status = "failed_no_report"
        elif _targets_met(scores, args):
            status = "passed"
        else:
            status = "reported"

        if quality is not None:
            record = {
                "iteration": iteration,
                "attempt_dir": str(attempt_dir),
                "status": status,
                "state": state,
                "scores": scores,
                "quality": quality,
                "pipeline_ok": ok,
                "started_at": datetime.utcnow().isoformat() + "Z",
            }
        else:
            record = {
                "iteration": iteration,
                "attempt_dir": str(attempt_dir),
                "status": "failed_no_report",
                "state": state,
                "scores": scores,
                "pipeline_ok": ok,
                "started_at": datetime.utcnow().isoformat() + "Z",
            }

        history.append(record)
        overall = scores.get("overall")
        if overall is not None and overall > best["overall"]:
            best["overall"] = overall
            best["iteration"] = iteration
            best["quality"] = quality
            best["state"] = state

        actions: List[str] = []
        _serialize_payload(record, attempt_dir / f"attempt_summary{ARTIFACT_SUFFIX}")

        if status == "passed":
            _print_progress(iteration, status, scores, [], attempt_dir)
            stop_reason = "targets_met"
            break

        state, actions = _propose_next_state(state, quality, args)
        _serialize_payload({"next_state": state, "actions": actions}, attempt_dir / f"next_state{ARTIFACT_SUFFIX}")
        _print_progress(iteration, status, scores, actions, attempt_dir)

        if iteration < args.max_iterations:
            continue

    if status != "passed":
        _print_progress(iteration, status, scores, actions, attempt_dir)

    summary = {
        "workspace": str(workspace),
        "iterations": len(history),
        "stop_reason": stop_reason,
        "target_overall_score": args.target_overall_score,
        "target_clinical_score": args.target_clinical_score,
        "target_statistical_score": args.target_statistical_score,
        "target_temporal_score": args.target_temporal_score,
        "target_correlation_score": args.target_correlation_score,
        "target_capacity_score": args.target_capacity_score,
        "best": best,
        "history": [
            {"iteration": h.get("iteration"), "attempt_dir": h.get("attempt_dir"), "status": h.get("status"), "scores": h.get("scores")}
            for h in history
        ],
    }
    _serialize_payload(summary, workspace / "loop_summary.json")

    print(f"\nLoop finished: {stop_reason}")
    print(f"Iterations: {len(history)} | Best overall: {best['overall']:.4f}")
    if best["iteration"] is not None:
        print(f"Best iteration: {best['iteration']:02d}")
    return 0 if stop_reason == "targets_met" else 1


if __name__ == "__main__":
    sys.exit(main())
