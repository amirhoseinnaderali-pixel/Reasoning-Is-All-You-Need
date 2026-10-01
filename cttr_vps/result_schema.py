from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = "1.0"


@dataclass
class ExperimentRecord:
    experiment_id: str
    problem_id: str
    method: str
    models: list[str]
    num_generation_attempts: int
    num_refinement_rounds: int
    candidate_count: int
    compiled: bool
    tests_passed: int
    tests_failed: int
    solved: bool
    latency_seconds: float | None
    model_calls: int
    verification_attempts: int
    tokens: int | None
    error_type: str | None
    started_at: str
    finished_at: str
    schema_version: str = SCHEMA_VERSION
    status: str = "completed"
    notes: list[str] | None = None
    stage_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_record(
    experiment_id: str,
    problem_id: str,
    method: str,
    *,
    models: list[str] | None = None,
    num_generation_attempts: int = 0,
    num_refinement_rounds: int = 0,
    candidate_count: int = 0,
    compiled: bool = False,
    tests_passed: int = 0,
    tests_failed: int = 0,
    solved: bool = False,
    latency_seconds: float | None = None,
    model_calls: int = 0,
    verification_attempts: int = 0,
    tokens: int | None = None,
    error_type: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    status: str = "completed",
    notes: list[str] | None = None,
    stage_metrics: dict[str, Any] | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=experiment_id,
        problem_id=problem_id,
        method=method,
        models=models or [],
        num_generation_attempts=num_generation_attempts,
        num_refinement_rounds=num_refinement_rounds,
        candidate_count=candidate_count,
        compiled=compiled,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        solved=solved,
        latency_seconds=latency_seconds,
        model_calls=model_calls,
        verification_attempts=verification_attempts,
        tokens=tokens,
        error_type=error_type,
        started_at=started_at or utc_now(),
        finished_at=finished_at or utc_now(),
        status=status,
        notes=notes or [],
        stage_metrics=stage_metrics or {},
    )
