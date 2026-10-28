"""Pydantic model and serialization contracts for Jitter Reports (Schema v0.1)."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from jitter.schema.runs import DiagnosisLabel, RunEntry

SCHEMA_VERSION = "0.1.0"


class ModelInfo(BaseModel):
    """Metadata describing the evaluated ML model."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Model file name or identifier")
    sha256: str = Field(
        ..., min_length=64, max_length=64, description="SHA256 hex digest of model artifact"
    )
    size_bytes: int = Field(..., ge=1, description="File size in bytes")


class DeviceInfo(BaseModel):
    """Specifications of the target hardware device."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., min_length=1, description="Device marketing model (e.g., Pixel 7a)")
    manufacturer: str = Field(..., min_length=1, description="Hardware manufacturer (e.g., Google)")
    soc: str | None = Field(default=None, description="System on Chip identifier (e.g., Tensor G2)")
    android_api: int | None = Field(
        default=None, ge=21, description="Android API level (SDK version)"
    )
    runner_version: str = Field(..., min_length=1, description="Version of the executing runner")


class BenchmarkConfig(BaseModel):
    """Execution configuration applied across reruns."""

    model_config = ConfigDict(extra="forbid")

    warmup_iterations: int = Field(
        ..., ge=0, description="Iterations before recording steady state"
    )
    measured_iterations: int = Field(..., ge=1, description="Iterations recorded per run")
    requested_delegate: str = Field(..., min_length=1, description="Primary requested delegate")
    input_spec: dict[str, Any] = Field(
        default_factory=dict, description="Input tensor shapes and types"
    )
    rest_policy: dict[str, Any] = Field(
        default_factory=lambda: {"cooldown_target_c": 32.0, "max_rest_s": 300},
        description="Conditions for thermal cooldown and stabilization",
    )


class EvidenceItem(BaseModel):
    """Single rule evaluation detail contributing to a diagnosis."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(..., min_length=1, description="Unique identifier of the rule")
    observed_value: float | str | int | bool = Field(
        ..., description="Value measured from run evidence"
    )
    threshold: float | str | int | bool = Field(..., description="Decision threshold applied")
    passed: bool = Field(..., description="Whether condition signature was satisfied")
    plain_language: str = Field(
        ..., min_length=1, description="Human-readable explanation of the evidence"
    )


class Diagnosis(BaseModel):
    """Root-cause classification with supporting evidence and confidence."""

    model_config = ConfigDict(extra="forbid")

    primary: DiagnosisLabel = Field(..., description="Primary identified cause")
    secondary: list[DiagnosisLabel] = Field(
        default_factory=list, description="Secondary or co-occurring causes"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    evidence: list[EvidenceItem] = Field(
        default_factory=list, description="Detailed rule evaluations"
    )
    inconclusive_reason: str | None = Field(
        default=None, description="Detailed explanation if verdict is inconclusive"
    )

    @model_validator(mode="after")
    def validate_inconclusive_consistency(self) -> "Diagnosis":
        if self.primary == DiagnosisLabel.INCONCLUSIVE and not self.inconclusive_reason:
            raise ValueError(
                "inconclusive_reason must be provided when primary diagnosis is 'inconclusive'"
            )
        return self


class RunStatistics(BaseModel):
    """Robust statistical summary of a single rerun."""

    model_config = ConfigDict(extra="forbid")

    n: int = Field(..., ge=1, description="Total sample count")
    n_eff: float = Field(
        ..., ge=0.0, description="Effective sample size accounting for autocorrelation"
    )
    median: float = Field(..., gt=0.0, description="Sample median in milliseconds")
    p95: float = Field(..., gt=0.0, description="95th percentile latency in milliseconds")
    p99: float = Field(..., gt=0.0, description="99th percentile latency in milliseconds")
    mad: float = Field(..., ge=0.0, description="Median absolute deviation scaled by 1.4826")
    ci_lower: float = Field(..., gt=0.0, description="Lower bound of median confidence interval")
    ci_upper: float = Field(..., gt=0.0, description="Upper bound of median confidence interval")
    method: str = Field(
        ..., min_length=1, description="Method used for confidence interval estimation"
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> "RunStatistics":
        if self.ci_lower > self.ci_upper:
            raise ValueError(f"ci_lower ({self.ci_lower}) cannot exceed ci_upper ({self.ci_upper})")
        return self


class PairwiseComparison(BaseModel):
    """Statistical hypothesis test comparing two benchmark reruns."""

    model_config = ConfigDict(extra="forbid")

    run_a: str = Field(..., min_length=1, description="Identifier of reference rerun")
    run_b: str = Field(..., min_length=1, description="Identifier of comparison rerun")
    diff_estimate: float = Field(
        ..., description="Difference or shift estimate (e.g. Hodges-Lehmann)"
    )
    p_value: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Hypothesis test p-value"
    )
    ci_lower: float | None = Field(default=None, description="Lower confidence bound on shift")
    ci_upper: float | None = Field(default=None, description="Upper confidence bound on shift")
    effect_size: float | None = Field(
        default=None, description="Effect size metric (e.g., Cliff's delta)"
    )
    effect_magnitude: str | None = Field(
        default=None, description="Qualitative magnitude: negligible, small, medium, large"
    )
    method: str = Field(..., min_length=1, description="Test name and methodology applied")


class ReportStatistics(BaseModel):
    """Aggregated statistics across individual runs and pairwise comparisons."""

    model_config = ConfigDict(extra="forbid")

    per_run: dict[str, RunStatistics] = Field(
        ..., description="Statistical summary mapped by run identifier"
    )
    comparisons: list[PairwiseComparison] = Field(
        default_factory=list, description="Pairwise statistical comparisons between runs"
    )


class SyntheticTruth(BaseModel):
    """Ground truth side-channel for synthetic fault injection; ignored by diagnosis."""

    model_config = ConfigDict(extra="allow")

    true_label: DiagnosisLabel
    injected_parameters: dict[str, Any] = Field(default_factory=dict)


class Report(BaseModel):
    """Canonical Jitter Diagnostic Report (v0.1)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default=SCHEMA_VERSION, description="Semver format of report schema"
    )
    jitter_version: str = Field(
        ..., min_length=1, description="Version of Jitter library that produced this report"
    )
    created_at: str = Field(..., description="UTC ISO 8601 creation timestamp")
    seed: int | None = Field(default=None, description="PRNG seed for reproducible generation")
    source: Literal["device", "desktop", "synthetic"] = Field(
        ..., description="Provenance origin of benchmark data"
    )
    model: ModelInfo
    device: DeviceInfo
    config: BenchmarkConfig
    runs: list[RunEntry] = Field(..., min_length=1, description="Ordered set of controlled reruns")
    diagnosis: Diagnosis
    statistics: ReportStatistics
    truth: SyntheticTruth | None = Field(
        default=None,
        description="Synthetic ground truth side-channel; strictly ignored by diagnosis engine",
    )

    @field_validator("created_at")
    @classmethod
    def validate_iso_timestamp(cls, v: str) -> str:
        try:
            # Must parse as valid ISO datetime
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                raise ValueError("created_at timestamp must have timezone information (UTC)")
        except Exception as e:
            raise ValueError(f"created_at must be valid ISO 8601 UTC timestamp: {e}") from e
        return v

    @classmethod
    def from_json(cls, json_str: str) -> "Report":
        """Parse Report from JSON string with descriptive field-level error messages."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as err:
            raise ValueError(f"Invalid JSON syntax in report: {err}") from err

        try:
            return cls.model_validate(data)
        except ValidationError as err:
            errors = []
            for e in err.errors():
                loc = ".".join(str(p) for p in e["loc"])
                msg = e["msg"]
                errors.append(f"Field '{loc}': {msg}")
            raise ValueError("Report validation failed:\n  " + "\n  ".join(errors)) from err

    @classmethod
    def from_file(cls, path: str | Path) -> "Report":
        """Load and validate a Report from a JSON file path."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Report file not found: {file_path}")
        return cls.from_json(file_path.read_text(encoding="utf-8"))

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to indented JSON string."""
        return self.model_dump_json(indent=indent, exclude_none=False)

    def to_file(self, path: str | Path, indent: int = 2) -> None:
        """Serialize report and write directly to file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(self.to_json(indent=indent), encoding="utf-8")
