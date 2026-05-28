"""
File: readiness.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Real model readiness checks for the PULSE inference pipeline.
License: MIT License
"""

from dataclasses import dataclass
import importlib
from types import ModuleType
from typing import Final

from pulse.inference.models import build_multitask_mamba_model
from pulse.inference.runtime import (
    build_mtl_config,
    get_checkpoint_path,
    get_model_device,
    is_real_model_enabled,
)

ERROR_SEVERITY: Final = "error"
WARNING_SEVERITY: Final = "warning"
INFO_SEVERITY: Final = "info"


@dataclass(frozen=True, slots=True)
class ReadinessIssue:
    """One real model readiness issue."""

    severity: str
    message: str


@dataclass(frozen=True, slots=True)
class RealModelReadiness:
    """Real model readiness check result."""

    issues: list[ReadinessIssue]

    @property
    def errors(self) -> list[ReadinessIssue]:
        """Return readiness errors."""

        return [issue for issue in self.issues if issue.severity == ERROR_SEVERITY]

    @property
    def warnings(self) -> list[ReadinessIssue]:
        """Return readiness warnings."""

        return [issue for issue in self.issues if issue.severity == WARNING_SEVERITY]

    @property
    def is_ready(self) -> bool:
        """Return whether the real model can be enabled."""

        return not self.errors


def create_issue(severity: str, message: str) -> ReadinessIssue:
    """Create a readiness issue."""

    return ReadinessIssue(
        severity=severity,
        message=message,
    )


def check_mamba_import() -> list[ReadinessIssue]:
    """Check whether mamba_ssm can be imported."""

    try:
        module = importlib.import_module("mamba_ssm")
    except Exception as error:
        return [
            create_issue(
                ERROR_SEVERITY,
                f"mamba_ssm import failed: {error}",
            ),
        ]

    if not isinstance(module, ModuleType):
        return [
            create_issue(
                ERROR_SEVERITY,
                "mamba_ssm import did not return a module.",
            ),
        ]

    if not hasattr(module, "Mamba"):
        return [
            create_issue(
                ERROR_SEVERITY,
                "mamba_ssm module does not expose Mamba.",
            ),
        ]

    return [
        create_issue(
            INFO_SEVERITY,
            "mamba_ssm import passed.",
        ),
    ]


def check_checkpoint_path() -> list[ReadinessIssue]:
    """Check whether the configured checkpoint exists."""

    checkpoint_path = get_checkpoint_path()

    if not checkpoint_path.exists():
        return [
            create_issue(
                ERROR_SEVERITY,
                f"Checkpoint was not found: {checkpoint_path}",
            ),
        ]

    if not checkpoint_path.is_file():
        return [
            create_issue(
                ERROR_SEVERITY,
                f"Checkpoint path is not a file: {checkpoint_path}",
            ),
        ]

    return [
        create_issue(
            INFO_SEVERITY,
            f"Checkpoint exists: {checkpoint_path}",
        ),
    ]


def check_model_config() -> list[ReadinessIssue]:
    """Check whether model config values are valid."""

    config = build_mtl_config()
    issues: list[ReadinessIssue] = []

    if config.in_dim <= 0:
        issues.append(create_issue(ERROR_SEVERITY, "Model_INPUT_DIM must be positive."))

    if config.d_model <= 0:
        issues.append(create_issue(ERROR_SEVERITY, "Model_D_MODEL must be positive."))

    if config.num_layers <= 0:
        issues.append(create_issue(ERROR_SEVERITY, "Model_NUM_LAYERS must be positive."))

    if config.mamba_d_state <= 0:
        issues.append(create_issue(ERROR_SEVERITY, "Model_MAMBA_D_STATE must be positive."))

    if config.mamba_d_conv <= 0:
        issues.append(create_issue(ERROR_SEVERITY, "Model_MAMBA_D_CONV must be positive."))

    if config.mamba_expand <= 0:
        issues.append(create_issue(ERROR_SEVERITY, "Model_MAMBA_EXPAND must be positive."))

    if issues:
        return issues

    return [
        create_issue(
            INFO_SEVERITY,
            (f"Model config passed: in_dim={config.in_dim}, d_model={config.d_model}, num_layers={config.num_layers}."),
        ),
    ]


def check_model_dry_build() -> list[ReadinessIssue]:
    """Check whether the real model can be instantiated."""

    try:
        device = get_model_device()
        config = build_mtl_config()
        model = build_multitask_mamba_model(
            config=config,
            device=device,
        )
    except Exception as error:
        return [
            create_issue(
                ERROR_SEVERITY,
                f"Model dry build failed: {error}",
            ),
        ]

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    return [
        create_issue(
            INFO_SEVERITY,
            f"Model dry build passed on device={device}; parameters={parameter_count:,}.",
        ),
    ]


def check_real_model_readiness() -> RealModelReadiness:
    """Run all real model readiness checks."""

    issues: list[ReadinessIssue] = []

    if is_real_model_enabled():
        issues.append(create_issue(INFO_SEVERITY, "Real model mode is enabled."))
    else:
        issues.append(
            create_issue(
                WARNING_SEVERITY,
                "Real model mode is disabled: Model_USE_REAL_MODEL=false.",
            ),
        )

    issues.extend(check_mamba_import())
    issues.extend(check_checkpoint_path())
    issues.extend(check_model_config())

    if not any(issue.severity == ERROR_SEVERITY for issue in issues):
        issues.extend(check_model_dry_build())

    return RealModelReadiness(issues=issues)


def format_real_model_readiness(readiness: RealModelReadiness) -> str:
    """Format readiness check result."""

    lines = ["PULSE real model readiness check"]

    for issue in readiness.issues:
        prefix = {
            ERROR_SEVERITY: "[ERROR]",
            WARNING_SEVERITY: "[WARN]",
            INFO_SEVERITY: "[INFO]",
        }.get(issue.severity, "[INFO]")

        lines.append(f"{prefix} {issue.message}")

    if readiness.is_ready:
        lines.append("[OK] Real model prerequisites look ready.")
    else:
        lines.append("[FAILED] Real model prerequisites are not ready.")

    return "\n".join(lines)


def main() -> None:
    """Run real model readiness checks from CLI."""

    readiness = check_real_model_readiness()
    print(format_real_model_readiness(readiness))

    if not readiness.is_ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
