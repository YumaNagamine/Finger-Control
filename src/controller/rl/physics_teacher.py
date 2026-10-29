"""Generate geometry-only teacher actions from validated moment-arm models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from controller.servo_mapping import TENDONS


JOINTS = ("DIP", "PIP", "MCP")
PHYSICS_TEACHER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MomentArmJointModel:
    joint: str
    coefficients_by_tendon: Mapping[str, np.ndarray]
    antiderivatives_by_tendon: Mapping[str, np.ndarray]
    valid_range_rad: tuple[float, float]
    source_path: Path
    source_sha256: str
    source_direction_by_tendon: Mapping[str, str]
    direction_policy: str


class MomentArmTeacherModel:
    """Validated separable moment-arm model with analytic integration."""

    def __init__(self, joint_models: Mapping[str, MomentArmJointModel]) -> None:
        if tuple(joint_models) != JOINTS:
            raise ValueError(f"joint_models must be ordered as {JOINTS}")
        self.joint_models = dict(joint_models)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "MomentArmTeacherModel":
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Moment-arm directory not found: {root}")
        models: dict[str, MomentArmJointModel] = {}
        for joint in JOINTS:
            path = root / f"moment_arm_{joint}_average.json"
            models[joint] = _load_joint_model(path, joint)
        return cls(models)

    @property
    def valid_ranges_rad(self) -> dict[str, tuple[float, float]]:
        return {
            joint: model.valid_range_rad
            for joint, model in self.joint_models.items()
        }

    def action_from_session_origin(
        self,
        session_initial_posture_rad: Sequence[float],
        target_posture_rad: Sequence[float],
    ) -> np.ndarray:
        """Return absolute target excursion relative to one session origin.

        The output is not a delta from a previous teacher action.  Each row is
        recomputed from the supplied session initial posture.
        """

        initial = _posture_vector(
            session_initial_posture_rad,
            "session_initial_posture_rad",
        )
        target = _posture_vector(target_posture_rad, "target_posture_rad")
        for index, joint in enumerate(JOINTS):
            low, high = self.joint_models[joint].valid_range_rad
            for name, value in (
                ("session initial", initial[index]),
                ("target", target[index]),
            ):
                if value < low - 1e-12 or value > high + 1e-12:
                    raise ValueError(
                        f"{joint} {name} angle {math.degrees(value):.6f} deg "
                        f"is outside [{math.degrees(low):.6f}, "
                        f"{math.degrees(high):.6f}] deg"
                    )

        action = np.zeros(len(TENDONS), dtype=np.float64)
        for joint_index, joint in enumerate(JOINTS):
            model = self.joint_models[joint]
            for tendon_index, tendon in enumerate(TENDONS):
                primitive = model.antiderivatives_by_tendon[tendon]
                action[tendon_index] += float(
                    np.polyval(primitive, target[joint_index])
                    - np.polyval(primitive, initial[joint_index])
                )
        if not np.isfinite(action).all():
            raise ValueError("Integrated teacher action contains a non-finite value")
        return action

    def provenance(self) -> dict[str, Any]:
        return {
            "joint_order": list(JOINTS),
            "tendon_order": list(TENDONS),
            "angle_unit": "rad",
            "moment_arm_unit": "mm/rad",
            "action_unit": "mm",
            "action_reference": "session_initial_posture",
            "action_is_incremental": False,
            "integration": (
                "analytic antiderivative per univariate joint polynomial; "
                "summed across joints"
            ),
            "path_dependence": (
                "path independent for the validated separable, "
                "direction-independent models"
            ),
            "excursion_sign_convention": (
                "delta_L_mm = integral r_mm_per_rad(theta) dtheta; no extra "
                "sign inversion, matching predict_excursion_for_control.py"
            ),
            "models": {
                joint: {
                    "source_path": str(model.source_path),
                    "source_sha256": model.source_sha256,
                    "valid_range_deg": [
                        math.degrees(model.valid_range_rad[0]),
                        math.degrees(model.valid_range_rad[1]),
                    ],
                    "source_direction_by_tendon": dict(
                        model.source_direction_by_tendon
                    ),
                    "direction_policy": model.direction_policy,
                }
                for joint, model in self.joint_models.items()
            },
        }


@dataclass
class PhysicsTeacherDataset:
    initial_postures_rad: np.ndarray
    target_postures_rad: np.ndarray
    actions_excursion_mm: np.ndarray
    movement_directions: list[tuple[str, str, str]]
    provenance: dict[str, Any]

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        csv_path = root / "physics_teacher_pairs.csv"
        manifest_path = root / "physics_teacher_manifest.json"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "teacher_pair_id",
                    *[f"initial_{joint}_rad" for joint in JOINTS],
                    *[f"target_{joint}_rad" for joint in JOINTS],
                    *[f"{joint}_movement_direction" for joint in JOINTS],
                    *[f"{tendon}_action_excursion_mm" for tendon in TENDONS],
                ]
            )
            for index in range(len(self.initial_postures_rad)):
                writer.writerow(
                    [
                        index,
                        *self.initial_postures_rad[index].tolist(),
                        *self.target_postures_rad[index].tolist(),
                        *self.movement_directions[index],
                        *self.actions_excursion_mm[index].tolist(),
                    ]
                )
        manifest = {
            "schema_version": PHYSICS_TEACHER_SCHEMA_VERSION,
            "dataset_kind": "synthetic_physics_teacher_pairs",
            "pair_count": int(len(self.initial_postures_rad)),
            "contains_real_hardware_transitions": False,
            "contains_tension": False,
            "contains_next_state": False,
            "contains_reward": False,
            "intended_use": "supervised physics-augmented policy pretraining",
            "must_not_be_mixed_as_real_awac_transitions": True,
            **self.provenance,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        return {"pairs": csv_path, "manifest": manifest_path}


class PhysicsTeacherGenerator:
    def __init__(self, model: MomentArmTeacherModel) -> None:
        self.model = model

    def generate_pairs(
        self,
        pairs: Iterable[tuple[Sequence[float], Sequence[float]]],
    ) -> PhysicsTeacherDataset:
        initial_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []
        action_rows: list[np.ndarray] = []
        directions: list[tuple[str, str, str]] = []
        for initial_values, target_values in pairs:
            initial = _posture_vector(initial_values, "session_initial_posture_rad")
            target = _posture_vector(target_values, "target_posture_rad")
            action = self.model.action_from_session_origin(initial, target)
            initial_rows.append(initial)
            target_rows.append(target)
            action_rows.append(action)
            directions.append(
                tuple(
                    "increasing"
                    if delta > 0.0
                    else "decreasing"
                    if delta < 0.0
                    else "static"
                    for delta in target - initial
                )
            )
        count = len(initial_rows)
        return PhysicsTeacherDataset(
            initial_postures_rad=np.asarray(initial_rows, dtype=np.float64).reshape(
                count,
                len(JOINTS),
            ),
            target_postures_rad=np.asarray(target_rows, dtype=np.float64).reshape(
                count,
                len(JOINTS),
            ),
            actions_excursion_mm=np.asarray(action_rows, dtype=np.float64).reshape(
                count,
                len(TENDONS),
            ),
            movement_directions=directions,
            provenance=self.model.provenance(),
        )


def _load_joint_model(path: Path, expected_joint: str) -> MomentArmJointModel:
    if not path.is_file():
        raise FileNotFoundError(f"Moment-arm JSON not found: {path}")
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    if str(payload.get("joint")) != expected_joint:
        raise ValueError(f"Expected joint={expected_joint} in {path}")
    if payload.get("angle_unit") != "rad":
        raise ValueError(f"Moment-arm angle_unit must be 'rad' in {path}")
    if payload.get("moment_arm_unit") != "mm/rad":
        raise ValueError(f"Moment-arm unit must be 'mm/rad' in {path}")
    if bool(payload.get("prediction_direction_dependent", False)):
        raise ValueError(
            f"Direction-dependent model requires explicit directional paths: {path}"
        )

    raw_coefficients = payload.get("moment_arm_coefficients")
    if not isinstance(raw_coefficients, Mapping):
        raise ValueError(f"Missing moment_arm_coefficients in {path}")
    fit_degree = int(payload.get("fit_degree", payload.get("moment_arm_fit_degree", -1)))
    coefficient_order = payload.get("coefficient_order")
    expected_order = [f"moment_arm_c{power}" for power in range(fit_degree, -1, -1)]
    if coefficient_order != expected_order:
        raise ValueError(
            f"Coefficient order does not match descending polynomial powers in {path}"
        )

    coefficients: dict[str, np.ndarray] = {}
    primitives: dict[str, np.ndarray] = {}
    for tendon in TENDONS:
        raw = raw_coefficients.get(tendon)
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != (fit_degree + 1,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {tendon} coefficients in {path}")
        primitive = np.polyint(values)
        if not np.allclose(np.polyder(primitive), values, rtol=0.0, atol=1e-12):
            raise ValueError(f"Analytic antiderivative validation failed in {path}")
        coefficients[tendon] = values
        primitives[tendon] = primitive

    valid_range_deg = _valid_range_deg(payload, path)
    valid_range_rad = tuple(math.radians(value) for value in valid_range_deg)
    _validate_analytic_integrals(coefficients, primitives, valid_range_rad, path)

    raw_directions = payload.get("source_direction_by_tendon", {})
    if not isinstance(raw_directions, Mapping):
        raise ValueError(f"source_direction_by_tendon must be an object in {path}")
    directions = {tendon: str(raw_directions.get(tendon, "unknown")) for tendon in TENDONS}
    return MomentArmJointModel(
        joint=expected_joint,
        coefficients_by_tendon=coefficients,
        antiderivatives_by_tendon=primitives,
        valid_range_rad=valid_range_rad,
        source_path=path.resolve(),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        source_direction_by_tendon=directions,
        direction_policy=str(payload.get("direction_policy", "")),
    )


def _valid_range_deg(payload: Mapping[str, Any], path: Path) -> tuple[float, float]:
    candidates: list[tuple[float, float]] = []
    for key in ("evaluation_range_deg", "fit_range_deg"):
        raw = payload.get(key)
        if isinstance(raw, list) and len(raw) == 2:
            candidates.append((float(raw[0]), float(raw[1])))
            if key == "evaluation_range_deg":
                break
    metrics = payload.get("fit_metrics")
    if isinstance(metrics, Mapping):
        tendon_ranges: list[tuple[float, float]] = []
        for tendon in TENDONS:
            item = metrics.get(tendon)
            if not isinstance(item, Mapping):
                continue
            low = item.get("angle_min_deg", item.get("fit_range_min_deg"))
            high = item.get("angle_max_deg", item.get("fit_range_max_deg"))
            if low is not None and high is not None:
                tendon_ranges.append((float(low), float(high)))
        if tendon_ranges:
            candidates.append(
                (
                    max(item[0] for item in tendon_ranges),
                    min(item[1] for item in tendon_ranges),
                )
            )
    if not candidates:
        raise ValueError(f"No valid angle range is recorded in {path}")
    low = max(item[0] for item in candidates)
    high = min(item[1] for item in candidates)
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError(f"Moment-arm valid angle ranges do not overlap in {path}")
    return low, high


def _validate_analytic_integrals(
    coefficients: Mapping[str, np.ndarray],
    primitives: Mapping[str, np.ndarray],
    valid_range_rad: tuple[float, float],
    path: Path,
) -> None:
    low, high = valid_range_rad
    nodes, weights = np.polynomial.legendre.leggauss(32)
    quadrature_angles = 0.5 * (high - low) * nodes + 0.5 * (high + low)
    for tendon in TENDONS:
        analytic = float(
            np.polyval(primitives[tendon], high)
            - np.polyval(primitives[tendon], low)
        )
        numerical = float(
            0.5
            * (high - low)
            * np.dot(
                weights,
                np.polyval(coefficients[tendon], quadrature_angles),
            )
        )
        if not math.isclose(analytic, numerical, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(
                f"Analytic and numerical integrals disagree for {tendon} in {path}"
            )


def _posture_vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(JOINTS),):
        raise ValueError(f"{name} must contain {JOINTS}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains a non-finite value")
    return vector


def _load_pair_csv(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            try:
                initial = np.deg2rad(
                    [float(row[f"initial_{joint}_deg"]) for joint in JOINTS]
                )
                target = np.deg2rad(
                    [float(row[f"target_{joint}_deg"]) for joint in JOINTS]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid teacher pair at row {row_number}") from exc
            pairs.append((initial, target))
    return pairs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate moment-arm physics teacher pairs."
    )
    parser.add_argument("moment_arm_directory", type=Path)
    parser.add_argument("pair_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model = MomentArmTeacherModel.from_directory(args.moment_arm_directory)
    dataset = PhysicsTeacherGenerator(model).generate_pairs(
        _load_pair_csv(args.pair_csv)
    )
    paths = dataset.save(args.output_dir)
    print(f"Teacher pairs: {len(dataset.initial_postures_rad)}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

