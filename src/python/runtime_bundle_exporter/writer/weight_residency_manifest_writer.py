"""Write auditable artifacts for bucket-local static weight plans."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..planner.weight_residency_planner import StaticWeightPlan


def write_static_weight_artifacts(
    plan: StaticWeightPlan,
    *,
    execution_plan_path: Path,
    weights_path: Path,
    manifest_path: Path,
    csv_path: Path,
    artifact_root: Path | None = None,
) -> dict:
    """Write the remapped plan, immutable blob, JSON manifest, and CSV index."""

    targets = (
        execution_plan_path, weights_path, manifest_path, csv_path,
    )
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
    execution_plan_path.write_bytes(plan.plan_bytes)
    weights_path.write_bytes(plan.weight_bytes)

    def display(path: Path) -> str:
        if artifact_root is not None:
            try:
                return path.resolve().relative_to(
                    artifact_root.resolve()
                ).as_posix()
            except ValueError:
                pass
        return path.as_posix()

    document = plan.to_dict(include_entries=True)
    document["artifacts"] = {
        "execution_plan": display(execution_plan_path),
        "weights": display(weights_path),
        "csv": display(csv_path),
    }
    manifest_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = (
            "tensor_id", "name", "dtype", "shape", "used",
            "first_use_operator", "last_use_operator", "operator_ids",
            "source_offset", "destination_offset", "byte_size",
            "alignment", "sha256",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for entry in plan.entries:
            writer.writerow({
                "tensor_id": entry.tensor_id,
                "name": entry.name or "",
                "dtype": entry.dtype or "",
                "shape": "x".join(str(value) for value in entry.shape),
                "used": entry.used,
                "first_use_operator": (
                    "" if entry.first_use_operator is None
                    else entry.first_use_operator
                ),
                "last_use_operator": (
                    "" if entry.last_use_operator is None
                    else entry.last_use_operator
                ),
                "operator_ids": " ".join(map(str, entry.operator_ids)),
                "source_offset": entry.source_offset,
                "destination_offset": entry.destination_offset,
                "byte_size": entry.byte_size,
                "alignment": entry.alignment,
                "sha256": entry.sha256,
            })
    return document


__all__ = ["write_static_weight_artifacts"]
