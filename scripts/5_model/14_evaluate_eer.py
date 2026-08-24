#!/usr/bin/env python3
"""Calculate bucket-wise EER and MinDCF from backend embedding manifests."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "src/python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from speaker_verification_evaluation import (  # noqa: E402
    EmbeddingSet,
    compute_verification_metrics,
    load_embedding_set,
    read_trials,
    score_trials,
)
from speaker_verification_evaluation.manifests import (  # noqa: E402
    ManifestError,
    ScoredTrial,
    file_sha256,
)
from speaker_verification_evaluation.metrics import MetricError  # noqa: E402


NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class EvaluationError(RuntimeError):
    """The EER evaluation cannot be completed safely."""


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _embedding_spec(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not NAME_PATTERN.fullmatch(name) or not raw_path:
        raise argparse.ArgumentTypeError(
            "embedding spec must be BACKEND=MANIFEST with a portable backend name"
        )
    return name, _path(raw_path)


def _bucket_tag(bucket: int | None) -> str:
    return "native" if bucket is None else str(bucket)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_scores(path: Path, scored: Sequence[ScoredTrial]) -> None:
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.writer(sink, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["trial_id", "enroll_id", "test_id", "target", "trial_type", "score"]
        )
        for item in scored:
            trial = item.trial
            writer.writerow(
                [
                    trial.trial_id,
                    trial.enroll_id,
                    trial.test_id,
                    trial.target,
                    trial.trial_type,
                    f"{item.score:.12g}",
                ]
            )


def _write_summary(
    output_dir: Path,
    document: dict[str, object],
    rows: Sequence[dict[str, object]],
) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Speaker verification EER",
        "",
        "| backend | frames | trials | EER | threshold | MinDCF | normalized MinDCF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['backend']} | {row['bucket_frames']} | {row['trial_count']} | "
            f"{float(row['eer_percent']):.4f}% | "
            f"{float(row['eer_threshold']):.8f} | "
            f"{float(row['min_dcf']):.8f} | "
            f"{float(row['min_dcf_normalized']):.8f} |"
        )
    (output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _discover_buckets(
    embedding_sets: Sequence[EmbeddingSet],
) -> list[int | None]:
    common: set[int | None] | None = None
    for embedding_set in embedding_sets:
        buckets = set(embedding_set.buckets)
        common = buckets if common is None else common & buckets
    if not common:
        raise EvaluationError("embedding manifests have no common bucket")
    return sorted(common, key=lambda value: (-1 if value is None else value))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        type=_path,
        default=ROOT / "benchmarks/campplus/manifests/trials.tsv",
    )
    parser.add_argument(
        "--embeddings",
        type=_embedding_spec,
        action="append",
        required=True,
        metavar="BACKEND=MANIFEST",
        help="repeat for ORT, C Runtime, or another backend",
    )
    parser.add_argument("--buckets", type=int, nargs="+")
    parser.add_argument("--reference-backend")
    parser.add_argument("--p-target", type=float, default=0.01)
    parser.add_argument("--c-miss", type=float, default=1.0)
    parser.add_argument("--c-false-alarm", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=_path,
        default=ROOT / "results/models/campplus/final_v3/eer",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    names = [name for name, _ in args.embeddings]
    if len(names) != len(set(names)):
        parser.error("backend names must be unique")
    if args.reference_backend is not None and args.reference_backend not in names:
        parser.error("--reference-backend must name one --embeddings backend")
    if args.buckets is not None and any(value <= 0 for value in args.buckets):
        parser.error("--buckets values must be positive")

    try:
        trials = read_trials(args.trials)
        embedding_sets = [
            load_embedding_set(path, backend=name) for name, path in args.embeddings
        ]
        if args.buckets is None:
            buckets = _discover_buckets(embedding_sets)
        else:
            buckets = list(dict.fromkeys(args.buckets))

        scored_by_key: dict[tuple[str, int | None], list[ScoredTrial]] = {}
        for embedding_set in embedding_sets:
            for bucket in buckets:
                scored_by_key[(embedding_set.backend, bucket)] = score_trials(
                    trials, embedding_set, bucket
                )

        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "ready": True,
                        "trial_count": len(trials),
                        "required_input_count": len(
                            {trial.enroll_id for trial in trials}
                            | {trial.test_id for trial in trials}
                        ),
                        "backends": names,
                        "buckets": [
                            "native" if bucket is None else bucket for bucket in buckets
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        summary_path = args.output_dir / "summary.json"
        if summary_path.exists() and not args.force:
            raise EvaluationError(
                f"output exists: {summary_path}; pass --force to replace it"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        score_dir = args.output_dir / "scores"
        metric_dir = args.output_dir / "metrics"
        score_dir.mkdir(parents=True, exist_ok=True)
        metric_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []
        metric_documents: dict[tuple[str, int | None], dict[str, object]] = {}
        for embedding_set in embedding_sets:
            for bucket in buckets:
                scored = scored_by_key[(embedding_set.backend, bucket)]
                metrics = compute_verification_metrics(
                    [item.score for item in scored],
                    [item.trial.target for item in scored],
                    p_target=args.p_target,
                    c_miss=args.c_miss,
                    c_false_alarm=args.c_false_alarm,
                )
                tag = _bucket_tag(bucket)
                score_path = score_dir / f"scores_{embedding_set.backend}_{tag}.tsv"
                metric_path = metric_dir / f"metrics_{embedding_set.backend}_{tag}.json"
                _write_scores(score_path, scored)
                metric_document: dict[str, object] = {
                    "schema_version": 1,
                    "backend": embedding_set.backend,
                    "bucket_frames": bucket,
                    "trial_manifest": _display_path(args.trials),
                    "trial_manifest_sha256": file_sha256(args.trials),
                    "embedding_manifest": _display_path(
                        embedding_set.manifest_path
                    ),
                    "embedding_manifest_sha256": file_sha256(
                        embedding_set.manifest_path
                    ),
                    "score_file": _display_path(score_path),
                    "metrics": metrics.to_dict(),
                }
                metric_path.write_text(
                    json.dumps(metric_document, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                metric_documents[(embedding_set.backend, bucket)] = metric_document
                rows.append(
                    {
                        "backend": embedding_set.backend,
                        "bucket_frames": tag,
                        "trial_count": metrics.trial_count,
                        "target_count": metrics.target_count,
                        "non_target_count": metrics.non_target_count,
                        "eer_rate": metrics.eer_rate,
                        "eer_percent": metrics.eer_rate * 100.0,
                        "eer_threshold": metrics.eer_threshold,
                        "min_dcf": metrics.min_dcf,
                        "min_dcf_normalized": metrics.min_dcf_normalized,
                    }
                )

        reference = args.reference_backend or names[0]
        comparisons: list[dict[str, object]] = []
        for bucket in buckets:
            reference_scores = scored_by_key[(reference, bucket)]
            reference_metrics = metric_documents[(reference, bucket)]["metrics"]
            assert isinstance(reference_metrics, dict)
            for backend in names:
                if backend == reference:
                    continue
                candidate_scores = scored_by_key[(backend, bucket)]
                deltas = [
                    candidate.score - baseline.score
                    for baseline, candidate in zip(reference_scores, candidate_scores)
                ]
                candidate_metrics = metric_documents[(backend, bucket)]["metrics"]
                assert isinstance(candidate_metrics, dict)
                reference_eer = reference_metrics["eer"]
                candidate_eer = candidate_metrics["eer"]
                reference_dcf = reference_metrics["min_dcf"]
                candidate_dcf = candidate_metrics["min_dcf"]
                assert isinstance(reference_eer, dict)
                assert isinstance(candidate_eer, dict)
                assert isinstance(reference_dcf, dict)
                assert isinstance(candidate_dcf, dict)
                comparisons.append(
                    {
                        "bucket_frames": bucket,
                        "reference_backend": reference,
                        "candidate_backend": backend,
                        "eer_delta_percentage_points": (
                            float(candidate_eer["percent"])
                            - float(reference_eer["percent"])
                        ),
                        "normalized_min_dcf_delta": (
                            float(candidate_dcf["normalized"])
                            - float(reference_dcf["normalized"])
                        ),
                        "score_max_abs_difference": max(abs(value) for value in deltas),
                        "score_mean_abs_difference": (
                            sum(abs(value) for value in deltas) / len(deltas)
                        ),
                    }
                )

        document: dict[str, object] = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "trial_manifest": _display_path(args.trials),
            "trial_manifest_sha256": file_sha256(args.trials),
            "reference_backend": reference,
            "configuration": {
                "threshold_rule": "accept_if_score_greater_than_or_equal",
                "p_target": args.p_target,
                "c_miss": args.c_miss,
                "c_false_alarm": args.c_false_alarm,
            },
            "rows": rows,
            "comparisons": comparisons,
        }
        _write_summary(args.output_dir, document, rows)
        print(f"summary: {args.output_dir / 'summary.json'}")
        return 0
    except (EvaluationError, ManifestError, MetricError, OSError, ValueError) as exc:
        print(f"EER evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
