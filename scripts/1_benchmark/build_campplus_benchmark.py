#!/usr/bin/env python3
"""Build reproducible CAM++ benchmark manifests on Arduino UNO Q.

The script does not copy audio. It scans the existing data tree and writes
small, Git-friendly metadata under benchmarks/campplus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import wave
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


POSITIVE_RE = re.compile(r"^(?P<speaker>.+)_id_(?P<index>\d+)$", re.IGNORECASE)
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class AudioRecord:
    input_id: str
    split: str
    speaker_id: str
    relative_path: str
    duration_sec: float | None
    sample_rate_hz: int | None
    channels: int | None
    sample_width_bytes: int | None
    sha256: str
    latency_selected: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CAM++ benchmark inputs, trials, checksums, and protocol."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.home() / "workspace/egs/accelerate_CAM",
        help="accelerate_CAM repository root",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "workspace/data",
        help="workspace data root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory (default: REPO/benchmarks/campplus)",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_wav_metadata(path: Path) -> tuple[float | None, int | None, int | None, int | None]:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            sample_rate = wav.getframerate()
            duration = frames / sample_rate if sample_rate else None
            return duration, sample_rate, wav.getnchannels(), wav.getsampwidth()
    except (wave.Error, EOFError):
        return None, None, None, None


def sanitize_id(value: str) -> str:
    cleaned = NON_ALNUM_RE.sub("_", value.lower()).strip("_")
    return cleaned or "audio"


def discover_wavs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Required directory is missing: {root}")
    return sorted(path for path in root.rglob("*.wav") if path.is_file())


def infer_positive_speaker(path: Path) -> tuple[str, int]:
    match = POSITIVE_RE.match(path.stem)
    if not match:
        raise ValueError(
            f"Positive filename must match SPEAKER_id_NUMBER.wav: {path.name}"
        )
    return match.group("speaker").lower(), int(match.group("index"))


def infer_enroll_speaker(path: Path, data_root: Path, speakers: list[str]) -> str:
    relative_text = path.relative_to(data_root).as_posix().lower()
    tokens = set(filter(None, NON_ALNUM_RE.split(relative_text)))
    candidates = []
    for speaker in speakers:
        if speaker in tokens or path.stem.lower().startswith(f"{speaker}_"):
            candidates.append(speaker)
    if len(candidates) != 1:
        raise ValueError(
            "Could not uniquely map enrollment audio to a registered speaker: "
            f"{relative_text}. Expected one of {speakers}, got {candidates}."
        )
    return candidates[0]


def make_input_id(split: str, path: Path, data_root: Path) -> str:
    relative_without_suffix = path.relative_to(data_root).with_suffix("").as_posix()
    return f"{split}__{sanitize_id(relative_without_suffix)}"


def build_records(data_root: Path) -> tuple[list[AudioRecord], dict[str, int]]:
    split_roots = {
        "enroll": data_root / "speaker_enroll_7sec",
        "positive": data_root / "speaker_identify_3sec_positive",
        "negative": data_root / "speaker_negative_unregistered",
    }
    files_by_split = {split: discover_wavs(root) for split, root in split_roots.items()}

    if not files_by_split["positive"]:
        raise ValueError("No positive identification WAV files were found")
    if not files_by_split["enroll"]:
        raise ValueError("No enrollment WAV files were found")
    if not files_by_split["negative"]:
        raise ValueError("No unregistered negative WAV files were found")

    positive_info = {
        path: infer_positive_speaker(path) for path in files_by_split["positive"]
    }
    registered_speakers = sorted({speaker for speaker, _ in positive_info.values()})

    records: list[AudioRecord] = []
    positive_indexes: dict[str, int] = {}
    for split in ("enroll", "positive", "negative"):
        for path in files_by_split[split]:
            if split == "positive":
                speaker, positive_index = positive_info[path]
            elif split == "enroll":
                speaker = infer_enroll_speaker(path, data_root, registered_speakers)
                positive_index = -1
            else:
                relative = path.relative_to(split_roots[split])
                group = relative.parts[0] if len(relative.parts) > 1 else path.stem
                speaker = f"unregistered:{sanitize_id(group)}"
                positive_index = -1

            duration, sample_rate, channels, sample_width = read_wav_metadata(path)
            record = AudioRecord(
                input_id=make_input_id(split, path, data_root),
                split=split,
                speaker_id=speaker,
                relative_path=path.relative_to(data_root).as_posix(),
                duration_sec=duration,
                sample_rate_hz=sample_rate,
                channels=channels,
                sample_width_bytes=sample_width,
                sha256=sha256_file(path),
            )
            records.append(record)
            if split == "positive":
                positive_indexes[record.input_id] = positive_index

    duplicate_ids = [
        input_id
        for input_id, count in _counts(record.input_id for record in records).items()
        if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"Duplicate input IDs were generated: {duplicate_ids}")

    select_latency_inputs(records, positive_indexes)
    return records, positive_indexes


def _counts(values):
    counts = defaultdict(int)
    for value in values:
        counts[value] += 1
    return counts


def select_latency_inputs(
    records: list[AudioRecord], positive_indexes: dict[str, int]
) -> None:
    """Select a small, deterministic latency subset while keeping all files for accuracy."""
    enroll_by_speaker: dict[str, list[AudioRecord]] = defaultdict(list)
    negatives_by_group: dict[str, list[AudioRecord]] = defaultdict(list)

    for record in records:
        if record.split == "enroll":
            enroll_by_speaker[record.speaker_id].append(record)
        elif record.split == "negative":
            negatives_by_group[record.speaker_id].append(record)

    for speaker in sorted(enroll_by_speaker):
        sorted(enroll_by_speaker[speaker], key=lambda item: item.relative_path)[0].latency_selected = True

    for record in records:
        if record.split == "positive" and positive_indexes[record.input_id] in {0, 2}:
            record.latency_selected = True

    for group in sorted(negatives_by_group):
        sorted(negatives_by_group[group], key=lambda item: item.relative_path)[0].latency_selected = True


def build_trials(records: list[AudioRecord]) -> list[dict[str, str | int]]:
    enrollments = sorted(
        (record for record in records if record.split == "enroll"),
        key=lambda item: item.input_id,
    )
    positives = sorted(
        (record for record in records if record.split == "positive"),
        key=lambda item: item.input_id,
    )
    negatives = sorted(
        (record for record in records if record.split == "negative"),
        key=lambda item: item.input_id,
    )

    trials: list[dict[str, str | int]] = []
    trial_number = 1
    for enrollment in enrollments:
        for test in positives:
            target = int(enrollment.speaker_id == test.speaker_id)
            trial_type = "target" if target else "registered_impostor"
            trials.append(
                {
                    "trial_id": f"T{trial_number:06d}",
                    "enroll_id": enrollment.input_id,
                    "test_id": test.input_id,
                    "target": target,
                    "trial_type": trial_type,
                }
            )
            trial_number += 1

        for test in negatives:
            trials.append(
                {
                    "trial_id": f"T{trial_number:06d}",
                    "enroll_id": enrollment.input_id,
                    "test_id": test.input_id,
                    "target": 0,
                    "trial_type": "unregistered_impostor",
                }
            )
            trial_number += 1
    return trials


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNCOMMITTED"


def write_outputs(
    repo_root: Path,
    data_root: Path,
    output: Path,
    records: list[AudioRecord],
    trials: list[dict[str, str | int]],
) -> None:
    manifests = output / "manifests"
    references = output / "reference_outputs"
    manifests.mkdir(parents=True, exist_ok=True)
    references.mkdir(parents=True, exist_ok=True)

    input_rows = []
    for record in sorted(records, key=lambda item: (item.split, item.relative_path)):
        row = asdict(record)
        row["duration_sec"] = (
            f"{record.duration_sec:.6f}" if record.duration_sec is not None else "UNKNOWN"
        )
        row["sample_rate_hz"] = record.sample_rate_hz or "UNKNOWN"
        row["channels"] = record.channels or "UNKNOWN"
        row["sample_width_bytes"] = record.sample_width_bytes or "UNKNOWN"
        row["latency_selected"] = int(record.latency_selected)
        input_rows.append(row)

    write_tsv(
        manifests / "inputs.tsv",
        input_rows,
        [
            "input_id",
            "split",
            "speaker_id",
            "relative_path",
            "duration_sec",
            "sample_rate_hz",
            "channels",
            "sample_width_bytes",
            "sha256",
            "latency_selected",
        ],
    )
    write_tsv(
        manifests / "trials.tsv",
        trials,
        ["trial_id", "enroll_id", "test_id", "target", "trial_type"],
    )

    with (output / "checksums.sha256").open("w", encoding="utf-8") as stream:
        for record in sorted(records, key=lambda item: item.relative_path):
            stream.write(f"{record.sha256}  {record.relative_path}\n")

    onnx_files = sorted((repo_root / "results/static").glob("*.onnx"))
    with (output / "onnx_checksums.sha256").open("w", encoding="utf-8") as stream:
        for model in onnx_files:
            stream.write(f"{sha256_file(model)}  {model.relative_to(repo_root).as_posix()}\n")

    split_counts = _counts(record.split for record in records)
    latency_records = sorted(
        (record for record in records if record.latency_selected),
        key=lambda item: (item.split, item.input_id),
    )
    trial_type_counts = _counts(str(trial["trial_type"]) for trial in trials)
    sample_rates = sorted(
        {record.sample_rate_hz for record in records if record.sample_rate_hz is not None}
    )
    channel_counts = sorted(
        {record.channels for record in records if record.channels is not None}
    )
    registered_speakers = sorted(
        {record.speaker_id for record in records if record.split == "positive"}
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "git_commit": git_commit(repo_root),
        "registered_speakers": registered_speakers,
        "input_counts": dict(sorted(split_counts.items())),
        "latency_input_count": len(latency_records),
        "trial_count": len(trials),
        "trial_type_counts": dict(sorted(trial_type_counts.items())),
        "sample_rates_hz": sample_rates,
        "channel_counts": channel_counts,
        "onnx_models": [model.relative_to(repo_root).as_posix() for model in onnx_files],
        "excluded_from_campp_baseline": [
            "hey_jarvis.wav",
            "speaker_enroll_wakeword/",
            "kss_cer_eval_bundle/",
            "kss_cer_eval_bundle.tar.gz",
        ],
    }
    (output / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    latency_list = "\n".join(
        f"- `{record.input_id}` — `{record.relative_path}`" for record in latency_records
    )
    readme = f"""# CAM++ Benchmark Dataset

이 폴더는 음성 파일을 복사하지 않고 `{data_root}` 아래 기존 데이터의 선택 규칙과 검증 정보를 저장한다.

## 선택한 데이터

| Split | 파일 수 | 용도 |
| --- | ---: | --- |
| Enrollment | {split_counts.get('enroll', 0)} | 등록 embedding 생성 |
| Positive | {split_counts.get('positive', 0)} | 등록 화자의 positive test |
| Negative | {split_counts.get('negative', 0)} | 미등록 화자의 impostor test |

등록 화자: {', '.join(registered_speakers)}

## 제외한 데이터

- `hey_jarvis.wav`: Wakeword 입력
- `speaker_enroll_wakeword/`: Wakeword 실험용
- `kss_cer_eval_bundle/`: ASR CER 평가용
- `kss_cer_eval_bundle.tar.gz`: 데이터 archive

## 파일

- `manifests/inputs.tsv`: 선택한 음성과 metadata
- `manifests/trials.tsv`: EER용 enrollment/test 쌍과 정답
- `checksums.sha256`: 선택한 음성 파일 checksum
- `onnx_checksums.sha256`: 현재 static ONNX checksum
- `benchmark_protocol.md`: 고정 측정 절차
- `selection_summary.json`: 생성 결과 요약
- `reference_outputs/`: PyTorch 기준 embedding 저장 위치

## Checksum 검증

데이터 루트에서 실행한다.

```bash
cd {data_root}
sha256sum -c {output / 'checksums.sha256'}
```

## Latency subset

{latency_list}
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    protocol = f"""# CAM++ Benchmark Protocol

## 1. 고정 대상

- Device: Arduino UNO Q, QRB2210 Linux 영역
- Repository commit: `{git_commit(repo_root)}`
- Data root: `{data_root}`
- Registered speakers: {', '.join(registered_speakers)}
- Sample rates found: {', '.join(map(str, sample_rates)) or 'UNKNOWN'} Hz
- Channel counts found: {', '.join(map(str, channel_counts)) or 'UNKNOWN'}
- Input manifest: `manifests/inputs.tsv`
- Trial manifest: `manifests/trials.tsv`
- Input checksums: `checksums.sha256`
- ONNX checksums: `onnx_checksums.sha256`

모델 checkpoint, export script commit, ONNX opset은 ONNX 검증 단계에서 추가로 기록한다.

## 2. 데이터 선택 규칙

### 정확도·EER

- `speaker_enroll_7sec/`의 모든 WAV를 enrollment로 사용한다.
- `speaker_identify_3sec_positive/`의 모든 WAV를 positive test로 사용한다.
- `speaker_negative_unregistered/`의 모든 WAV를 unregistered negative로 사용한다.
- Enrollment와 같은 등록 화자의 positive pair는 `target=1`이다.
- 다른 등록 화자의 positive pair는 `registered_impostor`, `target=0`이다.
- 미등록 화자 pair는 `unregistered_impostor`, `target=0`이다.

### Latency

- 화자별 첫 번째 7초 enrollment 파일을 선택한다.
- 등록 화자별 `id_0`, `id_2` positive 파일을 선택한다.
- 미등록 그룹별 첫 번째 파일을 선택한다.
- 정확한 목록은 `inputs.tsv`의 `latency_selected=1` 행이다.

### 제외

- Wakeword와 ASR CER 데이터는 CAM++ 기본 benchmark에서 제외한다.
- 원본 음성은 수정·복사하지 않는다.

## 3. 정확도 측정

1. 모든 enrollment와 test 입력에서 embedding을 생성한다.
2. Embedding에 모델의 기존 normalization 규칙을 동일하게 적용한다.
3. `trials.tsv`의 각 pair에 cosine similarity를 계산한다.
4. Target·non-target score로 EER과 MinDCF를 계산한다.
5. PyTorch FP32를 reference로 저장한다.
6. ONNX FP32, custom FP32, INT8 결과를 같은 trial list로 비교한다.

## 4. 지연시간 측정

### Cold start

- 새 프로세스에서 모델 load부터 첫 embedding 출력까지 측정한다.
- 최소 10회 별도 프로세스로 반복한다.

### Warm inference

- 같은 프로세스에서 20회 warm-up한다.
- 각 latency 입력을 100회 측정한다.
- p50, p95, p99, 평균, 표준편차, RTF를 기록한다.
- 모델 load와 audio file read 시간은 inference 시간과 분리한다.

### Thread

- CPU 1·2·4 thread를 각각 측정한다.
- 동일 실행에서 thread 수를 섞지 않는다.
- CPU affinity, governor, 보드 전원 조건을 결과에 기록한다.

## 5. 입력 길이 실험

- 정확도 평가는 원본 native duration을 사용한다.
- Static shape 성능 실험은 98·298·498·998 frame 모델의 실제 시간 대응 관계를 export script에서 먼저 확인한다.
- 길이 실험용 crop·zero padding은 정확도 trial과 분리한다.
- crop 시작점과 padding 정책은 config에 고정한다.

## 6. 기록 지표

- Cold start latency
- Warm p50·p95·p99 latency
- RTF
- Peak RSS와 arena size
- CPU thread 수와 cache miss
- GPU kernel·동기화 시간(사용 시)
- Embedding cosine similarity
- EER·MinDCF
- 온도와 throttling 여부

## 7. 결과 유효 조건

- `sha256sum -c checksums.sha256`가 모두 통과한다.
- Model checksum과 Git commit이 결과에 기록되어 있다.
- Warm-up·반복 수·thread 수가 기록되어 있다.
- Reference embedding 생성 실패가 없다.
- 동일 조건 반복 측정의 변동 원인을 설명할 수 있다.
"""
    (output / "benchmark_protocol.md").write_text(protocol, encoding="utf-8")

    reference_readme = """# Reference outputs

이 폴더에는 현재 CAM++ PyTorch FP32 checkpoint로 생성한 기준 embedding을 저장한다.

권장 파일:

```text
embeddings_fp32.tsv      input_id와 embedding 파일 연결
embeddings_fp32/*.npy    입력별 float32 embedding
scores_fp32.tsv          trials.tsv 순서의 cosine score
metrics_fp32.json        EER, MinDCF와 실행 metadata
```

수치 reference는 checkpoint 경로와 실제 inference 명령을 확인한 후 생성해야 한다. 임의 값을 만들지 않는다.
"""
    (references / "README.md").write_text(reference_readme, encoding="utf-8")


def validate_records(records: list[AudioRecord]) -> list[str]:
    warnings: list[str] = []
    unknown_metadata = [record.relative_path for record in records if record.duration_sec is None]
    if unknown_metadata:
        warnings.append(
            f"Could not read standard PCM WAV metadata for {len(unknown_metadata)} file(s)."
        )
    sample_rates = {record.sample_rate_hz for record in records if record.sample_rate_hz is not None}
    if len(sample_rates) > 1:
        warnings.append(f"Multiple sample rates detected: {sorted(sample_rates)}")
    channel_counts = {record.channels for record in records if record.channels is not None}
    if len(channel_counts) > 1 or (channel_counts and channel_counts != {1}):
        warnings.append(f"Non-uniform or non-mono channels detected: {sorted(channel_counts)}")
    return warnings


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else repo_root / "benchmarks/campplus"
    )

    if not repo_root.is_dir():
        print(f"ERROR: Repository root does not exist: {repo_root}", file=sys.stderr)
        return 2
    if not data_root.is_dir():
        print(f"ERROR: Data root does not exist: {data_root}", file=sys.stderr)
        return 2

    try:
        records, _ = build_records(data_root)
        trials = build_trials(records)
        write_outputs(repo_root, data_root, output, records, trials)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    split_counts = _counts(record.split for record in records)
    trial_counts = _counts(str(trial["trial_type"]) for trial in trials)
    print(f"Generated benchmark metadata: {output}")
    print(f"Inputs: {dict(sorted(split_counts.items()))}")
    print(f"Latency subset: {sum(record.latency_selected for record in records)}")
    print(f"Trials: {len(trials)} {dict(sorted(trial_counts.items()))}")
    for warning in validate_records(records):
        print(f"WARNING: {warning}")
    print("Reference embeddings remain pending until the model inference command is provided.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
