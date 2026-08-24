"""Resolve ONNX enrollment templates without mixing them with C templates."""

from __future__ import annotations

from pathlib import Path

from similarity_detect.scoring import SimilarityError


def resolve_onnx_speaker_embedding(pipeline_root: Path, value: str) -> Path:
    boundary = pipeline_root.resolve()
    candidate = Path(value)
    relative = candidate if candidate.is_absolute() else pipeline_root / candidate
    if relative.is_file():
        resolved = relative.resolve()
        onnx_boundary = (pipeline_root / "voice_onnx").resolve()
        if not resolved.is_relative_to(onnx_boundary):
            raise SimilarityError(
                f"ORT speaker embedding must stay inside {onnx_boundary}: {resolved}"
            )
        return resolved
    embedded_file = pipeline_root / "voice_onnx/embedded" / value
    if embedded_file.is_file():
        return embedded_file.resolve()
    template = embedded_file / "mean_embedding.f32"
    if template.is_file():
        return template.resolve()
    raise SimilarityError(
        f"cannot resolve ORT speaker embedding {value!r} below "
        f"{boundary / 'voice_onnx/embedded'}"
    )
