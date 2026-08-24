"""ONNX Runtime backend for the fixed-bucket microphone pipeline."""

from .runtime_onnx import OrtPipelineError, run_embedding_onnx

__all__ = ["OrtPipelineError", "run_embedding_onnx"]
