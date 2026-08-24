"""검증 결과를 재현 가능한 Reference Runtime 기준선으로 고정한다."""

from .reference_result_freezer import FreezeResultError, freeze_reference_results

__all__ = ["FreezeResultError", "freeze_reference_results"]
