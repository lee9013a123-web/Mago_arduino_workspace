# Generic address kernel

border, padding, dilation, spatial tail을 담당한다. padding point는 `NULL`로
기록하며 MAC은 정확히 `input_zero`를 사용해야 한다. 최적화된 interior 경로의
가정이 하나라도 실패하면 이 경로로 내려온다.

초기 단계에서는 기존 V4 generic 결과를 adapter로 재사용할 수 있다. 독립
구현으로 교체할 때 memory-safety와 모든 retained tensor bitwise 검증이 필요하다.
