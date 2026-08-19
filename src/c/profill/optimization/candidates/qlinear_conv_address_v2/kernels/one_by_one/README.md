# 1x1 address kernel

첫 input pointer를 한 번 계산한 뒤 `output_step_bytes` 덧셈만으로 최대 8개
spatial pointer를 만든다. 전체 tile에 대한 storage 검사는 처음과 마지막
pointer에서 한 번씩만 수행한다. 9x8 pointer table 전체 `memset`은 금지한다.
