# Runtime 원시 실행 결과

매 실행에서 생성되는 ORT/C 중간 Tensor, embedding, trace와 임시 비교 결과를 저장한다.
재현 가능한 원시 출력이며 확정 보고서는 `results/runtime/`에 따로 남긴다.

`kernel_optimization/`은 `dense_slab`, `cache_packed`, `e7` 최적화 단계의
bundle과 검증 작업 파일을 저장한다. `benchmarks/`는 이 bundle들을 대상으로 한
성능 측정 결과를 저장하므로 최적화 산출물과 분리한다.
