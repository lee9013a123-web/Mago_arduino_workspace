# 3x3 interior address kernel

row base, kernel-row step, kernel-column step, output-column step을 이용해 곱셈
없이 pointer를 전개한다. interior 범위는 plan 생성 시 증명하고 hot loop에서는
point별 padding/storage 검사를 반복하지 않는다.

후속 실험에서는 pointer table을 제거하고 base/stride를 MAC에 직접 넘기는
sliding-window contract를 별도 candidate로 둔다. 기존 contract를 즉시 바꾸지
않아 MAC multiply 실험과 결과를 분리한다.
