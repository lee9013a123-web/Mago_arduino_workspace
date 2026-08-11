/*
 * BatchNorm affine, ReLU, Quantize와 다음 Conv 입력 packing을 하나로 처리할 파일이다.
 * BN을 Conv weight에 단순 folding하는 연산이 아니며 중간 activation 생성을 제거하는 융합이다.
 */
