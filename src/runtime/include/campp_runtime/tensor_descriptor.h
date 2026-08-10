/*
 * Runtime Tensor의 정적 정보를 표현할 구조체를 선언할 파일이다.
 *
 * 포함할 필드:
 * - Tensor ID, dtype, rank, dimensions, strides
 * - byte size와 storage 종류
 * - constant section offset 또는 activation arena offset
 * - input, output, view, read-only 등의 flag
 *
 * 실제 data pointer는 descriptor와 분리하여 Runtime Context에서 관리한다.
 */
