#ifndef CAMPP_RUNTIME_STATUS_CODE_H
#define CAMPP_RUNTIME_STATUS_CODE_H

/*
 * Runtime 함수가 반환할 상태 코드와 오류 분류를 선언할 파일이다.
 *
 * 구분할 오류:
 * - 파일 열기와 읽기 실패
 * - binary magic, version, checksum 불일치
 * - 지원하지 않는 opcode 또는 dtype
 * - 입력 shape와 bucket 불일치
 * - 메모리 부족과 buffer 범위 초과
 * - kernel 실행 실패
 *
 * 모든 Runtime 함수는 CamppStatus를 반환하고 결과는 out parameter로 넘긴다.
 * 성공은 CAMPP_STATUS_OK 하나뿐이므로 호출자는 != OK만 확인하면 된다.
 */

typedef enum CamppStatus {
    CAMPP_STATUS_OK = 0,

    /* 호출 규약 */
    CAMPP_STATUS_INVALID_ARGUMENT = 1,
    CAMPP_STATUS_NOT_IMPLEMENTED = 2,

    /* 파일 입출력 */
    CAMPP_STATUS_FILE_NOT_FOUND = 10,
    CAMPP_STATUS_FILE_READ_FAILED = 11,

    /* binary 형식 */
    CAMPP_STATUS_INVALID_MAGIC = 20,
    CAMPP_STATUS_UNSUPPORTED_VERSION = 21,
    CAMPP_STATUS_CHECKSUM_MISMATCH = 22,
    CAMPP_STATUS_CORRUPT_PLAN = 23,
    CAMPP_STATUS_CORRUPT_WEIGHTS = 24,

    /* 그래프 내용 */
    CAMPP_STATUS_UNSUPPORTED_OPCODE = 30,
    CAMPP_STATUS_UNSUPPORTED_DTYPE = 31,
    CAMPP_STATUS_MISSING_KERNEL = 32,
    CAMPP_STATUS_MISSING_ATTRIBUTE = 33,
    CAMPP_STATUS_BACKEND_MISMATCH = 34,

    /* 실행 시 입력 */
    CAMPP_STATUS_SHAPE_MISMATCH = 40,
    CAMPP_STATUS_BUCKET_MISMATCH = 41,
    CAMPP_STATUS_TENSOR_NOT_BOUND = 42,

    /* 메모리 */
    CAMPP_STATUS_OUT_OF_MEMORY = 50,
    CAMPP_STATUS_BUFFER_OVERFLOW = 51,

    /* 실행 */
    CAMPP_STATUS_KERNEL_FAILED = 60
} CamppStatus;

/* 로그와 진단 메시지에 쓸 짧은 이름이다. 알 수 없는 값도 문자열을 돌려준다. */
static inline const char *campp_status_name(CamppStatus status)
{
    switch (status) {
    case CAMPP_STATUS_OK: return "OK";
    case CAMPP_STATUS_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
    case CAMPP_STATUS_NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
    case CAMPP_STATUS_FILE_NOT_FOUND: return "FILE_NOT_FOUND";
    case CAMPP_STATUS_FILE_READ_FAILED: return "FILE_READ_FAILED";
    case CAMPP_STATUS_INVALID_MAGIC: return "INVALID_MAGIC";
    case CAMPP_STATUS_UNSUPPORTED_VERSION: return "UNSUPPORTED_VERSION";
    case CAMPP_STATUS_CHECKSUM_MISMATCH: return "CHECKSUM_MISMATCH";
    case CAMPP_STATUS_CORRUPT_PLAN: return "CORRUPT_PLAN";
    case CAMPP_STATUS_CORRUPT_WEIGHTS: return "CORRUPT_WEIGHTS";
    case CAMPP_STATUS_UNSUPPORTED_OPCODE: return "UNSUPPORTED_OPCODE";
    case CAMPP_STATUS_UNSUPPORTED_DTYPE: return "UNSUPPORTED_DTYPE";
    case CAMPP_STATUS_MISSING_KERNEL: return "MISSING_KERNEL";
    case CAMPP_STATUS_MISSING_ATTRIBUTE: return "MISSING_ATTRIBUTE";
    case CAMPP_STATUS_BACKEND_MISMATCH: return "BACKEND_MISMATCH";
    case CAMPP_STATUS_SHAPE_MISMATCH: return "SHAPE_MISMATCH";
    case CAMPP_STATUS_BUCKET_MISMATCH: return "BUCKET_MISMATCH";
    case CAMPP_STATUS_TENSOR_NOT_BOUND: return "TENSOR_NOT_BOUND";
    case CAMPP_STATUS_OUT_OF_MEMORY: return "OUT_OF_MEMORY";
    case CAMPP_STATUS_BUFFER_OVERFLOW: return "BUFFER_OVERFLOW";
    case CAMPP_STATUS_KERNEL_FAILED: return "KERNEL_FAILED";
    default: return "UNKNOWN";
    }
}

#endif /* CAMPP_RUNTIME_STATUS_CODE_H */
