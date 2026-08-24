/* .camppmodel을 C 런타임 로더로 직접 열어 plan/weights가 살아 있는지 확인한다.
 *
 * Python 쪽에서 만든 v2 패키지를 C가 읽을 수 있는지 보는 것이 목적이다.  v2는
 * per-bucket 섹션(EXECUTION_PLAN 등)이 반복되고 그 flags에 bucket이 들어 있어
 * v1 규칙으로는 거부된다. */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "campp_runtime/status_code.h"
#include "internal/runtime_model.h"

int main(int argc, char **argv)
{
    CamppRuntimeModel model;
    CamppStatus status;
    int index;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.camppmodel> [...]\n", argv[0]);
        return 2;
    }
    for (index = 1; index < argc; ++index) {
        memset(&model, 0, sizeof(model));
        status = campp_runtime_model_load_package(argv[index], &model);
        if (status != CAMPP_STATUS_OK) {
            printf("FAIL %s: %s\n", argv[index], campp_status_name(status));
            return 1;
        }
        printf("OK   %s  bucket=%" PRIu32 " operators=%" PRIu32
               " tensors=%" PRIu32 "\n",
               argv[index], model.bucket_frames, model.operator_count,
               model.tensor_count);
        campp_runtime_model_release(&model);
    }
    return 0;
}
