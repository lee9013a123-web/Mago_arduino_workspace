# Bucket별 execution plan

다음 생성 파일이 위치할 폴더다.

- `plan_98.bin`: 1초/98-frame 입력
- `plan_298.bin`: 3초/298-frame 입력
- `plan_498.bin`: 5초/498-frame 입력
- `plan_998.bin`: 10초/998-frame 입력

각 plan에는 Tensor descriptor, Operator table, attribute section과 bucket 정보가 들어간다.
