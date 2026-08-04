# Reference outputs

이 폴더에는 현재 CAM++ PyTorch FP32 checkpoint로 생성한 기준 embedding을 저장한다.

권장 파일:

```text
embeddings_fp32.tsv      input_id와 embedding 파일 연결
embeddings_fp32/*.npy    입력별 float32 embedding
scores_fp32.tsv          trials.tsv 순서의 cosine score
metrics_fp32.json        EER, MinDCF와 실행 metadata
```

수치 reference는 checkpoint 경로와 실제 inference 명령을 확인한 후 생성해야 한다. 임의 값을 만들지 않는다.
