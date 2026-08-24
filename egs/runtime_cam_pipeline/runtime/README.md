# Generated runtime assets

`prepare_runtime.py`가 이 디렉터리에 `campp_runtime`, Torch-free
`campp_fbank`, Python/NumPy-free `campp_speaker_verify`, `assets.json`, 버킷별
model 자산을 복사한다. 생성된 binary와 weight는 Git에 넣지 않는다.

`assets.json`은 단순 경로 목록이 아니라 실행 계약이다. 네이티브 검증기는
runtime/frontend/application과 선택 bucket 자산의 경로 및 SHA-256을 대조하고,
capability·입력 shape·실제 로드 bucket까지 확인한 후에만 score를 출력한다.

Apache-2.0 고지 보존을 위해 dependency LICENSE도 `runtime/licenses/`에 함께
복사된다.

기본 package 모드는 `.camppmodel` 네 개를 직접 로드한다. windowed 모드는
버킷별 plan, weights, schedule을 함께 로드한다.

준비가 끝난 뒤 등록·추론 CLI의 기본 경로는 이 디렉터리 안에서만 해결된다.
