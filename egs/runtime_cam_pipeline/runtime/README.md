# Generated runtime assets

`prepare_runtime.py`가 이 디렉터리에 `campp_runtime`, Torch-free
`campp_fbank`, `assets.json`, 버킷별 model 자산을 복사한다. 생성된 binary와
weight는 Git에 넣지 않는다.

Apache-2.0 고지 보존을 위해 dependency LICENSE도 `runtime/licenses/`에 함께
복사된다.

기본 package 모드는 `.camppmodel` 네 개를 직접 로드한다. windowed 모드는
버킷별 plan, weights, schedule을 함께 로드한다.

준비가 끝난 뒤 등록·추론 CLI의 기본 경로는 이 디렉터리 안에서만 해결된다.
