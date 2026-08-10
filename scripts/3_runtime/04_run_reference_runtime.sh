#!/usr/bin/env bash
# Phase 3의 네 번째 실행 스크립트다.
#
# 역할:
# - weights.bin과 plan_98/298/498/998.bin을 차례로 선택한다.
# - ORT와 동일한 feature 입력으로 C Reference Runtime을 실행한다.
# - 중간 Tensor dump, execution trace, embedding을 저장한다.
# - 프로그램 종료 코드와 오류 로그를 확인한다.
#
# 원시 출력은 runs/runtime/c_reference에 저장한다.
