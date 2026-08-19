# QConv address-v2 standalone validation

Address provider는 현재 V4 runtime binary와 별도로 빌드한다.

```bash
bash scripts/4_profill/optimization/qconv_address_v2/01_build_and_test.sh
```

이 명령은 MAC, requant, store 또는 V4 dispatch source를 링크하지 않는다.
독립 pointer/border/storage/schedule 검증을 통과한 뒤 adapter를 통해 통합한다.
