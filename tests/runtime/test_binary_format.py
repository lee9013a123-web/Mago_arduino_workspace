# Python writer와 C model loader가 같은 binary 규격을 해석하는지 검증할 파일이다.
#
# 작은 fixture binary를 만든 뒤 header, descriptor, offset, alignment를 왕복 확인한다.
# 잘못된 magic, version, section 크기와 checksum을 Runtime이 거부하는지도 검사한다.
