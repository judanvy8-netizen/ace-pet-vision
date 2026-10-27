# ACE 실행 코드

전체 실행 파일은 `main.py`입니다. IntelliJ에서 우클릭 → Run으로 실행하세요. 기존 예제로 실행하려면 `--samples` 옵션을 사용합니다.

| 파일 | 역할 |
|---|---|
| config.py | 프로젝트 공통 경로와 처리 장수·탐지·분류 기준 |
| data_loader.py | 사진 목록 찾기, 한글 경로 읽기, EXIF 회전 반영 |
| pet_detector.py | YOLO11n으로 개·고양이 위치 찾기 |
| breed_classifier.py | 크롭 이미지를 37품종으로 분류하고 낮은 점수·종 불일치 보류 |
| visualizer.py | 박스·품종 표시, 이미지·CSV 저장 |
| pet_model.py | 탐지 → 크롭 → 분류 연결 |
| model_loader.py | 고정 버전 모델 다운로드와 SHA256 검증 |
| oxford_data.py | Oxford 공식 정답 목록 읽기와 예측 비교 |
| main.py | 입력 선택, 실행·오류 처리, 결과 저장 호출 |
| examples/ | 데이터셋 확인·동물 탐지·품종 분류 예제 3개 |

팀원별 설명할 함수와 실습 항목은 [루트 README](../README.md)의 역할 표를 참고하세요.
입력 파일 위치는 [입력 준비 안내](../data/INPUTS_준비.md)에 있습니다.
인터프리터는 프로젝트 `.venv/Scripts/python.exe`를 선택합니다.
