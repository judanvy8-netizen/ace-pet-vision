# ACE — 반려동물 사진 탐지·품종 분류

기본 제공 05번 프로젝트의 **입력 → YOLO 탐지 → 개수 집계 → 저장** 흐름에 품종 분류를 추가했습니다.
사진은 기존 `data/images`의 20장을 사용합니다. 대용량 압축파일의 사진은 이번 변경에서 넣지 않았습니다.

## 구조

```text
ace-pet-vision/
├─ data/
│  ├─ images/              # 기존 사진 20장
│  ├─ ground_truth.csv     # 사람이 작성할 개수 정답
│  ├─ metadata.csv
│  └─ SOURCES.md
├─ outputs/                # 결과 사진·CSV
├─ shared/
│  ├─ __init__.py          # shared를 Python 패키지로 표시
│  ├─ common.py            # 공통 모델 호출·사진 읽기·파일 저장
│  ├─ pet_model.py         # 모델 준비·품종 분류
│  ├─ visualizer.py        # 박스·품종·마릿수 표시
│  └─ models/              # 학습된 모델 (Git 제외)
├─ source/
│  └─ main.py              # 직접 실행하는 파일
├─ tests/
├─ test_cases.csv
├─ requirements.txt
├─ setup.bat
├─ setup_env.py
├─ run.bat
├─ .gitignore
└─ README.md
```

기본 자료의 `shared/common.py` 원본은 제공된 폴더에 없어, 기본 `main.py`에서 확인한 호출 규격에 맞춰 공통 함수를 구현했습니다. 원본 파일을 그대로 복사한 것으로 설명하지 않습니다.
기본 코드의 `yolo, image_files, read_image, save_image, save_csv`를 사용하고, 품종 분류·마릿수 표시·정답 비교만 추가합니다.
현재 저장소 안에 `shared`를 함께 두므로, `main.py`는 프로젝트 루트를 Python 경로에 추가합니다.

## 실행

처음에는 Python 3.11(64비트)을 설치하고 `setup.bat`을 실행합니다. IntelliJ의 Python SDK와 Module SDK는 `.venv\Scripts\python.exe`로 선택합니다.
이미 준비했다면 **`source/main.py` 우클릭 → Run 'main'** 또는 아래 명령만 실행하세요.

```powershell
.\run.bat
```

기본 입력은 `data/images`, 기본 처리 수는 처음 20장입니다. `--limit 0`은 전체입니다.
사진 한 장: `.\run.bat --input data/images/03_dog_beagle.jpg`.
탐지 기준 비교: `.\run.bat --conf 0.5`.
기존 `--samples` 명령도 같은 20장 사진을 사용합니다.

## 모델

- 탐지: `shared/models/yolo11n.pt`
- 품종: `shared/models/oxford-pets-resnet50.pth`

예전 프로젝트 `models/`에 같은 모델이 있으면 검증 후 새 위치로 복사해 재사용합니다. 기존 파일은 삭제하지 않습니다.
없으면 첫 실행 때 모델 파일만 자동 다운로드합니다. Oxford 원본 사진을 받을 필요는 없습니다.
Python 3.11·Ultralytics·OpenCV·CPU 실행을 사용하며 품종 분류에는 PyTorch·torchvision·Pillow를 추가 사용합니다.

## 결과

`outputs`를 펼치면 결과가 바로 보입니다.

```text
outputs/
├─ pet_counts.csv
├─ pet_breeds.csv
├─ 01_cat_abyssinian_pets.jpg
├─ 02_dog_basset_hound_pets.jpg
└─ ...
```

사진 위쪽에 Dogs/Cats 수, 동물마다 박스·품종 후보·점수를 표시합니다. 점수가 낮거나 종이 불일치하면 보류합니다.
`pet_counts.csv`에는 사진별 cat/dog 수, 정답과 개수 오차, 처리 상태가 기록됩니다.
`pet_breeds.csv`에는 동물별 품종 후보가 기록됩니다. 오류는 `errors.log`에서 확인합니다.

같은 이름의 결과는 다시 실행하면 갱신합니다. 이전 결과를 보관하려면 복사하거나 `--output outputs/experiment_02`로 별도 경로를 지정하세요.
현재 실행 결과는 CSV에 기록된 파일 기준으로 확인합니다. 이전 버전의 결과 폴더·설정 파일은 자동 삭제하지 않습니다.
원본 사진은 수정하지 않으며, 결과 이미지 위쪽에 마릿수 표시 여백을 추가합니다. 박스 좌표는 방향을 반영한 원본 사진 기준입니다.

## 정답과 실습

`data/ground_truth.csv`의 `actual_cat`, `actual_dog`에 **사람이 확인한 진짜 개수만** 입력하세요. 0마리는 `0`, 확인 전에는 빈 칸입니다.
실행 결과의 `cat_error`, `dog_error`는 |예측 수 − 실제 수|이며 미작성 정답이나 처리 실패에는 계산하지 않습니다.
`test_cases.csv`에 고양이·개·둘 다·동물 없음·가림·잘못된 입력의 결과를 기록하세요.
각자 수정 1건, 함수 1개 설명, 실패 사례 3개와 원인 분석을 준비합니다.
모델의 검출 수·품종은 정답을 보장하지 않습니다. 기존 사진에서도 중복 탐지·미탐이 있으며, 품종 후보 점수는 정확도가 아닙니다.

## 대용량 파일과 출처

압축파일(`*.tar.gz`, `*.tar`), `data/oxford/` 등의 원본 자료, 모델, outputs, 개인 환경은 Git에서 제외합니다.
`images (1).tar.gz`도 이 규칙에 포함됩니다. 이번에는 압축을 풀거나 기존 사진을 바꾸지 않았습니다.
포함된 공개 사진 20장의 이름·내용·출처는 유지합니다. 개인 촬영 자료는 공개 저장소에 무단 업로드하지 마세요.

- [YOLO11n](https://docs.ultralytics.com/models/yolo11/) / [라이선스](https://www.ultralytics.com/license)
- [품종 모델](https://huggingface.co/flaviodell/oxford-pets-resnet50): MIT 표기의 커뮤니티 배포 모델. 버전·검증 해시는 `shared/pet_model.py` 참조.
- [사진 출처](data/SOURCES.md), [품종·출처 표](data/metadata.csv)

자동 검사: `.venv\Scripts\python.exe -m unittest discover -s tests -v`.
