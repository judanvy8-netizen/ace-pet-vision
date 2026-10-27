# ACE — 반려동물 탐지 및 Oxford 37품종 분류

Python 3.11 / IntelliJ IDEA 프로젝트입니다. YOLO11n으로 개·고양이를 찾고, 잘라낸 영역을 Oxford-IIIT Pet으로 학습된 ResNet50에 전달합니다. 별도 학습 없이 품종 후보와 결과 이미지·CSV를 저장합니다.

## 1. 프로젝트 구조

```text
ace-pet-vision/
├─ data/
│  ├─ images/                 # 기존 예제 사진 20장
│  ├─ oxford-iiit-pet/        # 공식 images/ + annotations/, 실제 데이터는 Git 제외
│  ├─ smoke-test/             # 예제 사진의 출처·품종 기록
│  └─ INPUTS_준비.md           # 사진·정답 파일 준비 안내
├─ models/                    # 다운로드한 모델, Git 제외
├─ outputs/                   # 결과 이미지·CSV, Git 제외
├─ source/
│  ├─ main.py                 # 전체 실행
│  ├─ config.py               # 공통 경로·기본 설정값
│  ├─ data_loader.py          # 이미지 읽기
│  ├─ pet_detector.py         # 개·고양이 탐지
│  ├─ breed_classifier.py     # 품종 분류
│  ├─ visualizer.py           # 결과 표시·저장
│  ├─ pet_model.py            # 탐지 → 크롭 → 분류 연결
│  ├─ model_loader.py         # 모델 다운로드·검증
│  ├─ oxford_data.py          # 공식 정답 읽기·비교
│  └─ examples/               # 실행 예제 3개
├─ tests/                     # 자동 검증 코드
├─ .gitignore
├─ PROJECT.md                 # 프로젝트 목적·진행 상태
├─ README.md                  # 구조·역할·실행 안내
├─ requirements.txt
├─ setup.bat                  # 가상환경·라이브러리 설치
├─ setup_env.py
├─ run.bat                    # 프로그램 실행
├─ test_cases.csv             # 예제 사진 20장의 확인 기준
├─ SETUP.md
└─ MODEL.md
```

`.venv`, `.idea`는 각 컴퓨터에서 생성됩니다. 실제 데이터셋, 모델, 실행 결과는 `.gitignore`로 제외합니다. 각 폴더의 `.gitkeep`은 빈 폴더 유지용 파일입니다. `out-of-scope` 폴더와 골든 리트리버 사진은 제거했으며 기본 예제 사진 20장은 유지합니다.

## 역할별 코드와 실습

LOOK의 역할별 모듈 구성과 참고 화면의 데이터·모델·결과 분리를 참고했습니다. 수업 화면에 맞춰 data / models / outputs / source로 구성하고, 기능별로 구현을 나눴습니다. 팀원 이름은 실제 역할 배정 후 적으세요.

| 역할 | 담당 파일 | 설명할 함수·설정 |
|---|---|---|
| 통합·실행 | `source/main.py`, `source/pet_model.py` | `main`, `process_images`, `PetModel.predict`: 전체 흐름 연결 |
| 입력 데이터 | `source/data_loader.py`, `source/oxford_data.py` | `read_image`, `find_images`, `load_split`: 사진·정답 읽기 |
| 동물 탐지 | `source/pet_detector.py` | `PetDetector.detect`: 개·고양이 위치 찾기 |
| 품종 분류 | `source/breed_classifier.py` | `classify`, `choose_breed`: 품종 후보와 보류 판정 |
| 결과 화면 | `source/visualizer.py` | `annotate`: 박스·라벨 표시, `DOG_COLOR`·`CAT_COLOR` 수정 |
| 결과 저장·검증 | `source/visualizer.py`, `source/oxford_data.py` | `save_image`, `write_csv`, `compare_predictions` |
| 공통 설정·모델 준비 | `source/config.py`, `source/model_loader.py` | 경로·기본값, `prepare_models`: 모델 다운로드·검증 |

실행 흐름: `data_loader → pet_detector → 동물별 크롭 → breed_classifier → visualizer`.
`main.py`가 사진별 처리를 진행하고 `pet_model.py`가 탐지·분류를 연결합니다.

처음 실습할 때는 `config.py`의 `IMAGE_LIMIT`(처리 장수), `DETECTION_CONFIDENCE`(탐지 기준), `BREED_CONFIDENCE`(품종 보류 기준) 중 하나만 바꿔 결과를 비교하세요. 명령줄 옵션을 직접 주면 해당 값이 우선합니다. 모델 버전·검증 해시는 임의로 바꾸지 않습니다.

입력 준비: [INPUTS_준비.md](data/INPUTS_준비.md). `source`의 각 파일 맨 위에도 역할과 설명할 함수를 적었습니다.

## 2. 환경 준비

1. Python 3.11(64비트)을 설치합니다.
2. `setup.bat`을 실행합니다. 기존 `.venv`가 있으면 재사용합니다.
3. IntelliJ의 Python SDK로 `.venv\Scripts\python.exe`를 선택합니다.

메뉴 안내: [SETUP.md](SETUP.md).

## 3. 수업용 Oxford 데이터셋 준비

[Oxford-IIIT Pet 공식 페이지](https://www.robots.ox.ac.uk/~vgg/data/pets/)에서 `images.tar.gz`와 `annotations.tar.gz`를 받습니다. 이미 팀원이 받고 있다면 다시 받을 필요 없이 해당 파일을 사용합니다.

두 압축 파일을 **data/oxford-iiit-pet** 안에 풀어 아래 경로가 되게 하세요. 압축 프로그램이 `.tar`만 만들었다면 그 파일도 한 번 더 풀어야 합니다.

```text
data/oxford-iiit-pet/images/Abyssinian_201.jpg
data/oxford-iiit-pet/annotations/test.txt
data/oxford-iiit-pet/annotations/trainval.txt
```

원본 압축 파일도 `data/oxford-iiit-pet/` 안에 보관합니다. `data/oxford-iiit-pet` 내용과 두 압축 파일명은 Git에서 제외합니다. 데이터셋은 모델 파일이 아니며, 수업에서 지정한 사진·정답 자료로 사용합니다. 프로그램은 대용량 데이터셋을 자동 다운로드하지 않습니다.

## 4. IntelliJ에서 실행

프로젝트 루트에서 `run.bat`을 실행해도 됩니다. 첫 설치가 안 되어 있으면 `setup.bat`을 먼저 실행합니다.

```powershell
.\run.bat --samples   # 기존 예제 사진
.\run.bat             # 공식 Oxford 사진
```

`test_cases.csv`는 기존 예제 20장의 파일 경로·종·품종 확인 기준이며 프로그램이 자동으로 읽는 설정 파일은 아닙니다. 공식 데이터셋의 정답은 `annotations/test.txt`를 사용합니다.


**source/main.py → 우클릭 → Run 'main'**으로 실행합니다.

- 기본값: 공식 `annotations/test.txt`에 있는 사진 중 **처음 20장**을 처리합니다.
- 원본 데이터셋이 없으면 필요한 경로를 안내하고 종료합니다. 임의의 예제로 조용히 바꾸지 않습니다.
- 첫 실행에는 모델 두 개(약 101MB 합계)를 자동 다운로드합니다. 이후 같은 파일을 검증하고 재사용합니다.
- 결과 위치: `outputs/run_날짜_시간/`.

IntelliJ 하단 PowerShell에서도 실행할 수 있습니다.

```powershell
# 수업 데이터셋: test 목록의 처음 20장
.\.venv\Scripts\python.exe source\main.py

# 데이터셋 다운로드 전, 기존 예제 20장으로 실행 확인
.\.venv\Scripts\python.exe source\main.py --samples

# 공식 test 목록 전체 실행: CPU에서는 오래 걸릴 수 있음
.\.venv\Scripts\python.exe source\main.py --limit 0

# 직접 준비한 사진 한 장
.\.venv\Scripts\python.exe source\main.py --input data/images/beagle.jpg

# 모델만 미리 다운로드
.\.venv\Scripts\python.exe source\main.py --prepare-models
```

옵션: `--dataset 다른폴더`, `--split test|trainval`, `--limit 20`, `--conf 0.25`, `--breed-conf 0.5`, `--output 다른결과폴더`. 직접 지정한 상대 경로도 프로젝트 루트 기준으로 해석합니다. `--samples`와 `--input`은 동시에 쓰지 않습니다.

## 5. 짧은 샘플 코드

각 파일을 IntelliJ에서 우클릭하여 실행하세요.

| 파일 | 실습 내용 |
|---|---|
| [01_check_dataset.py](source/examples/01_check_dataset.py) | 공식 데이터셋 사진 경로·정답 5개 확인. 원본 데이터셋 필요 |
| [02_detect_pet.py](source/examples/02_detect_pet.py) | 예제 비글 사진에 YOLO 박스 표시, outputs/examples/detection.png 저장 |
| [03_predict_one.py](source/examples/03_predict_one.py) | 예제 비글 사진의 동물 종류·품종 상위 3개 출력 |

2번과 3번은 기존 예제 사진을 사용하므로 대용량 데이터셋이 없어도 실행됩니다. 다른 사진은 코드의 `photo_path`를 바꾸면 됩니다.

## 6. 결과 파일과 읽는 방법

| 파일/폴더 | 내용 |
|---|---|
| images/ | 동물 박스와 영문 품종 라벨. 미탐 사진도 저장 |
| crops/ | 검출된 개체별 크롭 |
| results.csv | 동물 한 마리당 한 행, 품종 후보·점수·박스·상위 3개 |
| image_summary.csv | 사진 한 장당 한 행, ok/no_detection/error와 처리시간 |
| dataset_comparison.csv | Oxford 모드에서 정답 라벨과 예측 비교 |
| run_config.json | 입력 모드, split, 처리 개수, 임계값, 모델 버전, 완료 여부 |
| errors.log | 손상 사진·다운로드 등 오류 기록 |

기본 품종 점수 0.5 미만이거나 분류 결과의 동물 종류가 YOLO 결과와 다르면 `breed=unknown`, `status=review`로 보류합니다. 후보 자체는 `candidate_breed`와 `top3_json`에 남깁니다. 점수는 전체 37클래스 softmax 값이며 종별 재정규화를 하지 않습니다. **이 점수는 실제 정확도나 보장된 확률과 같지 않습니다.** 미지원 품종도 높은 점수로 오분류할 수 있습니다.

원본 사진은 수정하지 않습니다. EXIF 방향을 반영한 좌표로 탐지·크롭합니다. 반복 실행마다 새 폴더를 만들어 이전 결과와 비교할 수 있습니다. 이미지 이름에 원래 확장자를 남긴 뒤 `.png`를 붙여 같은 이름의 JPG/PNG가 충돌하지 않습니다. 결과 CSV는 Excel용 UTF-8 BOM입니다. 처리시간 합계는 사진당 한 행인 image_summary.csv를 사용하세요.

정답 비교에서는 정답을 보지 않고 **가장 큰 탐지 박스**를 대표 개체로 선택합니다. 여러 동물이 검출된 사진은 `multiple_animals_review=True`로 표시하므로 직접 확인하세요. 정답 품종 이름은 공식 이미지 ID에서 읽고, 모델 출력 인덱스와 공식 클래스 ID가 같다고 가정하지 않습니다. 공식 XML의 박스는 머리 영역이므로 전신 YOLO 박스와 비교한 IoU 탐지 정확도를 주장하지 않습니다.

분류 모델 제공자의 학습 이미지 목록·분할을 독립적으로 확인하지 못했습니다. 공식 test 사진과 모델 학습 데이터가 겹칠 수 있으므로 **비교표의 일치 수는 기능 확인용이며 독립적인 일반화 성능 검증이 아닙니다.** 현재 실행한 일부 사진의 결과를 37품종 전체 정확도로 설명하지 마세요.

## 7. 검증 및 자료

- Python 3.11 CPU에서 기존 예제 20장 처리 완료: 이미지 처리 오류 0건, 동물 검출 사진 18장, 미탐 사진 2장. 중복 탐지도 있어 검출 정확도를 의미하지 않습니다.
- 실제 YOLO 탐지 및 한 장 품종 예제 실행, 결과 이미지 표시 확인. 모듈 분리 후 기존 예제 1장으로 전체 흐름 재확인.
- 역할별 모듈 분리 후에도 회귀 테스트 6개 통과: 보류·종 불일치, 다운로드 검증, 다중 개체, 미탐, 손상 파일, 공식 정답 형식·경로 검증 등.
- 공식 test 정답 파일 3,669행과 37품종 이름을 확인했습니다. **원본 데이터셋 전체 사진에 대한 추론은 아직 실행하지 않았습니다.**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

사용 모델·버전·라이선스: [MODEL.md](MODEL.md). 기존 예제의 [출처와 라이선스](data/smoke-test/SOURCES.md), [품종 표](data/smoke-test/metadata.csv), [수집 기록](data/smoke-test/REVIEW.md)은 유지합니다.

## 기존 폴더에서 업데이트한 팀원

Git Pull을 하면 저장소의 예제 사진은 `input/images`에서 `data/images`로 이동합니다.
이미 내려받은 Oxford 데이터셋은 로컬에서 `datasets/oxford-iiit-pet` → `data/oxford-iiit-pet`으로 옮기세요. 다시 다운로드할 필요는 없습니다.
이전 `output/` 결과는 자동으로 삭제하거나 옮기지 않으며 새 실행부터 `outputs/`에 저장합니다. 가상환경과 `models/`는 그대로 사용합니다.
