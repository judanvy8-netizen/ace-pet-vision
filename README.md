# ACE — 반려동물 탐지·품종 분류

사진에서 개·고양이를 찾아 품종 후보를 표시하고 결과 이미지와 CSV를 저장합니다. Python 3.11을 사용합니다.

## 폴더와 파일

```text
ace-pet-vision/
├─ data/
│  ├─ images/             # 포함된 사진 20장
│  ├─ oxford-iiit-pet/    # 수업 데이터셋 (Git 제외)
│  ├─ metadata.csv        # 사진별 품종·출처 정보
│  └─ SOURCES.md          # 사진 출처·라이선스
├─ models/                # 모델 자동 다운로드 위치
├─ outputs/               # 실행 결과 이미지·CSV
├─ source/                # 프로그램 코드 (기존 구조 유지)
├─ tests/                 # 코드 변경 후 자동 검증
├─ .gitignore             # 대용량·개인 환경 업로드 제외
├─ requirements.txt       # 필요한 라이브러리
├─ setup.bat              # 처음 환경 설치
├─ setup_env.py           # setup.bat이 사용하는 설치 코드
├─ run.bat                # 프로그램 실행
└─ README.md              # 이 안내서
```

`data/images/`의 이름은 **번호_종류_품종**입니다.
예: `01_cat_abyssinian.jpg`, `02_dog_basset_hound.jpg`, `03_dog_beagle.jpg`.
`cat`은 고양이, `dog`는 개입니다. 번호는 정렬용이며 품종 클래스 번호가 아닙니다. 파일명은 참고 라벨이고 AI는 사진을 보고 품종을 추론합니다.

## 처음 한 번 설치

1. 64비트 Python 3.11을 설치합니다.
2. 프로젝트 폴더에서 `setup.bat`을 실행합니다. `Environment ready.`가 나오면 설치 완료입니다.
3. IntelliJ의 Python SDK를 프로젝트 안의 `.venv\Scripts\python.exe`로 선택합니다.
4. `No Python interpreter` 경고가 남으면 **Project Structure → Modules → Dependencies → Module SDK**에도 그 환경을 선택하고 Apply를 누릅니다.

SDK 목록에 없다면 **Project Structure → SDKs → + → Python SDK → 기존 환경**에서 등록합니다. 메뉴 이름은 IntelliJ 버전에 따라 다를 수 있습니다.
Git Pull만으로 설치되지는 않습니다. `.venv`는 각 PC에서 만들며, 설치와 IntelliJ 연결은 별개입니다. 이미 설치했다면 다시 만들 필요가 없습니다.

## 실행

IntelliJ 아래 터미널에서 프로젝트 폴더를 기준으로 실행합니다.

```powershell
# 포함된 사진 20장
.\run.bat --samples

# 수업용 Oxford 데이터셋: 처음 20장
.\run.bat

# 사진 한 장
.\run.bat --input data/images/03_dog_beagle.jpg
```

`source/main.py`를 IntelliJ에서 실행해도 됩니다. 기본 입력은 Oxford 데이터셋이며, 포함된 사진을 쓰려면 실행 옵션에 `--samples`를 넣습니다.
모델은 첫 실행 때 자동으로 다운로드합니다. 추가 학습은 하지 않습니다.

## Oxford 데이터셋 넣기

[공식 페이지](https://www.robots.ox.ac.uk/~vgg/data/pets/)의 `images.tar.gz`, `annotations.tar.gz`를 받아 아래처럼 압축을 풉니다.

```text
data/oxford-iiit-pet/
├─ images/       # Abyssinian_201.jpg 등 원본 사진
└─ annotations/  # test.txt, trainval.txt 등 정답 목록
```

이미 받은 파일은 다시 받지 않아도 됩니다. **Oxford 원본 이름은 정답 목록과 연결되므로 변경하지 않습니다.**
대용량 데이터셋은 자동으로 다운로드하지 않습니다. GitHub에서 `.gitkeep`만 보이는 것은 빈 폴더를 유지하기 위한 정상 상태입니다.

## 결과 보기

실행할 때마다 `outputs/run_날짜_시간/`이 생성됩니다.

| 결과 | 내용 |
|---|---|
| `images/` | 박스·품종이 표시된 사진 |
| `crops/` | 동물 부분만 자른 사진 |
| `results.csv` | 동물별 예측 품종·후보 점수 |
| `image_summary.csv` | 사진별 검출 수·처리 상태 |
| `dataset_comparison.csv` | Oxford 모드에서 정답과 비교 |
| `run_config.json`, `errors.log` | 실행 설정과 오류 기록 |

`ok`는 처리가 끝났다는 뜻이며 정답을 보장하지 않습니다. `no_detection`은 동물을 찾지 못한 경우입니다.
품종 점수가 낮거나 탐지·분류의 동물 종류가 다르면 `unknown/review`로 표시합니다. 후보 점수는 실제 정확도가 아닙니다.

## 코드 역할

| 파일 (`source/`) | 역할 |
|---|---|
| `main.py` | 입력 선택과 전체 실행 |
| `config.py` | 폴더 경로·기본 설정값 |
| `data_loader.py` | 사진 목록·이미지 읽기 |
| `pet_detector.py` | YOLO로 개·고양이 위치 찾기 |
| `breed_classifier.py` | 37품종 분류·보류 판정 |
| `pet_model.py` | 탐지 → 크롭 → 분류 연결 |
| `model_loader.py` | 모델 다운로드·파일 검증 |
| `visualizer.py` | 박스 표시·이미지·CSV 저장 |
| `oxford_data.py` | Oxford 정답 읽기·결과 비교 |

## 모델과 자료 출처

- 탐지: [Ultralytics YOLO11n](https://docs.ultralytics.com/models/yolo11/), COCO의 cat(15)·dog(16). [라이선스 안내](https://www.ultralytics.com/license).
- 품종: [flaviodell/oxford-pets-resnet50](https://huggingface.co/flaviodell/oxford-pets-resnet50), MIT 표기. Oxford 대학의 공식 모델이 아닌 커뮤니티 배포 모델입니다. 고정 버전은 `3253ad292cf0778f5b64fa5a9156f2bdb0a84a3b`이며 다운로드 해시는 `model_loader.py`에 있습니다.
- 포함 사진: [출처·라이선스](data/SOURCES.md), [사진별 품종·출처 표](data/metadata.csv).

공식 test 사진과 분류 모델 학습 데이터의 중복 여부는 미확인입니다. 정답 비교표를 독립적인 정확도 검증으로 해석하지 않습니다. 전체 Oxford 사진에 대한 평가는 아직 실행하지 않았습니다.

## 공유와 업데이트

코드·사진 20장·출처 기록은 GitHub에 올립니다. 실제 Oxford 데이터, 모델, 실행 결과, `.venv`, `.idea`는 제외합니다.
개인 사진은 Git에서 제외되는 `data/my-photos/`에 보관하고 `--input data/my-photos`로 실행할 수 있습니다.
예전 `datasets/oxford-iiit-pet`에 받은 자료는 `data/oxford-iiit-pet`으로 옮기세요. 이전 `output/` 결과는 그대로 남고 새 결과는 `outputs/`에 저장됩니다.
사진 이름 변경 전의 실행 결과도 그대로 남습니다. 다시 실행하면 새 이름으로 저장됩니다.

코드를 바꾼 뒤 검증하려면 다음을 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
