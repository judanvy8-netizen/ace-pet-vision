# ACE — 개·고양이 수와 품종 분석

**`source/main.py`를 실행하면 `data/images`의 모든 사진을 분석합니다.**
Oxford 원본 사진이나 정답 파일을 다운로드할 필요가 없습니다. 이미 학습된 모델 두 개를 사용합니다.

## 폴더 구조

```text
ace-pet-vision/
├─ data/
│  ├─ images/         # 분석할 사진 (현재 20장)
│  ├─ metadata.csv    # 포함된 사진의 품종·출처 정보
│  └─ SOURCES.md      # 사진 출처·라이선스
├─ models/            # 모델 자동 다운로드 위치
├─ outputs/           # 결과 사진·CSV
├─ source/
│  ├─ main.py         # 사진 읽기·전체 실행·마릿수 출력
│  ├─ pet_model.py    # 모델 준비·동물 탐지·품종 분류
│  └─ visualizer.py   # 마릿수·박스·품종 표시, 결과 저장
├─ tests/             # 코드 변경 후 자동 검증
├─ requirements.txt
├─ setup.bat
├─ setup_env.py
├─ run.bat
├─ .gitignore
└─ README.md
```

사진 이름은 `01_cat_abyssinian.jpg`, `02_dog_basset_hound.jpg`처럼 번호·종류·품종 순서입니다. `cat`은 고양이, `dog`는 개입니다.
파일명은 참고 정보이며 프로그램은 사진을 보고 판단합니다.

## 처음 한 번 설치

1. 64비트 Python 3.11을 설치합니다.
2. 프로젝트 폴더에서 `setup.bat`을 실행합니다. `Environment ready.`가 나오면 완료입니다.
3. IntelliJ의 Python SDK로 프로젝트 안의 `.venv\Scripts\python.exe`를 선택합니다.
4. 인터프리터 경고가 남으면 **Project Structure → Modules → Dependencies → Module SDK**에도 같은 환경을 선택합니다.

SDK 목록에 없다면 **Project Structure → SDKs → + → Python SDK → 기존 환경**에서 등록합니다. 메뉴 이름은 버전에 따라 다를 수 있습니다.
기존 가상환경과 모델 파일이 있으면 재사용합니다.

## 실행

**IntelliJ에서 `source/main.py`를 우클릭 → Run 'main'. 별도 실행 옵션은 필요 없습니다.**

또는 프로젝트 터미널에서:

```powershell
.\run.bat
```

사진을 추가하려면 `data/images`에 JPG·PNG 등을 넣고 다시 실행하세요. 첫 실행에는 학습된 모델 파일만 자동 다운로드합니다.
이전 명령 `.\run.bat --samples`도 같은 사진 폴더를 처리합니다.

## 무엇이 표시되나요?

터미널 표시 형식:

```text
[1/20] 03_dog_beagle.jpg: 개 1마리 / 고양이 0마리
  1. 개: Beagle / 후보 점수 53.3%
```

- 결과 사진 위쪽: `Dogs: 1 | Cats: 0`처럼 검출된 개·고양이 수.
- 동물마다: 네모 박스, 번호, 개/고양이 종류, 품종 후보와 점수.
- 품종 점수가 낮거나 동물 종류가 불일치하면: `review`로 표시하고 후보만 제공합니다.
- 동물을 찾지 못한 사진: 개·고양이 모두 0마리.
- 사진을 읽거나 처리하지 못한 경우: 오류로 기록하며 미탐과 구분합니다.

**마릿수는 모델이 검출한 수라서 중복 탐지·미탐이 있을 수 있습니다. 품종 점수도 실제 정답률을 뜻하지 않습니다.**
모델은 37품종 후보를 지원하며, 다른 품종이나 믹스견·믹스묘도 잘못 분류할 수 있습니다.

## 저장 위치

실행마다 `outputs/run_날짜_시간/`에 저장합니다.

| 항목 | 내용 |
|---|---|
| `images/` | 마릿수·박스·품종이 표시된 결과 사진 |
| `crops/` | 동물별로 잘라낸 사진 |
| `results.csv` | 동물 한 마리당 한 행: 종류·품종·후보 점수 |
| `image_summary.csv` | 사진별 `dog_count`·`cat_count`·처리 상태 |
| `run_config.json`, `errors.log` | 실행 설정과 오류 기록 |

원본 사진은 수정하지 않습니다. 결과 사진에는 위쪽 요약 여백을 추가하며 CSV의 박스 좌표는 EXIF 회전을 반영한 원본 사진 기준입니다.
예전 실행 결과는 그대로 남으므로 새로 생성된 폴더를 확인하세요.

## 선택 사항

```powershell
# 사진 한 장만 분석
.\run.bat --input data/images/03_dog_beagle.jpg

# 처음 5장만 분석 (기본값 0 = 전체)
.\run.bat --limit 5
```

실행 흐름은 `main.py → pet_model.py → visualizer.py`입니다.
사진·결과 경로는 `main.py` 상단에, 모델의 탐지·분류 기준은 `pet_model.py`에 있습니다.

## 모델·사진 출처

- 탐지: [Ultralytics YOLO11n](https://docs.ultralytics.com/models/yolo11/) — cat(15), dog(16). [라이선스](https://www.ultralytics.com/license).
- 품종: [flaviodell/oxford-pets-resnet50](https://huggingface.co/flaviodell/oxford-pets-resnet50) — Oxford 데이터로 학습된 커뮤니티 모델, MIT 표기. 학습에 사용된 사진을 직접 받을 필요는 없습니다. 다운로드 버전·해시는 `pet_model.py`에 고정되어 있습니다.
- 포함 사진: [출처·라이선스](data/SOURCES.md), [품종·출처 표](data/metadata.csv).

모델·결과·가상환경은 Git에서 제외합니다. 개인 사진은 Git에서 제외되는 `data/my-photos/`에 두고 `--input data/my-photos`로 실행할 수 있습니다.
과거에 내려받은 Oxford 자료는 사용하지 않습니다. 이미 있는 로컬 자료는 자동 삭제하지 않습니다.

코드 변경 후 검증: `.venv\Scripts\python.exe -m unittest discover -s tests -v`.
