# ACE — 반려동물 사진 탐지와 품종 분류

`project_05_pet_detection`의 기본 요구사항을 바탕으로 만들었습니다.
**`source/main.py`를 실행하면 `data/images`의 사진에서 개·고양이 수와 품종 후보를 분석합니다.**

## 기본 도구와 적용 범위

| 기본 자료의 요소 | ACE 적용 |
|---|---|
| Python 3.11 / CPU / 웹캠 불필요 | 동일 |
| Ultralytics YOLO11n | 개·고양이 위치와 수 탐지 |
| OpenCV | 한글 경로 이미지 읽기·결과 이미지 저장 |
| 입력 → 분석 → 새 파일 저장 | 동일, 원본 수정 없음 |
| `*_pets.jpg`, `pet_counts.csv` | 동일 |
| 수작업 정답표·테스트 시나리오 | `data/ground_truth.csv`, `test_cases.csv` |
| 품종 식별 없음 | 요청에 따라 사전 학습 ResNet50 품종 분류 추가 |

품종 분류에는 PyTorch·torchvision이, 전처리와 라벨 표시에는 Pillow가 추가로 필요합니다. 설치 목록은 `requirements.txt`에 있습니다.
기본 패키지의 상위 `shared/common.py`가 제공되지 않았으므로 해당 기능을 기존 소스 3개에 넣었습니다. 모델은 프로젝트 안 `models/`에 준비합니다.

## 구조

```text
ace-pet-vision/
├─ data/
│  ├─ images/             # 분석할 사진 20장
│  ├─ ground_truth.csv    # 사람이 직접 센 정답 (처음에는 빈 칸)
│  ├─ metadata.csv        # 포함 사진의 품종·출처
│  └─ SOURCES.md          # 사진 출처·라이선스
├─ models/                # 학습된 모델 파일
├─ outputs/               # 결과 이미지·CSV
├─ source/
│  ├─ main.py             # 입력·전체 실행·개수 비교
│  ├─ pet_model.py        # 모델 준비·탐지·품종 분류
│  └─ visualizer.py       # 박스·품종·마릿수 표시와 저장
├─ tests/                 # 자동 검증
├─ test_cases.csv         # 시나리오별 팀 검증 기록
├─ requirements.txt
├─ setup.bat
├─ setup_env.py
├─ run.bat
├─ .gitignore
└─ README.md
```

## 설치와 실행

1. 64비트 Python 3.11을 설치합니다.
2. `setup.bat`을 실행해 프로젝트 `.venv`와 라이브러리를 설치합니다.
3. IntelliJ의 Python SDK와 Module SDK를 `.venv\Scripts\python.exe`로 선택합니다.
4. **`source/main.py` 우클릭 → Run 'main'**, 또는 `run.bat` 실행.

이미 설치했다면 환경을 다시 만들 필요가 없습니다. 기본 자료의 공용 `../.venv_yolo`만 있는 경우에도 `run.bat`이 사용할 수 있습니다. 품종 분류용 패키지가 없다면 이 프로젝트의 `setup.bat`으로 설치하세요.

```powershell
.\run.bat
# 사진 한 장
.\run.bat --input data/images/03_dog_beagle.jpg
# 탐지 기준 비교 실습
.\run.bat --conf 0.5
```

입력은 기본적으로 `data/images` 전체이며, `--limit 5`로 처음 5장만 처리할 수 있습니다. 이전 `--samples` 명령도 지원합니다.
사진 목록이 비었거나 형식이 맞지 않으면 모델 실행 전에 안내합니다. 손상 사진은 오류로 기록하고 다음 사진을 처리합니다.
모델 파일은 첫 실행에 자동으로 받으며 이미 있으면 검증 후 재사용합니다. **Oxford 원본 사진·정답 자료는 내려받을 필요가 없습니다.**

## 결과

`outputs/` 바로 아래에 저장합니다.

| 파일 | 내용 |
|---|---|
| `03_dog_beagle_pets.jpg` 등 | 원본과 다른 이름의 결과 사진. 위쪽에 Dogs/Cats 수, 동물별 박스·품종 후보 표시 |
| `pet_counts.csv` | 사진별 `cat`, `dog`, 실제 개수와 절대오차, 처리 상태 |
| `pet_breeds.csv` | 동물별 종류·품종 후보·점수·박스 좌표 |
| `run_config.json`, `errors.log` | 실행 설정과 오류 기록 |

기본 자료처럼 같은 이름의 결과는 재실행 시 갱신합니다. 비교할 이전 결과는 다른 폴더에 복사하거나 `--output outputs/experiment_02`를 사용하세요.
이전 입력의 결과 사진이 남을 수 있으므로 **현재 실행의 CSV에 적힌 결과 파일**을 기준으로 확인합니다. `run_config.json`의 `completed`는 실행 완료 여부입니다.
같은 폴더에 `pet.jpg`와 `pet.png`가 있으면 결과 이름에 순번을 붙여 구분합니다.
이전 버전의 `run_날짜_시간/` 폴더는 자동 삭제하지 않습니다.

마릿수는 모델의 검출 수입니다. 중복 탐지·미탐이 있을 수 있으며 품종 후보 점수는 실제 정확도가 아닙니다.
낮은 점수·종 불일치는 `unknown/review`로 보류합니다. 지원 품종은 37개이며 믹스견·지원 밖 품종은 잘못 분류할 수 있습니다.

## 정답표와 검증

1. 포함된 사진 또는 사용 허가를 받은 사진 20장을 준비합니다.
2. `data/ground_truth.csv`에서 `filename`을 실제 입력 파일명과 맞춥니다. 하위 폴더는 `folder/photo.jpg`처럼 입력 폴더 기준으로 적습니다.
3. 사람이 사진을 확인하고 `actual_cat`, `actual_dog`에 진짜 개수만 입력합니다. 대상이 없으면 `0`, 아직 확인하지 않았으면 빈 칸으로 둡니다.
4. 실행하면 `pet_counts.csv`에 `cat_error`, `dog_error` = **|예측 수 − 실제 수|**가 생성됩니다. 미작성 정답·처리 실패는 오차를 계산하지 않습니다.
5. `test_cases.csv`의 실제 입력 파일·실행 결과·통과 여부·메모를 팀에서 작성합니다.

정답은 파일명이나 AI 예측으로 채우지 않았습니다. `metadata.csv`의 품종 표기는 출처 기반 참고 라벨이지 마릿수 정답이 아닙니다.
개만·고양이만·둘 다·동물 없음·가림·잘못된 입력 조건을 확인하세요. 포함 사진 20장이 이 조건을 모두 충족한다고 가정하지 않습니다.
정밀도 90% 같은 목표를 미리 주장하지 않고 실패 사례 3개와 원인을 설명합니다. 개수 일치와 박스 위치 정확도는 다릅니다.

## 팀원 실습과 제출

- 통합: `main.py`의 입력→탐지→분류→저장 설명
- AI: `pet_model.py`의 `--conf 0.25 / 0.5` 비교, 품종 입력 크기·정규화 설명
- 표시·저장: `visualizer.py`의 색상·글자 크기 변경
- 검증: 수작업 정답표, 개수 오차와 실패 사례 기록
- 제출: 소스, 결과 사진 2개 이상, 정답표, 테스트 기록, 수정 사항과 3분 시연

각자 수정 1건과 함수 1개 설명을 준비합니다. 웹캠·영상·GUI·새 모델 학습은 구현 범위에 포함하지 않았습니다.

## 출처와 공유

- [YOLO11n](https://docs.ultralytics.com/models/yolo11/) / [Ultralytics 라이선스](https://www.ultralytics.com/license)
- [품종 분류 모델](https://huggingface.co/flaviodell/oxford-pets-resnet50): 커뮤니티 배포 ResNet50, MIT 표기. 고정 버전·해시는 `pet_model.py` 참조.
- [포함 사진 출처·라이선스](data/SOURCES.md), [품종·출처 표](data/metadata.csv)

모델·출력·개인 환경은 Git에서 제외합니다. 촬영한 사진이나 교사가 허가한 자료는 공개 저장소에 무단으로 올리지 마세요. 개인 자료는 Git에서 제외되는 `data/my-photos`와 `--input`으로 사용할 수 있습니다.
자동 검사: `.venv\Scripts\python.exe -m unittest discover -s tests -v`.
