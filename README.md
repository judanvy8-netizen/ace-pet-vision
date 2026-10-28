# ACE — 반려동물 AI 프로젝트

**기존 AI 사용 → 추론 → 수기 정답 비교 → 문제 수정 → 재실행**을 반복하는 7명 팀 프로젝트입니다. YOLO11은 동물을 찾고, ResNet50은 Oxford-IIIT Pet 37품종 후보를 제시합니다.

## 코드 버전

| 파일 | 설명 |
|---|---|
| `source/main.py` | **선생님 원본. 수정 금지** |
| `source/main-1.py` | 최초 37품종 분류·결과 저장 |
| `source/main-2.py` | 중복 박스 제거·YOLO11s 추가 검사 |
| `source/main-3.py` | 종 충돌 검토·품종 보완·자동 평가 (**현재 버전**) |

1·2단계 당시 공통 코드는 `legacy_v1/`, `legacy_v2/`에 보관했습니다. **모델 파일 저장 경로만** 현재 `shared/models/`로 통일했으며, 과거 추론 로직은 유지합니다.

## 실행 방법 (Windows, Python 3.11)

1. 최초 1회 `setup.bat` 실행.
2. **최초에는 `run.bat`를 실행**해 원래 PNG였던 05·07번까지 실제 JPEG로 바꾼 뒤, 이후에는 `run.bat` 또는 IntelliJ에서 `source/main-3.py` 실행.
3. 단계별 비교를 할 때는 각각 별도의 `outputs` 폴더를 사용.

```powershell
.\.venv\Scripts\python.exe source\main-1.py --output outputs\1단계 --limit 0
.\.venv\Scripts\python.exe source\main-2.py --output outputs\2단계 --limit 0
.\.venv\Scripts\python.exe source\main-3.py --output outputs\3단계 --limit 0
```

## 사진과 수기 기록

- 사진 파일: `data/images/img_001.jpg` ~ `img_020.jpg` (번호만 표시).
- [ground_truth.csv](data/ground_truth.csv): **제목 행 없는 5열 수기 정답표**. 각 줄은 `파일명,동물 종류,고양이 수,개 수,짧은 장면 특징` 순서입니다. 예: `img_003.jpg,개,0,1,잔디 위 앉은 개 측면`. **마지막 설명 칸은 사람이 보는 메모이며 AI 예측이나 정확도 계산에 사용하지 않습니다.** 1~3단계 개선본은 이 형식과 기존 제목 포함 형식을 모두 읽습니다.
- [metadata.csv](data/metadata.csv): 출처·라이선스·품종 표기 근거.
- [test_cases.csv](test_cases.csv): 실제/미검증 테스트 항목.

**중요:** 20장 모두 실제 사진에는 동물이 1마리씩 있습니다. 선생님 계획표에 적힌 ‘여러 대상’·‘대상 없음’은 **향후 확보할 사진의 계획**이지 현재 검증한 상황이 아닙니다. 05·07은 **파일명은 JPG로 통일했지만, 저장소 안의 원본 바이트가 PNG**입니다. 첫 `run.bat`가 이를 실제 JPEG로 변환하고 `metadata.csv`를 수정합니다. 변환 뒤 전체 20장을 재검증해야 합니다.

## 기록

- [개발기록.md](개발기록.md): 단계별 문제와 수정 2~3개씩 요약
- [검증결과.md](검증결과.md): 과거 20장 실측·오류 분석
- [팀역할.md](팀역할.md) · [회의록.md](회의록.md) · [사진출처.md](사진출처.md)
- `outputs/`: 실행할 때 결과 이미지·`pet_counts.csv`·`pet_breeds.csv` 생성. 3단계는 `검증_상세.csv`·`검증_요약.md`도 생성.

**선생님 필수:** 7명 모두 **자신이 직접 수정한 코드 1건과 함수 설명 1개**를 남겨야 합니다. 역할 배정이 개인 구현 완료를 의미하지는 않습니다. 선택 기능은 교사와 협의합니다.

## 개·고양이 혼합 사진 4장 — 별도 테스트

- `data/mixed_samples/mix_01.webp` ~ `mix_04.webp`: 대화 캡처의 2×2 이미지에서 각 칸을 분리한 약 117×117픽셀 테스트용 사진. 원본 이미지가 아니며 AI 생성 여부와 저작권·사용 허가는 확인되지 않았습니다.
- `data/mixed_ground_truth.csv`: **제목 행 없는 5열 수기 마릿수 정답표**. 네 장 모두 고양이 1마리·개 1마리입니다. 품종 정답이나 모델 실행 결과를 뜻하지 않습니다.
- 원래 20장과 섞이지 않도록 별도 입력 폴더에 보관합니다. 이 샘플은 실제 모델 추론이 **아직 미검증**이며 낮은 해상도 때문에 원본보다 탐지가 어려울 수 있습니다.
- 실제 모델로 실행하려면 Windows에서 아래 명령을 사용하고 각 사진의 두 개체에 모두 박스가 그려졌는지 결과 이미지로 직접 확인합니다.

```powershell
.\.venv\Scripts\python.exe source\main-3.py --input data\mixed_samples --truth data\mixed_ground_truth.csv --output outputs\mixed_samples --limit 0
```

- 결과: `outputs/mixed_samples/pet_counts.csv`, `pet_breeds.csv`, `mix_01_pets.jpg` ~ `mix_04_pets.jpg`. 개·고양이 중 한 종류만 검출되면 현재 추가 탐지 단계가 자동 실행되지 않을 수 있어 검토가 필요합니다.
- `tests/test_mixed_pets.py`: 모델 다운로드 없이 입력 파일·정답표 및 개/고양이 2개 박스의 분리 처리를 모의 검증합니다. 이 모의 테스트 통과를 실제 YOLO/ResNet50 정확도로 해석하면 안 됩니다.
