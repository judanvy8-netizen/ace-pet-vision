# ACE — 반려동물 AI 프로젝트

**기존 AI 사용 → 추론 → 수기 정답 비교 → 문제 수정 → 재실행**을 반복하는 7명 팀 프로젝트입니다. YOLO11은 동물을 찾고, ResNet50은 Oxford-IIIT Pet 37품종 후보를 제시합니다.

## 코드 버전

| 파일 | 설명 |
|---|---|
| `source/main.py` | **선생님 원본. 수정 금지** |
| `source/main-1.py` | 최초 37품종 분류·결과 저장 |
| `source/main-2.py` | 중복 박스 제거·YOLO11s 추가 검사 |
| `source/main-3.py` | 종 충돌 검토·품종 보완·자동 평가 (**현재 버전**) |

1·2단계 당시 공통 코드는 `legacy_v1/`, `legacy_v2/`에 보관했습니다. 최신 `shared/` 코드와 섞어 실행하지 않습니다.

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
- [사진기록.csv](data/사진기록.csv): **한국어 기록표**. 품종·실제 마릿수는 확인된 기존 결과를 옮겼고, **허가·담당자 칸은 직접 작성**합니다.
- [ground_truth.csv](data/ground_truth.csv): 모델이 읽는 개·고양이 정답표.
- [metadata.csv](data/metadata.csv): 출처·라이선스·품종 표기 근거.
- [test_cases.csv](test_cases.csv): 실제/미검증 테스트 항목.

**중요:** 20장 모두 실제 사진에는 동물이 1마리씩 있습니다. 선생님 계획표에 적힌 ‘여러 대상’·‘대상 없음’은 **향후 확보할 사진의 계획**이지 현재 검증한 상황이 아닙니다. 05·07은 **파일명은 JPG로 통일했지만, 저장소 안의 원본 바이트가 PNG**입니다. 첫 `run.bat`가 이를 실제 JPEG로 변환하고 `metadata.csv`를 수정합니다. 변환 뒤 전체 20장을 재검증해야 합니다.

## 기록

- [개발기록.md](개발기록.md): 단계별 문제와 수정 2~3개씩 요약
- [검증결과.md](검증결과.md): 과거 20장 실측·오류 분석
- [팀역할.md](팀역할.md) · [회의록.md](회의록.md) · [사진출처.md](사진출처.md)
- `outputs/`: 실행할 때 결과 이미지·`pet_counts.csv`·`pet_breeds.csv` 생성. 3단계는 `검증_상세.csv`·`검증_요약.md`도 생성.

**선생님 필수:** 7명 모두 **자신이 직접 수정한 코드 1건과 함수 설명 1개**를 남겨야 합니다. 역할 배정이 개인 구현 완료를 의미하지는 않습니다. 선택 기능은 교사와 협의합니다.
