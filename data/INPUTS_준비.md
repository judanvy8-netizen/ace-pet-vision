# ACE 입력 데이터 준비

## 1. 다운로드 없이 먼저 확인

`data/images/`에 기존 예제 사진 20장이 있습니다. 프로젝트 루트의 PowerShell에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe source\main.py --samples
```

모델은 첫 실행 때 자동으로 받습니다. 예제 사진의 출처와 품종 기록은 `data/smoke-test/`에 있습니다.

## 2. 수업용 Oxford 데이터셋

[공식 페이지](https://www.robots.ox.ac.uk/~vgg/data/pets/)의 `images.tar.gz`와 `annotations.tar.gz`를 풀어 다음처럼 배치합니다. 이미 받은 파일은 다시 받을 필요가 없습니다.

```text
data/oxford-iiit-pet/
├─ images/
│  ├─ Abyssinian_201.jpg
│  └─ ...
└─ annotations/
   ├─ test.txt
   ├─ trainval.txt
   └─ ...
```

`test.txt`와 `trainval.txt`가 공식 사진 목록과 정답 정보를 담고 있습니다. 별도 `ground_truth.csv`를 만들 필요는 없습니다. 기본 실행은 `test.txt`의 처음 20장을 읽습니다. 모델의 품종 인덱스와 공식 클래스 번호가 같다고 가정하지 않고 사진 ID의 품종 이름을 대응시킵니다.

```powershell
.\.venv\Scripts\python.exe source\main.py
```

`source/examples/01_check_dataset.py`는 데이터셋 경로와 정답 5개를 확인하는 예제입니다. 원본 사진이 준비되어 있어야 합니다.

## 3. 직접 찍은 사진

사진은 로컬 `data/my-photos/`에 넣으면 Git에서 제외됩니다. 한 장이나 폴더를 지정할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe source\main.py --input data/my-photos
```

개인 사진은 Git에 올라가는 예제 폴더 `data/images/`와 구분해서 보관하세요.

## 4. Git으로 공유되는 것

- 공유: 코드, 설명서, 기존 예제 20장과 출처 기록, 빈 폴더용 `.gitkeep`.
- 제외: `data/oxford-iiit-pet/`의 실제 데이터, `models/`의 모델 파일, `outputs/`의 실행 결과, `.venv/`, `.idea/`.

Git으로 제외된 파일은 팀원 PC에 자동으로 전달되지 않습니다. 각 PC에 데이터셋을 준비하고 `setup.bat`으로 가상환경을 생성합니다.

## 5. 결과 확인

`outputs/run_날짜_시간/`에 결과 이미지, 개체별 `results.csv`, 사진별 `image_summary.csv`를 저장합니다. Oxford 모드에서는 `dataset_comparison.csv`도 생성합니다.
학습 데이터와 공식 test 분할이 겹칠 수 있으므로 비교표를 독립적인 정확도 검증으로 사용하지 않습니다.
