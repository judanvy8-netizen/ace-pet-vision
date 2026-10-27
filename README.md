# ACE — 반려동물 사진기 및 품종 분류

사진을 입력해 개·고양이를 탐지하고, 품종 후보와 결과 이미지·CSV를 제공하는 고등학교 팀 프로젝트입니다.

## 1. 개발 환경

- IDE: IntelliJ IDEA (Python 플러그인)
- 운영체제: Windows
- Python: **3.11, 64비트**
- 가상환경: 프로젝트 내부 `.venv`
- 주요 라이브러리: PyTorch, torchvision, Ultralytics YOLO, OpenCV, NumPy, pandas, Pillow
- 버전 목록: [requirements.txt](requirements.txt)

## 2. 프로젝트 폴더 구조

```text
ace-pet-vision/
├─ .idea/                 # IntelliJ가 로컬에서 생성, Git 제외
├─ .venv/                 # setup.bat으로 로컬에서 생성, Git 제외
├─ input/                 # 입력 사진
│  ├─ images/             # 기본 입력 대상: 준비된 사진 20장
│  └─ out-of-scope/       # 지원 범위 밖 사진, 기본 입력에서 제외
├─ output/                # 처리 결과 이미지·CSV 저장 위치
├─ source/                # Python 프로그램 코드
│  └─ main.py            # 시작 파일: 현재 입력 사진 목록 확인
├─ models/                # 사전 학습 모델 저장 위치
├─ data/                  # 출처·품종 표, 검토 기록, 향후 정답표
│  ├─ smoke-test/
│  └─ out-of-scope/
├─ .gitignore
├─ requirements.txt
├─ setup.bat              # Windows 자동 환경 설정
├─ setup_env.py
├─ SETUP.md
└─ README.md
```

`input → source에서 처리 → output` 순서로 사용합니다. `models`에는 모델을, `data`에는 출처와 정답 정보를 둡니다. `output`과 `models`는 폴더 유지용 .gitkeep만 공유하고 실제 생성 결과·모델은 Git에서 제외합니다. `.idea`, `.venv`, `*.iml`은 각 컴퓨터의 설정이므로 GitHub에 올리지 않습니다.

## 3. 처음 실행하기

1. Python 3.11(64비트)을 설치하고 저장소를 내려받습니다.
2. 프로젝트 루트의 **setup.bat**을 실행합니다. 가상환경 생성과 CPU용 라이브러리 설치를 진행합니다.
3. 완료 후 IntelliJ에서 프로젝트 SDK로 **`.venv\\Scripts\\python.exe`**를 선택합니다.
4. **source/main.py**를 우클릭하고 **Run 'main'**을 누릅니다.
5. 준비된 입력 사진 20장의 목록이 출력되는지 확인합니다.

현재 main.py는 프로젝트 경로와 입력 목록을 확인하는 시작 파일입니다. **동물 탐지·품종 추론·결과 저장 기능은 아직 구현하지 않았습니다.** 현재 상태와 모델 검증 범위를 아래에 구분해 기록했습니다.

IntelliJ 하단 PowerShell Terminal에서 실행할 수도 있습니다.

```powershell
.\setup.bat
.\.venv\Scripts\python.exe source\main.py
```

상세 설정: [SETUP.md](SETUP.md). Clone/Pull만으로 설치가 자동 실행되지는 않습니다. 이미 설정했다면 setup.bat은 기존 .venv를 재사용합니다.

## 현재 상태

- 사진 입력·추론 흐름을 확인할 후보 사진 20장을 준비했습니다. 고양이 10장, 개 10장으로 서로 다른 20품종입니다.
- 기존 지원 품종 사진 4장을 보존하고 16장을 추가했습니다. Oxford-IIIT Pet의 37품종 목록에 없는 골든 리트리버는 원본과 출처를 `input/out-of-scope/`에 별도 보관했습니다.
- 사진 파일의 육안 확인, 디코딩, 크기·SHA-256 대조, 파일과 CSV의 일대일 대응, 중복 후보 검사를 수행했습니다.
- **프로그램 구현과 모델 추론·속도·정확도는 아직 검증하지 않았습니다.** 현재 실행 명령은 입력 목록 확인용이며, 추론 명령은 구현·검증 후 추가합니다.

## 사진과 출처

|자료|위치|
|---|---|
|지원 품종 사진 20장|[input/images](input/images)|
|사진별 품종·저자·출처·라이선스·해시|[metadata.csv](data/smoke-test/metadata.csv)|
|라이선스 근거와 품질 주의사항|[SOURCES.md](data/smoke-test/SOURCES.md)|
|사진 검토·실제 다운로드 주소|[photo-review.json](data/smoke-test/photo-review.json)|
|파일 검사 결과|[validation-results.json](data/smoke-test/validation-results.json)|
|수집·수정 기록|[REVIEW.md](data/smoke-test/REVIEW.md)|
|지원 범위 밖 원본 보관|[data/out-of-scope](data/out-of-scope)|

개별 사진 페이지에서 이미지 자체가 CC0 또는 저작권자에 의해 퍼블릭 도메인으로 공개된 자료만 채택했습니다. 원본 파일 9장과 Wikimedia 공식 축소본 11장이 포함되어 있으며, 축소 여부는 `metadata.csv`의 `modifications`에 기록했습니다. 외형만으로 품종을 추정하지 않았고, 출처의 품종 표기는 혈통의 독립적인 검증을 뜻하지 않습니다.

이 자료는 **입력·추론 확인용 후보 자료**입니다. 최종 정확도 평가에 사용하려면 모델의 학습 데이터와 중복되는지, 정답 라벨을 신뢰할 수 있는지 확인해야 합니다. 20장으로 37품종 전체의 성능을 검증할 수 없습니다.

## 구현 방향

1. 교사의 최종 지침에 따라 사전 학습 모델 사용과 추가 학습의 허용 범위를 확인합니다.
2. 선택한 모델의 라이선스와 출력 라벨을 확인하고 사진 파일 한 장으로 로컬 추론을 검증합니다.
3. 품종 후보 표시와 결과 저장을 연결합니다.

[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)은 고양이 12품종·개 25품종의 데이터셋입니다. 이번 사진 수집 묶음은 Oxford 데이터셋에서 사진을 추출한 것이 아닙니다. 개·고양이 구분과 품종 분류는 서로 다른 작업이며, 모든 품종·혼종의 혈통을 확정하거나 정확도를 보장하는 서비스로 설명하지 않습니다.

## 작업 기록

코드, 회의 결정, AI 프롬프트 요약, 수정 내용과 테스트 결과를 기록합니다. 공개 저장소에는 대화 원문, 불필요한 개인정보, 비밀번호나 API 키를 올리지 않습니다.
