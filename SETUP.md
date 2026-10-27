# IntelliJ + Python 3.11 기본 설정

가상환경 폴더를 공유하는 대신 `setup.bat`, `setup_env.py`, `requirements.txt`를 공유합니다. 팀원마다 자신의 컴퓨터에서 실행하면 프로젝트 안에 `.venv`가 만들어집니다.

## 처음 한 번

1. **Windows용 64비트 Python 3.11**을 설치합니다. 이미 설치되어 있다면 넘어갑니다. 설치 프로그램에서 Python Launcher 또는 `Add python.exe to PATH`를 활성화하세요.
2. IntelliJ의 **Get from VCS / Clone Repository**로 이 저장소를 내려받거나, GitHub ZIP을 내려받아 압축을 풉니다.
3. 프로젝트 폴더의 **`setup.bat`을 더블클릭**합니다. IntelliJ 하단 PowerShell Terminal에서는 `./setup.bat`을 실행해도 됩니다.
4. 인터넷으로 CPU용 PyTorch와 필요한 라이브러리를 설치합니다. 처음에는 수백 MB 이상을 내려받으므로 몇 분 이상 걸릴 수 있습니다. 마지막에 **`Environment ready.`**가 나오면 설치가 끝났습니다.
5. IntelliJ에서 프로젝트의 **`.venv\Scripts\python.exe`**를 Python SDK로 선택합니다.

GitHub에서 Clone/Pull만 한다고 컴퓨터에서 스크립트가 저절로 실행되지는 않습니다. `setup.bat`을 한 번 실행해야 합니다. 이 스크립트는 Python 자체를 설치하지 않으며, IntelliJ의 SDK 선택은 처음 한 번 직접 해야 합니다.

## IntelliJ에서 가상환경 선택

1. `File → Settings → Plugins`에서 JetBrains의 **Python** 플러그인을 설치/활성화합니다.
2. `File → Project Structure → SDKs → + → Add Python SDK from disk`로 들어갑니다.
3. **기존 환경**을 선택하고 이 프로젝트의 `.venv\Scripts\python.exe`를 지정합니다. 이미 스크립트가 만든 환경이므로 새로 생성하지 않습니다.
4. `Project Structure → Project → SDK`에서 방금 추가한 Python 환경을 선택합니다. 모듈 SDK가 별도로 설정되어 있다면 프로젝트 SDK를 상속하도록 맞춥니다.

버전에 따라 메뉴 이름이 조금 다릅니다. [JetBrains 공식 Python SDK 안내](https://www.jetbrains.com/help/idea/configuring-python-sdk.html)를 참고하세요.

## 자동으로 처리되는 일

- Python 3.11 및 64비트 여부 확인
- 프로젝트 내부 `.venv` 생성
- 기존 `.venv`가 올바른 환경이면 재사용
- CPU용 PyTorch, torchvision 및 `requirements.txt`의 라이브러리 설치
- 라이브러리 의존성 및 주요 라이브러리 로딩 확인
- IntelliJ에서 선택할 인터프리터 경로 출력

기존 `.venv`가 손상되었거나 다른 Python 버전이면 삭제하지 않고 안내 후 종료합니다. `.venv` 이름을 `.venv-backup` 등으로 바꾼 뒤 재실행하세요. 백업 폴더 역시 Git에 올리지 마세요.

`.venv`, 모델 파일, 개인 IntelliJ 설정, 실행 결과는 `.gitignore`로 제외합니다. **가상환경 폴더 자체는 GitHub에 올리지 않습니다.**

## 이후 사용

팀원이 새로 참여하면 같은 `setup.bat`을 실행하면 됩니다. `requirements.txt`가 바뀌면 다시 실행하세요. 같은 환경을 재사용하므로 프로젝트 코드를 지우거나 덮어쓰지 않습니다.

설치 없이 확인만 하려면 IntelliJ Terminal에서 다음을 실행합니다.

```powershell
.\.venv\Scripts\python.exe setup_env.py --check
```

인터넷 연결 없이 가상환경만 먼저 만들려면 다음을 실행합니다. 이 옵션은 라이브러리를 설치하지 않습니다.

```powershell
.\setup.bat --venv-only
```

이 파일들은 개발 환경을 준비합니다. 반려동물 사진을 추론하는 프로그램과 모델 다운로드는 별도 구현 단계입니다.

## 프로젝트 구조와 시작 파일

입력 사진은 `input/images/`, 결과는 `output/`, 코드는 `source/`, 모델은 `models/`, 출처·정답 관련 표는 `data/`에 둡니다. 기존 data/smoke-test/images의 20장은 input/images로 이동했습니다.

SDK 선택 후 `source/main.py`를 우클릭하고 **Run 'main'**을 누르세요. 현재 시작 파일은 사진 목록과 경로만 확인합니다. 모델 추론은 다음 구현 단계입니다.
