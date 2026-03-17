# 꽃 조절

웹캠으로 손가락 펼침 정도를 인식해서 꽃 영상의 개화 정도를 실시간으로 조절하는 Python 프로젝트입니다.  
`MediaPipe`로 손 랜드마크를 추적하고, `OpenCV`로 카메라 화면과 꽃 영상을 합성합니다.

## 포함된 파일

- `hand_joint_recognition.py`: 메인 실행 코드
- `assets/flower_input.mp4`: 기본 꽃 영상
- `models/hand_landmarker.task`: 손 랜드마크 모델
- `run.bat`: Windows에서 바로 실행하기 위한 배치 파일

## 필요한 환경

- Windows PC
- 웹캠
- Python 3.x

`mediapipe` 설치가 잘 안 되면 Python 3.11 환경에서 시도하는 것이 가장 안정적입니다.

## 설치

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 실행

가장 간단한 방법:

```powershell
.\run.bat
```

직접 실행:

```powershell
py hand_joint_recognition.py
```

옵션 예시:

```powershell
py hand_joint_recognition.py --camera 0 --flower-video "assets\flower_input.mp4" --model "models\hand_landmarker.task" --width 1280 --height 720
```

종료는 `q` 또는 `Esc` 키로 할 수 있습니다.

## 폴더 구조

```text
.
├─ assets/
│  └─ flower_input.mp4
├─ models/
│  └─ hand_landmarker.task
├─ .gitignore
├─ hand_joint_recognition.py
├─ README.md
├─ requirements.txt
└─ run.bat
```

## GitHub 업로드

이 폴더 내용 전체를 그대로 새 GitHub 저장소에 올리면 됩니다.  
배포용으로 만들어졌던 `.exe`, `_internal`, 백업 폴더들은 제외한 상태라서 소스 저장소용으로 바로 사용 가능합니다.
