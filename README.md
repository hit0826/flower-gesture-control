
This is a Python project that uses a webcam to detect how open your fingers are and control the bloom state of a flower video in real time.  
It uses `MediaPipe` for hand landmark tracking and `OpenCV` to composite the camera feed with the flower video.
손 관절마디를 보이게 하고 꽃을 움직여 봅시다
### Included Files

- `hand_joint_recognition.py`: Main application script
- `assets/flower_input.mp4`: Default flower video
- `models/hand_landmarker.task`: Hand landmark model
- `run.bat`: Batch file for quick execution on Windows

### Requirements

- Windows PC
- Webcam
- Python 3.x

If `mediapipe` installation fails, Python 3.11 is usually the most reliable option.

### Installation

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run

The simplest way:

```powershell
.\run.bat
```

Run directly:

```powershell
py hand_joint_recognition.py
```

Example with options:

```powershell
py hand_joint_recognition.py --camera 0 --flower-video "assets\flower_input.mp4" --model "models\hand_landmarker.task" --width 1280 --height 720
```

Press `q` or `Esc` to quit.

### Folder Structure

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
