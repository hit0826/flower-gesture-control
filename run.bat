@echo off
setlocal
cd /d "%~dp0"
py hand_joint_recognition.py --camera 0 --flower-video "assets\flower_input.mp4" --model "models\hand_landmarker.task"
endlocal
