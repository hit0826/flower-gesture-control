import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


WINDOW_NAME = "Hand Joint Recognition"
LANDMARK_COLOR = (0, 255, 255)
LABEL_TEXT_COLOR = (0, 255, 255)
OPEN_START = 0.02
OPEN_FULL = 0.98
FLOWER_MAX_RATIO = 0.58
FLOWER_RESPONSE_GAMMA = 1.0
SMOOTHING_KEEP = 0.35
SMOOTHING_APPLY = 0.65
PLAYHEAD_GAIN = 0.85
PLAYHEAD_STEP_LIMIT = 24.0


BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def resource_path(*relative_parts: str) -> Path:
    return BASE_DIR.joinpath(*relative_parts)


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


MODEL_PATH = resolve_existing_path(
    resource_path("models", "hand_landmarker.task"),
    resource_path("hand_landmarker.task"),
)
DEFAULT_FLOWER_VIDEO = str(
    resolve_existing_path(
        resource_path("assets", "flower_input.mp4"),
        resource_path("flower_input.mp4"),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control the bloom state of a flower video with hand gestures."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--flower-video", default=DEFAULT_FLOWER_VIDEO)
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def load_video_frames(video_path: str) -> list[np.ndarray]:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Flower video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No decodable frames in video: {path}")
    return frames


def resize_to_cover(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    scale = max(width / src_w, height / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    x0 = max(0, (new_w - width) // 2)
    y0 = max(0, (new_h - height) // 2)
    return resized[y0 : y0 + height, x0 : x0 + width].copy()


def round_rect_mask(height: int, width: int, radius: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1)
    cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, height - radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, height - radius), radius, 255, -1)
    return mask


def composite_pip(canvas: np.ndarray, camera_frame: np.ndarray) -> None:
    pip_w = max(220, canvas.shape[1] // 4)
    pip_h = int(pip_w * 9 / 16)
    pip = resize_to_cover(camera_frame, pip_w, pip_h)
    radius = max(12, pip_w // 16)
    mask = round_rect_mask(pip_h, pip_w, radius)

    x0 = canvas.shape[1] - pip_w
    y0 = canvas.shape[0] - pip_h

    roi = canvas[y0 : y0 + pip_h, x0 : x0 + pip_w]
    inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(roi, roi, mask=inv)
    fg = cv2.bitwise_and(pip, pip, mask=mask)
    canvas[y0 : y0 + pip_h, x0 : x0 + pip_w] = cv2.add(bg, fg)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_landmarks(frame: np.ndarray, hand_landmarks, handedness_label: str) -> None:
    h, w = frame.shape[:2]
    points: list[tuple[int, int]] = []
    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
        cv2.circle(frame, (x, y), 4, LANDMARK_COLOR, -1)

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], LANDMARK_COLOR, 2)

    label = "LEFT" if handedness_label.lower().startswith("right") else "RIGHT"
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    x0 = max(8, points[0][0] - 6)
    y0 = max(8 + text_size[1], points[0][1] - 16)
    cv2.putText(frame, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, LABEL_TEXT_COLOR, 2, cv2.LINE_AA)


def finger_open_score(hand_landmarks, handedness_label: str) -> float:
    pts = hand_landmarks

    def up(tip: int, pip: int) -> float:
        return 1.0 if pts[tip].y < pts[pip].y else 0.0

    score = 0.0
    score += up(8, 6)
    score += up(12, 10)
    score += up(16, 14)
    score += up(20, 18)

    if handedness_label.lower().startswith("right"):
        thumb_open = pts[4].x > pts[3].x
    else:
        thumb_open = pts[4].x < pts[3].x
    score += 1.0 if thumb_open else 0.0

    return score / 5.0


def bloom_progress(raw_score: float) -> float:
    normalized = np.clip((raw_score - OPEN_START) / max(OPEN_FULL - OPEN_START, 1e-6), 0.0, 1.0)
    return float(normalized ** FLOWER_RESPONSE_GAMMA)


def main() -> None:
    args = parse_args()
    flower_frames = load_video_frames(args.flower_video)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Hand landmarker model not found: {model_path}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)

    playhead = 0.0
    smoothed = 0.0
    frame_max = max(0, len(flower_frames) - 1)
    bloom_frame_max = max(1, int(frame_max * FLOWER_MAX_RATIO))

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with vision.HandLandmarker.create_from_options(options) as hands:
        while True:
            ok, camera_frame = cap.read()
            if not ok:
                break

            camera_frame = cv2.flip(camera_frame, 1)
            rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            result = hands.detect_for_video(mp_image, timestamp_ms)

            target = smoothed
            if result.hand_landmarks and result.handedness:
                for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    label = handedness[0].category_name
                    draw_landmarks(camera_frame, hand_landmarks, label)
                    target = bloom_progress(finger_open_score(hand_landmarks, label))
                    break

            smoothed = smoothed * SMOOTHING_KEEP + target * SMOOTHING_APPLY
            desired = smoothed * bloom_frame_max
            delta = desired - playhead
            playhead += float(np.clip(delta * PLAYHEAD_GAIN, -PLAYHEAD_STEP_LIMIT, PLAYHEAD_STEP_LIMIT))
            playhead = float(np.clip(playhead, 0.0, bloom_frame_max))

            flower = flower_frames[int(playhead)]
            canvas = resize_to_cover(flower, args.width, args.height)
            composite_pip(canvas, camera_frame)

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
