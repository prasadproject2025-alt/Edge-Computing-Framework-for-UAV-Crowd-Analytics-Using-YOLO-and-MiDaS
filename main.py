import argparse
import math
import os
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None

try:
    import tflite_runtime.interpreter as tflite
except ImportError:  # pragma: no cover - optional dependency
    tflite = None


MIDAS_MODEL_PATH = "midas_v2_1_small.tflite"
YOLO_MODEL_PATH = "yolo11n.pt"

# Tuning dials for UAV altitude calibration
CROWD_THRESHOLD = 4
SPATIAL_RADIUS = 150
DEPTH_WEIGHT = 2.0


class PersonDetector:
    """Wraps YOLO if available, otherwise falls back to OpenCV HOG."""

    def __init__(self, model_path: str):
        self.model = None
        self.hog = None
        self.model_path = model_path

        if YOLO is not None and os.path.exists(model_path):
            self.model = YOLO(model_path)
        elif YOLO is not None:
            self.model = YOLO("yolo11n.pt") if os.path.exists("yolo11n.pt") else None

        if self.model is None:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.model is not None:
            try:
                results = self.model(frame, classes=[0], conf=0.35, imgsz=640, stream=False, verbose=False)
                boxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, max(1, x2 - x1), max(1, y2 - y1)))
                return boxes
            except Exception:
                pass

        if self.hog is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes, _ = self.hog.detectMultiScale(
                gray,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )
            return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in boxes]

        return []


class DepthEstimatorThread(threading.Thread):
    def __init__(self, model_path: str):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.current_frame: Optional[np.ndarray] = None
        self.latest_depth_map: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.use_simulation = True

        if tflite is not None and os.path.exists(model_path):
            try:
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.use_simulation = False
            except Exception:
                self.interpreter = None

    def _simulate_depth(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        center_x, center_y = w / 2.0, h / 2.0
        distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        depth = (255 - np.clip(distance / 4.0, 0, 255)).astype(np.uint8)
        return depth

    def run(self) -> None:
        while True:
            with self.lock:
                frame_to_process = self.current_frame.copy() if self.current_frame is not None else None

            if frame_to_process is not None:
                if self.use_simulation:
                    depth_map = self._simulate_depth(frame_to_process)
                else:
                    input_shape = self.input_details[0]["shape"]
                    input_size = (input_shape[1], input_shape[2])
                    img = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, input_size)
                    img = img.astype(np.float32) / 255.0
                    img = np.expand_dims(img, axis=0)

                    self.interpreter.set_tensor(self.input_details[0]["index"], img)
                    self.interpreter.invoke()
                    depth_output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
                    depth_min = depth_output.min()
                    depth_max = depth_output.max()
                    if depth_max == depth_min:
                        depth_map = np.zeros(frame_to_process.shape[:2], dtype=np.uint8)
                    else:
                        depth_map = ((255 * (depth_output - depth_min) / (depth_max - depth_min)).astype("uint8"))
                        h, w = frame_to_process.shape[:2]
                        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_CUBIC)

                with self.lock:
                    self.latest_depth_map = depth_map

            time.sleep(0.05)


def calculate_spatial_density(
    pedestrians: List[Tuple[int, int, int, int]],
    depth_map: Optional[np.ndarray],
    crowd_threshold: int = CROWD_THRESHOLD,
    spatial_radius: int = SPATIAL_RADIUS,
    depth_weight: float = DEPTH_WEIGHT,
) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
    """Return whether the scene is crowded and which boxes are implicated."""
    if not pedestrians or depth_map is None:
        return False, []

    points_3d = []
    for x, y, w, h in pedestrians:
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        cx = max(0, min(cx, depth_map.shape[1] - 1))
        cy = max(0, min(cy, depth_map.shape[0] - 1))
        z = float(depth_map[cy, cx])
        points_3d.append((cx, cy, z, (x, y, w, h)))

    crowded_bboxes = []
    for i, (x1, y1, z1, bbox1) in enumerate(points_3d):
        neighbors = 0
        for j, (x2, y2, z2, _) in enumerate(points_3d):
            if i == j:
                continue
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (depth_weight * (z2 - z1)) ** 2)
            if dist < spatial_radius:
                neighbors += 1
        if neighbors >= crowd_threshold:
            crowded_bboxes.append(bbox1)

    return bool(crowded_bboxes), crowded_bboxes


def build_demo_frame(frame_idx: int) -> np.ndarray:
    frame = np.full((480, 640, 3), (40, 55, 80), dtype=np.uint8)
    for idx in range(8):
        x = 70 + (idx * 65 + frame_idx * 4) % 430
        y = 120 + (idx % 3) * 85
        cv2.rectangle(frame, (x, y), (x + 35, y + 70), (40, 180, 255), -1)
        cv2.circle(frame, (x + 18, y + 24), 12, (255, 255, 255), -1)
    return frame


def annotate_frame(
    frame: np.ndarray,
    pedestrians: List[Tuple[int, int, int, int]],
    crowded_boxes: List[Tuple[int, int, int, int]],
    is_crowded: bool,
) -> np.ndarray:
    for x, y, w, h in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for x, y, w, h in crowded_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.putText(frame, f"Headcount: {len(pedestrians)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if is_crowded:
        cv2.putText(frame, "ALERT: CROWDED (HIGH SPATIAL DENSITY)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    return frame


def run_pipeline(args: argparse.Namespace) -> None:
    detector = PersonDetector(args.yolo_model)
    depth_thread = DepthEstimatorThread(args.midas_model)
    depth_thread.start()

    if args.source.lower() == "demo":
        cap = None
        frame_idx = 0
    else:
        if args.source.isdigit():
            source = int(args.source)
        else:
            source = args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {args.source}")

    writer = None
    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (640, 480))

    frame_count = 0
    while True:
        if cap is None:
            frame = build_demo_frame(frame_idx)
            frame_idx += 1
        else:
            ret, frame = cap.read()
            if not ret:
                break

        with depth_thread.lock:
            depth_thread.current_frame = frame
            current_depth_map = depth_thread.latest_depth_map

        pedestrians = detector.detect(frame)
        is_crowded, crowded_boxes = calculate_spatial_density(pedestrians, current_depth_map)
        frame = annotate_frame(frame, pedestrians, crowded_boxes, is_crowded)

        if current_depth_map is not None:
            depth_colormap = cv2.applyColorMap(current_depth_map, cv2.COLORMAP_JET)
            pip_w, pip_h = int(frame.shape[1] / 3), int(frame.shape[0] / 3)
            depth_pip = cv2.resize(depth_colormap, (pip_w, pip_h))
            frame[0:pip_h, frame.shape[1] - pip_w:frame.shape[1]] = depth_pip

        if writer is not None:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("Edge UAV Crowd Analytics", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1
        if args.max_frames and frame_count >= args.max_frames:
            break

    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge-based UAV crowd analytics with YOLO and MiDaS")
    parser.add_argument("--source", default="0", help="Camera index, video path, or 'demo'")
    parser.add_argument("--yolo-model", default=YOLO_MODEL_PATH, help="Path to YOLO weights (.pt)")
    parser.add_argument("--midas-model", default=MIDAS_MODEL_PATH, help="Path to MiDaS TFLite model")
    parser.add_argument("--output", default="", help="Optional output video file")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames")
    parser.add_argument("--no-display", action="store_true", help="Disable the video display window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[INFO] Starting Edge UAV Crowd Analytics")
    run_pipeline(args)


if __name__ == "__main__":
    main()