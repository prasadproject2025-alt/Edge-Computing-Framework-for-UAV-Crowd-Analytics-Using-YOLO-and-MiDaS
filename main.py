import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm


# ---------------- Device setup (auto CPU/GPU, no hardcoding) ----------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device_index = 0 if torch.cuda.is_available() else "cpu"  # for YOLO's predict(device=...)
print(f"[INFO] Using device: {device}")

# ---------------- Load MiDaS model ----------------
# Use MiDaS_small for edge/UAV deployment (much faster than DPT_Hybrid/DPT_Large)
model_type = "MiDaS_small"
# model_type = "DPT_Hybrid"
# model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device)
midas.eval()

# Load MiDaS transforms (small model uses the "small_transform", not dpt_transform)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "MiDaS_small":
    transform = midas_transforms.small_transform
else:
    transform = midas_transforms.dpt_transform

# ---------------- Load YOLO model ONCE (was reloaded every frame — fixed) ----------------
model = YOLO('yolov8n.pt')  # nano detection model (lighter than seg model, no masks needed for counting)
PERSON_CLASS_ID = 0  # COCO class 0 = person

# ---------------- Video source ----------------
# For a live webcam feed:
video_path = 0
# For a drone RTSP stream, use something like:
# video_path = "rtsp://<drone-ip>:<port>/stream"
# For a local video file:
# video_path = "path/to/your_video.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {video_path}")

# Output video file
output_video_path = 'output_depth_headcount.mp4'

# Video settings
fps_in = cap.get(cv2.CAP_PROP_FPS)
fps_in = fps_in if fps_in and fps_in > 0 else 20  # webcams often report 0 fps
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps_in, (frame_width, frame_height))


def picture_in_picture(main_image, overlay_image, img_ratio=3, border_size=3, x_margin=30, y_offset_adjust=-100):
    """
    Overlay an image onto a main image with a white border.
    """
    if main_image is None or overlay_image is None:
        raise FileNotFoundError("One or both images not found.")

    new_height = main_image.shape[0] // img_ratio
    new_width = int(new_height * (overlay_image.shape[1] / overlay_image.shape[0]))
    overlay_resized = cv2.resize(overlay_image, (new_width, new_height))

    overlay_with_border = cv2.copyMakeBorder(
        overlay_resized,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    x_offset = main_image.shape[1] - overlay_with_border.shape[1] - x_margin
    y_offset = (main_image.shape[0] // 2) - overlay_with_border.shape[0] + y_offset_adjust

    main_image[y_offset:y_offset + overlay_with_border.shape[0],
               x_offset:x_offset + overlay_with_border.shape[1]] = overlay_with_border

    return main_image


# Variables for FPS calculation
frameId = 0
start_time = time.time()
fps_display = str()

# total_frames may be 0/unreliable for live streams — guard the progress bar
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
use_progress_bar = total_frames > 0
progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame") if use_progress_bar else None

while True:
    frameId += 1

    ret, frame = cap.read()
    if not ret:
        break
    img = frame.copy()
    image = frame.copy()

    # Transform the image for MiDaS
    input_batch = transform(img).to(device)

    interpolation_mode = 'bilinear'

    # Run MiDaS Depth
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode=interpolation_mode,
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map = 1.0 - depth_map_normalized

    # ---------------- Run YOLO Detector (person-only, model loaded once above) ----------------
    image = img.copy()
    results = model.predict(image, verbose=False, device=device_index, classes=[PERSON_CLASS_ID])

    head_count = 0

    for predictions in results:
        if predictions is None or predictions.boxes is None:
            continue

        head_count = len(predictions.boxes)  # number of people detected this frame

        for bbox in predictions.boxes:
            scores = bbox.conf[0]
            classes = bbox.cls[0]
            bbox_coords = bbox.xyxy[0]

            xmin = bbox_coords[0]
            ymin = bbox_coords[1]
            xmax = bbox_coords[2]
            ymax = bbox_coords[3]
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 225), 2)

            # Get the depth values within the bounding box
            depth_values_bbox = depth_map[int(ymin):int(ymax), int(xmin):int(xmax)]
            if depth_values_bbox.size == 0:
                continue
            depth_value = np.median(depth_values_bbox)

            # Relative distance estimate (NOT calibrated metric distance — MiDaS is relative depth)
            scale_factor = 15
            distance = depth_value * scale_factor

            overlay_frame = image.copy()
            font_scale = 0.4
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_color = (255, 255, 255)
            background_color = (30, 30, 30)
            line_spacing = int(15 * font_scale)
            text_x = int(xmin) + 5
            text_y = int(ymin) + 5

            text_lines = [
                str(predictions.names[int(classes)]),
                str(round(float(scores) * 100, 1)) + '%',
                f'Dist: {distance:.3f} (rel)'
            ]
            text_sizes = [cv2.getTextSize(line, font, font_scale, 1)[0] for line in text_lines]
            max_width = max(w for w, h in text_sizes) + 3
            total_height = sum(h for w, h in text_sizes) + (len(text_lines) - 1) * line_spacing + 3

            cv2.rectangle(image, (text_x - 5, text_y - text_sizes[0][1] - 5),
                          (text_x + max_width, text_y + total_height - 5),
                          background_color, cv2.FILLED)

            for i, line in enumerate(text_lines):
                line_y = text_y + i * (text_sizes[i][1] + line_spacing)
                cv2.putText(image, line, (text_x, line_y), font, font_scale, font_color, 1)

            image = cv2.addWeighted(overlay_frame, 0.5, image, 0.5, 0)

    # ---------------- FPS calculation ----------------
    if frameId % 10 == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps_current = 10 / elapsed_time if elapsed_time > 0 else 0
        fps_display = f'FPS: {fps_current:.2f}'
        start_time = time.time()

    # Depth map visualization (picture-in-picture)
    depth_map_colored = plt.cm.plasma(depth_map / depth_map.max())[:, :, :3]
    depth_map_colored = (depth_map_colored * 255).astype(np.uint8)
    image = picture_in_picture(image, depth_map_colored)

    # Overlay FPS and head count
    cv2.putText(image, fps_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f'Head Count: {head_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('UAV Crowd Analytics - YOLO + MiDaS', image)
    out.write(image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if use_progress_bar:
        progress_bar.update(1)

if use_progress_bar:
    progress_bar.close()

cap.release()
out.release()
cv2.destroyAllWindows()