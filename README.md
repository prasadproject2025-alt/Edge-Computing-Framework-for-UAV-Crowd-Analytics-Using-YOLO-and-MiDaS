# Edge-Based UAV Crowd Analytics

This project demonstrates a lightweight crowd-monitoring pipeline for UAV footage that combines person detection with depth-based spatial analysis. It is designed for edge hardware such as a Raspberry Pi, but it also includes a built-in demo mode so it can run on a regular desktop machine without downloading heavyweight models.

## Features
- Person detection with YOLO if a YOLO weights file is available
- Fallback person detection with OpenCV HOG when YOLO is unavailable
- Depth estimation using a MiDaS TFLite model if the model file exists
- Fallback synthetic depth estimation for offline/demo use
- Spatial density analysis that highlights crowded regions and raises an alert
- Demo mode for immediate testing without a camera

## Project structure
- main.py: complete runnable pipeline
- requirements.txt: Python dependencies
- tests/test_density.py: small regression tests for the crowd-density logic

## Installation
On Windows or Linux, install the dependencies with:

```bash
pip install -r requirements.txt
```

If you want to use the YOLO and MiDaS paths exactly as described in the paper, install:

```bash
pip install ultralytics tflite-runtime
```

## Run
### Demo mode (works immediately)
```bash
python main.py --source demo --max-frames 20 --no-display
```

### Webcam mode
```bash
python main.py --source 0
```

### Video file mode
```bash
python main.py --source path/to/video.mp4
```

### Save output to a file
```bash
python main.py --source demo --output outputs/videos/crowd_demo.mp4 --max-frames 50 --no-display
```

## Notes
- The project uses a simple spatial-density heuristic based on person bounding boxes and depth values.
- If YOLO or MiDaS weights are absent, the app automatically switches to fallback methods so that the application remains runnable.
- To use the full paper-style pipeline, place your YOLO weights at yolo11n.pt and your MiDaS model as midas_v2_1_small.tflite in the project folder.
 - Recommended output folder: use `outputs/videos/` to keep recordings separate from source.
	 Create it with `mkdir -p outputs/videos` (or `mkdir outputs\videos` on Windows) before running, or pass `--output` and the script will create parent folders automatically.

## Testing
```bash
pytest -q
```

