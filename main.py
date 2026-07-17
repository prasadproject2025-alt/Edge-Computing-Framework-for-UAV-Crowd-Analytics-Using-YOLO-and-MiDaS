import cv2
import numpy as np
import threading
import time
import math
from ultralytics import YOLO
import tflite_runtime.interpreter as tflite

# --- Configuration & Thresholds ---
MIDAS_MODEL_PATH = "midas_v2_1_small.tflite"
YOLO_MODEL_PATH = "yolo11n.pt"
CROWD_THRESHOLD = 4        # Number of people clustered together to trigger an alert
SPATIAL_RADIUS = 150       # 3D distance threshold (pixels + depth units)
DEPTH_WEIGHT = 2.0         # Weight multiplier for depth in 3D distance calculation

class DepthEstimatorThread(threading.Thread):
    def __init__(self, model_path):
        super().__init__()
        self.daemon = True
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Shared variables with the main thread
        self.current_frame = None
        self.latest_depth_map = None
        self.lock = threading.Lock()
        
    def run(self):
        # Asynchronous background loop for depth estimation
        input_shape = self.input_details[0]['shape']
        input_size = (input_shape[1], input_shape[2]) # Usually (256, 256)
        
        while True:
            with self.lock:
                frame_to_process = self.current_frame.copy() if self.current_frame is not None else None
            
            if frame_to_process is not None:
                # Preprocess for MiDaS
                img = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, input_size)
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=0)

                # Run Inference
                self.interpreter.set_tensor(self.input_details[0]['index'], img)
                self.interpreter.invoke()
                depth_output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                # Normalize depth map to 0-255
                depth_min = depth_output.min()
                depth_max = depth_output.max()
                depth_normalized = (255 * (depth_output - depth_min) / (depth_max - depth_min)).astype("uint8")
                
                # Resize back to original frame size
                h, w = frame_to_process.shape[:2]
                depth_resized = cv2.resize(depth_normalized, (w, h), interpolation=cv2.INTER_CUBIC)

                with self.lock:
                    self.latest_depth_map = depth_resized
            
            time.sleep(0.05) # Prevent CPU hogging

def calculate_spatial_density(pedestrians, depth_map):
    """
    Pedestrians: List of tuples (x, y, w, h)
    Returns: Boolean (Is Crowded?), List of crowded bounding boxes
    """
    if not pedestrians or depth_map is None:
        return False, []

    points_3d = []
    for (x, y, w, h) in pedestrians:
        cx, cy = int(x + w/2), int(y + h/2)
        # Ensure coordinates are within frame bounds
        cx = max(0, min(cx, depth_map.shape[1] - 1))
        cy = max(0, min(cy, depth_map.shape[0] - 1))
        
        # MiDaS provides relative inverse depth (higher = closer)
        z = depth_map[cy, cx] 
        points_3d.append((cx, cy, z, (x, y, w, h)))

    crowded_bboxes = []
    is_crowded = False

    # O(N^2) pairwise distance calculation
    for i in range(len(points_3d)):
        neighbors = 0
        x1, y1, z1, bbox1 = points_3d[i]
        
        for j in range(len(points_3d)):
            if i == j:
                continue
            x2, y2, z2, _ = points_3d[j]
            
            # Calculate pseudo-3D distance
            # Z is weighted to account for perspective scaling
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (DEPTH_WEIGHT * (z2 - z1))**2)
            
            if dist < SPATIAL_RADIUS:
                neighbors += 1
                
        if neighbors >= CROWD_THRESHOLD:
            crowded_bboxes.append(bbox1)
            is_crowded = True

    return is_crowded, crowded_bboxes

def main():
    print("[INFO] Loading YOLO11 Nano...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    print("[INFO] Starting MiDaS Depth Estimation Thread...")
    depth_thread = DepthEstimatorThread(MIDAS_MODEL_PATH)
    depth_thread.start()
    
    # Initialize Video Stream (0 for default camera, or path to UAV video file)
    cap = cv2.VideoCapture(0)
    
    print("[INFO] Starting spatial analytics loop...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Update frame for the background depth thread
        with depth_thread.lock:
            depth_thread.current_frame = frame
            current_depth_map = depth_thread.latest_depth_map
            
        # 1. Run YOLO Object Detection (Main Thread)
        results = yolo_model(frame, classes=[0], verbose=False) # class 0 is 'person'
        pedestrians = []
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            pedestrians.append((x1, y1, w, h))
            
        # 2. Correlate 2D boxes with localized Depth Data
        is_crowded, crowded_boxes = calculate_spatial_density(pedestrians, current_depth_map)
        
        # 3. Visualization & UI
        # Draw standard detections
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        # Highlight crowded individuals
        for (x, y, w, h) in crowded_boxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            
        # UI Overlays
        cv2.putText(frame, f"Pedestrians: {len(pedestrians)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
        if is_crowded:
            cv2.putText(frame, "ALERT: CROWDED (HIGH SPATIAL DENSITY)", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            # You could trigger an MQTT or GPIO hardware alert here.

        # (Optional) Display Depth map for debugging
        if current_depth_map is not None:
            depth_colormap = cv2.applyColorMap(current_depth_map, cv2.COLORMAP_JET)
            # Resize depth map for Pi display picture-in-picture
            pip_w, pip_h = int(frame.shape[1]/3), int(frame.shape[0]/3)
            depth_pip = cv2.resize(depth_colormap, (pip_w, pip_h))
            frame[0:pip_h, frame.shape[1]-pip_w:frame.shape[1]] = depth_pip

        cv2.imshow("Edge UAV Crowd Analytics", frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()