import argparse
import json
import torch
import av
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from config import Config

# CANDIDATE:
# Level 1: Add code to analyze frame_numpy by the model, handle timestamps and preductions in the way it will allow to return JSON with scene segments (see the doc)
# Level 2a: Implement error handling for cases when the model is not available or the input frame is not recognized.
# Level 2b: Implement a stride in the frame analysis to improve performance. Use two frames per second.
# Level 3: Implement a mechanism to detect scene transitions between segments. Use the OpenCV library for this.

class Analyzer:
    """
    1. Loads the model and processes video frames through it.
    2. Handles missing or corrupted models gracefully.
    3. Implements frame skipping (stride of 2 frames per second) for better performance.
    4. Detects scene transitions using OpenCV.
    Returns a structured JSON output with scene segments and predictions.
    """
    def __init__(self, model_path: str):
        self.timestamps = []
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def load_model(self, model_path):
        """Load the model and handle errors if the model is missing or corrupted."""
        try:
            model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            model = model.to(Config.DEVICE)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None  # Handle cases where the model cannot be loaded

    def analyze_frame(self, frame_numpy):
        """Process a single frame through the model and return predictions."""
        if self.model is None:
            return None  # If model failed to load, return None

        input_tensor = self.transform(frame_numpy).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        return predicted_class, probabilities.numpy().tolist()

    def detect_scene_change(self, prev_frame, current_frame):
        """Detect scene transitions using OpenCV (SSIM or frame difference)."""
        threshold = Config.SSIM_THRESHOLD
        if prev_frame is None:
            return False  # No previous frame to compare

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Compute absolute difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        mean_diff = np.mean(diff)
        
        return mean_diff > threshold  # Return True if significant scene change

    def __call__(self, request):
        url = request.get("url", None)
        if url is None:
            print("Missing URL in request!")
            return None

        headers = request.get("headers", "")
        headers_av = headers
        options = {}
        if headers != "":
            if headers.startswith("'") and headers.endswith("'"):
                headers_av = headers[1:-1]
            options = {"headers": headers_av}

        try:
            container = av.open(url, options=options)
        except Exception as e:
            print(f"Error opening video: {e}")
            return None

        generator = container.decode(video=0)
        frame_rate = container.streams.video[0].average_rate  # FPS
        stride = int(frame_rate // 2)  # Analyze two frames per second

        segments = []
        prev_frame = None
        current_segment = {"start_time": 0, "predictions": []}

        for frame_nb, frame in enumerate(generator):
            if frame_nb % stride != 0:
                continue  # Skip frames to process only 2 per second
            
            frame_numpy = frame.to_ndarray(format="bgr24")
            timestamp = frame.time
            prediction, probabilities = self.analyze_frame(frame_numpy)

            if prediction is not None:
                if prev_frame is not None and self.detect_scene_change(prev_frame, frame_numpy):
                    # Store completed segment
                    if current_segment["predictions"]:
                        current_segment["end_time"] = timestamp
                        segments.append(current_segment)
                    
                    # Start a new segment
                    current_segment = {"start_time": timestamp, "predictions": []}

                current_segment["predictions"].append({
                    "time": timestamp,
                    "class_id": prediction,
                    "probabilities": probabilities
                })
            
            prev_frame = frame_numpy  # Store frame for next comparison

        # Append last segment
        if current_segment["predictions"]:
            current_segment["end_time"] = container.duration / 1e6  # Convert microseconds to seconds
            segments.append(current_segment)

        return json.dumps({"segments": segments}, indent=4)

def parse_args():
    model_type = Config.MODEL.lower()
    parser = argparse.ArgumentParser(description="Video shot-type analyzer")
    parser.add_argument("--url", help="Path to the input video file", type=str, required=True)
    parser.add_argument("--headers", help="Headers", type=str, default="")
    parser.add_argument("--model_path", help="Path to the model", type=str, default="./saved_models/best_model_"+model_type+".pth")
    return parser.parse_args()

if __name__ == "__main__":
    #CANDIDATE: check tests/test_analyzer.py
    # ALL Levels: Implement at least 3 unit tests for this module. Note that you may choose the unit test framework you're most familiar with. Install it in docker
    args = parse_args()
    analyzer = Analyzer(args.model_path)
    request_data = {"url": args.url, "headers": args.headers}
    result = analyzer(request_data)
    if result:
        print(result)
