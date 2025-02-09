import pytest
import torch
import numpy as np
import cv2
from src.inference import Analyzer

# Test 1: Model Loading
def test_model_loading():
    """Test if the model loads correctly and handles invalid paths."""
    analyzer = Analyzer("non_existent_model.pth")
    assert analyzer.model is None  # Model should be None if it fails to load

    # Test with a dummy model
    dummy_model_path = "tests/dummy_model.pth"
    dummy_model = torch.nn.Linear(10, 2)
    torch.save(dummy_model, dummy_model_path)

    analyzer = Analyzer(dummy_model_path)
    assert analyzer.model is not None  # Model should load successfully

# Test 2: Frame Analysis
def test_frame_analysis():
    """Test frame analysis with a dummy image."""
    analyzer = Analyzer("tests/dummy_model.pth")

    frame = np.zeros((224, 224, 3), dtype=np.uint8)  # Black image
    prediction = analyzer.analyze_frame(frame)

    assert prediction is None or isinstance(prediction, tuple)  # Model should return a tuple (class_id, probabilities) or None

# Test 3: Scene Change Detection
def test_scene_change_detection():
    """Test scene transition detection using OpenCV."""
    analyzer = Analyzer("./dummy_model.pth")

    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)  # Black frame
    frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White frame

    assert analyzer.detect_scene_change(frame1, frame2) is True  # Scene should change
    assert analyzer.detect_scene_change(frame1, frame1) is False  # No change
