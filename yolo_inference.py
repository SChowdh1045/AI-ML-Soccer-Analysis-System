from ultralytics import YOLO
import torch

# Load model
model = YOLO('models\\best.pt') # using yolov5l model

# Perform inference
results = model('input_videos\\08fd33_4.mp4', save=True)
# "results[0]" consists of detected boxes, labels, and scores

# Iterate over detected boxes
# "results[0].boxes" is a list of BoundingBox objects
for box in results[0].boxes:
    print(box)
