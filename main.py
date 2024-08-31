import cv2
import torch
import warnings
import numpy as np
from sort.sort import Sort

warnings.filterwarnings("ignore", category=FutureWarning)

# Function to get the center of bounding boxes
def get_centers(x1, y1, x2, y2):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return int(cx), int(cy)

# Load YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

up = []
down = []
yl1 = 380

# Initialize the SORT tracker
trackers = Sort(max_age=4, iou_threshold=0.5)

# Set model to evaluation mode and move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.eval().to(device)

# Open video file
cap = cv2.VideoCapture("media/original.mp4")

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize the VideoWriter object
out = cv2.VideoWriter('media/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (1080,720))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (1080, 720))

    # Crop the frame to focus on the area of interest
    wframe = frame[250:, :]

    # Perform detection
    with torch.no_grad():
        results = model(wframe)

    cv2.line(frame, (0, yl1), (980, yl1), (255, 255, 255), 3)
    # Process detections
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]
    
    # Filter detections for cars and trucks (COCO class indices 2 and 7)
    car_truck_detections = detections[(detections[:, -1] == 2) | (detections[:, -1] == 7)]

    # Update tracker with the detected cars and trucks only if there are any
    if len(car_truck_detections) > 0:
        tracked_objects = trackers.update(car_truck_detections)
    else:
        tracked_objects = []  # No update if no relevant detections

    # Plot bounding boxes and track objects
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj
        y1 += 250  # Adjust y-coordinates based on the crop
        y2 += 250

        # Draw bounding box and ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cx, cy = get_centers(x1, y1, x2, y2)
        if (yl1 - 10) <= cy <= (yl1 + 10):

            if 570 <= cx < 1000:
                up.append(track_id)
                              
            elif cx <= 480:
                down.append(track_id)

        text1 = f"Up: {len(set(up))}"
        text2 = f"Down: {len(set(down))}"

        text_size, _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_width, text_height = text_size

        cv2.rectangle(frame, (455, 382), (455 + text_width, 382 + (text_height * 2) + 5), (250, 250, 250), -1)
        cv2.putText(frame, text1, (455, 382 + 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)
        cv2.putText(frame, text2, (455, 382 + 20 + text_height), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

    # Display the result
    cv2.imshow('Tracking', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  # Don't forget to release the VideoWriter
cv2.destroyAllWindows()
