import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

# ---------------------------
# Helper: Load YOLO Model
# ---------------------------
@st.cache_resource
def load_yolo():
    # Load class labels
    labelsPath = os.path.join("yolo", "coco.names")
    with open(labelsPath, "r") as f:
        classes = f.read().strip().split("\n")
    
    # Load YOLO model configuration and weights
    net = cv2.dnn.readNet(os.path.join("yolo", "yolov4.weights"), os.path.join("yolo", "yolov4.cfg"))
    
    # Uncomment these lines if you have GPU support
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    
    return net, output_layers, classes

net, output_layers, classes = load_yolo()

# ---------------------------
# Helper: Process Video
# ---------------------------
def process_video(video_path, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    processed_frames = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        # Loop over detections
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Non-max suppression to remove duplicates
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        object_count = {}
        
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f'{label} {int(confidence * 100)}%', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                object_count[label] = object_count.get(label, 0) + 1
        
        # Display object counts on frame
        y_offset = 30
        for label, count in object_count.items():
            cv2.putText(frame, f'{label}: {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            y_offset += 30
        
        processed_frames.append(frame)
        frame_count += 1

    cap.release()
    return processed_frames

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Real-Time Object Detection and Counting")
st.write("Upload a video file to run object detection using YOLOv4.")

# Video file uploader
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    st.video(uploaded_file)  # Show original video
    
    if st.button("Run Object Detection"):
        st.write("Processing video...")
        processed_frames = process_video(tfile.name, max_frames=100)
        
        # Create a directory for output frames (optional)
        output_dir = Path("output_frames")
        output_dir.mkdir(exist_ok=True)
        frame_paths = []
        
        # Save processed frames as images
        for i, frame in enumerate(processed_frames):
            frame_path = output_dir / f"frame_{i:03d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
        
        st.success("Processing complete!")
        
        # Display processed frames as a gallery (or create a video if needed)
        st.write("Processed Frames:")
        for frame_path in frame_paths:
            st.image(frame_path, channels="BGR")
        
        # Optionally, you could create a video file from frames and offer a download link.
