# Fall Detection using YOLOv8 and LSTM

This project is focused on detecting human falls in videos or real-time streams using a combination of **pose estimation** (YOLOv8n-pose) and **temporal modeling** (LSTM).

---

## 📌 Overview

- Extracts **17 human keypoints** using Ultralytics YOLOv8n-pose.
- Converts keypoints into meaningful time-based features.
- Uses an LSTM neural network to classify behavior as either `FALL` or `NO FALL`.
- Supports both **live webcam feed** and **MP4 video files**.
- Lightweight and works on **CPU** and **GPU (CUDA)** environments.

---

## 📁 Folder Structure

fall_detection_lstm/
│
├── data/ # Extracted keypoints (.npy)
│ ├── keypoints_fall_.npy
│ └── keypoints_nofall_.npy
├── videos/ # Input videos (fall_1.mp4, nofall_1.mp4, ...)
├── lstm_model.py # LSTM network definition
├── make_multiple_videos.py # Optional - Converts frame sequences to videos
├── extract_keypoints_batch.py # Batch keypoint extraction using YOLOv8
├── generate_sequences.py # Feature engineering and labeling
├── train_fall_lstm.py # Model training and saving
├── run_fall_detection_lstm.py # Real-time or MP4-based inference
├── yolov8n-pose.pt # YOLOv8 pose weights (must download)
├── requirements.txt # Required Python libraries
└── README.md # Project documentation




## ⚙️ Installation

Install Python packages using pip:

```bash
pip install -r requirements.txt



Make sure to manually download YOLOv8n-pose weights:

Download yolov8n-pose.pt

Or use this command to auto-download:

from ultralytics import YOLO
YOLO("yolov8n-pose.pt")


🚀 How to Run the Project

Step 1 – [Optional] Convert image frames to videos

python make_multiple_videos.py

Step 2 – Extract pose keypoints from videos

python extract_keypoints_batch.py

Step 3 – Generate features and labels

python generate_sequences.py

Step 4 – Train the LSTM model

python train_fall_lstm.py

Step 5 – Run fall detection

use_webcam = True  # or False to use mp4

video_path = "videos/fall_1.mp4"


Then run:

python run_fall_detection_lstm.py



🎯 Accuracy and Performance
Current accuracy: ~84%

Accuracy improves significantly by:

Increasing number of training videos

Using more diverse camera angles and subjects

Increasing video resolution and lighting quality




💡 Notes
The model automatically detects CUDA (GPU) if available, otherwise runs on CPU.

Messages and logs are enhanced with emojis to improve visual clarity (✅, ❌, 🔄).

The LSTM model is saved as lstm_fall_model.pth, so retraining is optional unless updating the dataset.


Project Title: Fall Detection using YOLOv8 and LSTM
Author: Mohammad Mahdi Salehi
Date: May 2025


