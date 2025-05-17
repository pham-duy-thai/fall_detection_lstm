# Fall Detection using YOLOv8 and LSTM

## Project Overview  
This project aims to detect human falls from videos.  
It uses **YOLOv8n-Pose** to extract 17 keypoints of the human body and an **LSTM model** to analyze the sequence of movements and classify them as either **Fall** or **No Fall (Normal Activity)**.

---

## Technologies Used  

- Python 3.13  
- PyTorch  
- Ultralytics YOLOv8  
- OpenCV  
- NumPy  
- Scikit-learn  

---

## Project Workflow  

1. **(Optional) Create videos from image frames**  
   If you have frame-by-frame images, use `make_multiple_videos.py` to generate `.mp4` videos.

2. **Extract pose keypoints from videos**  
   Run `extract_keypoints_batch.py` to extract body keypoints from videos inside the `videos/` folder. Output files (`.npy`) are saved to the `data/` folder.

3. **Generate sequence data (features)**  
   Use `generate_sequences.py` to extract features (e.g., torso angle, head distance) and create `sequences.npy` and `labels.npy`.

4. **Train LSTM model**  
   Run `train_fall_lstm.py` to train the LSTM using the generated sequences.  
   The trained model will be saved as `lstm_fall_model.pth`.

5. **Run the model (live or on test video)**  
   Execute `run_fall_detection_lstm.py`.  
   - Set `use_webcam = True` for real-time detection  
   - Set `use_webcam = False` to test on an `.mp4` file

---

## Folder Structure and File Naming  

- Training and test videos must be placed inside the `videos/` folder.  
- Video names must start with either `fall_` or `nofall_` (e.g., `fall_1.mp4`, `nofall_3.mp4`)  
- The output `.npy` files will be automatically saved inside the `data/` folder (no need to create manually).  
- Trained model is saved as `lstm_fall_model.pth`

---

## Model Features  

- Automatically runs on **GPU** if available, otherwise falls back to **CPU**  
- Supports both real-time (live webcam) and offline (`.mp4`) detection  
- Input features include torso angle, torso length, and head-to-hip distance  

---

## Final Accuracy  

Using approximately **70 training videos**, the model achieved an accuracy of about **84%**.  
This accuracy can be improved by increasing the number and diversity of training videos  
(e.g., different camera angles, lighting conditions, or real-world falls).

---

## Resume Training (Optional)  

If `lstm_fall_model.pth` already exists, you can **resume training** by loading it into `train_fall_lstm.py`.  
This allows you to **continue training** the model instead of starting from scratch.

---

## Compatibility Notes  

- Python 3.13 was used for development and testing  
- Works on other Python versions (e.g., 3.10/3.11) with compatible libraries  
- Torch automatically detects CUDA and runs on GPU if available  
- The project avoids hard dependencies on system hardware and runs on CPU by default

---

## Additional Notes  

- Code includes readable emoji-style messages to visually indicate progress and errors (e.g., ✅, ❌)  
- The system is suitable for use in **elderly care**, **smart security**, or **healthcare applications**

---

## Final Words  

This fall detection system provides a solid foundation and performs reliably using only body keypoints and temporal features.  
With more video data and augmentation, the accuracy can be improved further.

---

## Project Info  

Project by: Mohammad Mahdi Salehi

Supervisor: Seyyed Mostafa Hashemi Motlagh

Date: May 2025


