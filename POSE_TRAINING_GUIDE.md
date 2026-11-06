# YOLOv8-Pose: vùng giường và người nằm

## 1. Chuẩn bị dataset

Các file `.npy` trong `data/` hiện là chuỗi keypoint cho LSTM, không phải ảnh
và nhãn pose YOLO. Cần tự gán nhãn ảnh/video thành dataset pose riêng:

```text
pose_bed_dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Mỗi ảnh cần một file `.txt` cùng tên trong thư mục labels tương ứng. Dòng nhãn
pose có dạng `class cx cy width height x1 y1 v1 ... x17 y17 v17`; bbox và tọa
độ keypoint được chuẩn hóa theo kích thước ảnh. COCO 17 điểm phải giữ đúng thứ
tự YOLOv8 (mũi, mắt, tai, vai, khuỷu, cổ tay, hông, gối, cổ chân). Với YAML mẫu,
class `0` là `lying_person`; chỉ gán nhãn cho người nằm mục tiêu. Nên có ảnh
âm tính/đối tượng gây nhiễu (người đứng cạnh giường, gối, chăn, đồ vật) trong
ảnh nền, nhưng chia train/val/test theo video hoặc cảnh quay, không chia ngẫu
nhiên frame liền kề để tránh rò rỉ dữ liệu.

YAML đang trỏ tới `./pose_bed_dataset` tương đối với thư mục dự án
`fall_detection_lstm`. Chỉnh `path`, số keypoint hoặc thứ tự điểm nếu format
dataset của bạn khác COCO 17 điểm.

## 2. Train

Từ PowerShell:

```powershell
cd D:\detect\fall_detection_lstm
python train_pose_bed.py
```

Script fine-tune `yolov8n-pose.pt` và lưu checkpoint tốt nhất tại:

```text
runs/pose/lying_person_pose_v1/weights/best.pt
```

Tham số hiện dùng `degrees=60` (xoay ngẫu nhiên trong khoảng ±60°),
`fliplr=0.50`, `flipud=0.10`, `mosaic=0.50`, `mixup=0.05`, và tắt mosaic ở
10 epoch cuối. `flipud` chỉ nên giữ nếu hướng camera có thể đảo/ngược; nếu góc
camera cố định, giảm về `0.0`. `mixup` được để thấp để không làm méo hình người
và keypoint; augmentation phải được kiểm tra trên video thực tế.

## 3. Dùng ngưỡng confidence

`conf` là ngưỡng score detection khi predict, không phải tham số train và
không phải ngưỡng riêng tự động cho mỗi class. Backend lấy các ứng viên từ YOLO
với `conf=0.05`, sau đó chọn class `0` nếu confidence đạt
`PRIMARY_PERSON_CONFIDENCE` (mặc định `0.20`) và bbox nằm trong vùng trung tâm.
Đặt ngưỡng thấp hơn giúp giữ nhiều ứng viên nhưng tăng false positive; đặt cao
hơn giảm false positive nhưng có thể bỏ sót. Bắt đầu thử `0.20`, rồi so sánh
precision/recall trên video xác nhận và điều chỉnh khoảng `0.15–0.40`.

Ví dụ predict hậu kiểm thủ công:

```python
result = model(frame, conf=0.05, verbose=False)[0]
person = select_primary_person(
    result,
    frame.shape,
    confidence_threshold=0.30,  # class 0: lying_person
    person_class_id=0,
    center_region=(0.15, 0.10, 0.85, 0.92),
)
```

Không có ngưỡng `conf` riêng theo class trong `model.predict`; hãy lấy ứng viên
bằng một ngưỡng thấp rồi lọc `boxes.cls` và `boxes.conf` ở post-processing như
`select_primary_person()`.

## 4. Chọn checkpoint mới trong backend

Sau khi xem kết quả `best.pt` trên video validation, chạy backend với biến môi
trường tùy chọn:

```powershell
$env:FALL_POSE_MODEL="D:\detect\fall_detection_lstm\runs\pose\lying_person_pose_v1\weights\best.pt"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Nếu không đặt biến này, backend tiếp tục dùng checkpoint hiện tại.

Muốn dùng pipeline hai giai đoạn (model detect người trước, crop bbox được
chọn rồi mới gọi pose), đặt thêm đường dẫn checkpoint YOLO detect người:

```powershell
$env:FALL_PERSON_DETECTOR="yolov8n.pt"
```

Checkpoint detect cần có class người ở class id `0`. Ultralytics sẽ detect
người, chọn bbox lớn nhất trong ROI, crop bbox có thêm padding 12%, rồi gọi
model pose trên crop đó. Nếu không đặt `FALL_PERSON_DETECTOR`, backend dùng
YOLOv8-Pose hiện tại một lần và chỉ chuyển tiếp keypoint cùng index với bbox
được chọn; không cần tải thêm model.

## 5. Chọn người chính

`primary_person.py` chọn box có diện tích lớn nhất trong ROI trung tâm và bỏ
các detection ở rìa. Nếu vùng trung tâm không có ứng viên đủ confidence, frame
được xem là không có người đủ điều kiện.
