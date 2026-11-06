# Dữ liệu nhận dạng cũ để nén

Thư mục `dataset_archive_old` là bản tập hợp độc lập của bộ dữ liệu nhận dạng cũ:

- `videos/`: video gốc dùng cho pipeline hiện tại
- `data/`: keypoint `.npy` đã trích xuất
- `datafall/` và `datanofall/`: các video nguồn được phân loại

Không đưa `reviewed/` vào bộ cũ; đây là dataset mới dùng để train.

## Train dataset mới (`reviewed`)

Dataset mới gồm `reviewed/fall` và `reviewed/nofall`. Không dùng thư mục `reviewed/uncertain`.

```powershell
$env:KEYPOINT_INPUT_DIR = "reviewed"
$env:KEYPOINT_OUTPUT_DIR = "data_reviewed"
python extract_keypoints_batch.py

$env:SEQUENCE_OUTPUT = "sequences_reviewed.npy"
$env:LABEL_OUTPUT = "labels_reviewed.npy"
python generate_sequences.py

$env:SEQUENCE_INPUT = "sequences_reviewed.npy"
$env:LABEL_INPUT = "labels_reviewed.npy"
$env:MODEL_OUTPUT = "lstm_fall_model_reviewed.pth"
$env:RESUME_TRAINING = "0"
python train_fall_lstm.py
```

Các biến môi trường giúp tách hoàn toàn output mới khỏi `data/`, `sequences.npy`,
`labels.npy` và model cũ.
