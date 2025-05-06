import cv2
import os

def make_video_from_frames(input_folder, output_video_path, fps=25):
    images = sorted([img for img in os.listdir(input_folder) if img.endswith(".jpg") or img.endswith(".png")])
    if not images:
        print(f"❌ No images found in {input_folder}")
        return False

    first_frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, _ = first_frame.shape
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for img_name in images:
        frame = cv2.imread(os.path.join(input_folder, img_name))
        out.write(frame)

    out.release()
    print(f"✅ Video saved to {output_video_path}")
    return True

# ساخت پوشه videos اگر وجود نداشته باشه
if not os.path.exists("videos"):
    os.makedirs("videos")

# ساخت ویدیو برای پوشه‌های fall
for i in range(1, 31):  # فرض بر اینکه 30 سقوط داری
    folder = f"photos_fall_{i}"
    if os.path.exists(folder):
        output_path = f"videos/fall_{i}.mp4"
        make_video_from_frames(folder, output_path)

# ساخت ویدیو برای پوشه‌های nofall
for i in range(1, 41):  # فرض بر اینکه 40 رفتار عادی داری
    folder = f"photos_nofall_{i}"
    if os.path.exists(folder):
        output_path = f"videos/nofall_{i}.mp4"
        make_video_from_frames(folder, output_path)
