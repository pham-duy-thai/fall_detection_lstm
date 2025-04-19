import cv2
import os

# Folder containing input images
image_folder = 'photos'  

# Output video path
video_name = 'videos/input.mp4'

# Get sorted list of image files
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")])
if not images:
    print(" No image files found in the 'photos' folder.")
    exit()

# Read the first image to get dimensions
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = frame.shape

# Initialize video writer
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

# Write images to video
for image in images:
    img = cv2.imread(os.path.join(image_folder, image))
    out.write(img)

out.release()
print(f" Video successfully created and saved as '{video_name}'.")
