# make a video out of the figures, mp4 format with h.264 codec
import cv2
import os

# here the folder where you have saved the figures from the plotting:
image_folder = 'figures/susten_icethickness'
video_name = 'icethickness_evolution.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
# sort images by filename
images.sort()
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
cv2.destroyAllWindows()
video.release()