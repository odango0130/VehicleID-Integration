# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:43:25 2023

@author: odayuki
"""
from moviepy.editor import VideoFileClip, ImageSequenceClip
import numpy as np
import time

def create_rolling_median_video(video_path, output_video_path, window_size=12, fps=24):
    # Load the video
    video = VideoFileClip(video_path)
    width, height = video.size
    duration = video.duration
    total_frames = int(duration * fps)

    # Initialize a list to hold all median images
    median_images = []

    # Generate median images with a rolling window
    for start_frame in range(0, total_frames - window_size + 1):
        pixel_rows = []

        for y in range(height):
            frame_pixel_rows = []

            for frame_number in range(start_frame, start_frame + window_size):
                frame = video.get_frame(frame_number / fps)
                pixel_row = frame[y, :, :]
                if pixel_row.shape == (width, 3):
                    frame_pixel_rows.append(pixel_row.reshape(1, width, 3))

            time_slice_image = np.vstack(frame_pixel_rows)
            median_row = np.median(time_slice_image, axis=0).astype(np.uint8)
            pixel_rows.append(median_row)

        final_image_array = np.vstack(pixel_rows)
        median_images.append(final_image_array)

    # Convert images to video clip
    final_video = ImageSequenceClip(median_images, fps=fps)

    # Write video file
    final_video.write_videofile(output_video_path, codec='libx264')

# Example usage
video_path = '1900_tracked_crop2-5_fixed.mp4'
output_video_path = 'rolling_median_video.mp4'
start_time = time.time()
create_rolling_median_video(video_path, output_video_path)
end_time = time.time()
print("実行時間:", end_time - start_time, "秒")
