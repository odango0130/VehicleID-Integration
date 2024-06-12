# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 06:18:51 2023

@author: yuuki
"""

from moviepy.editor import VideoFileClip
import numpy as np
from PIL import Image

def create_median_time_slice_image(video_path, output_image_path):
    # Load the video
    video = VideoFileClip(video_path)

    # Video properties
    width, height = video.size  # Get the dimensions of the video
    duration = video.duration  # Duration in seconds
    fps = video.fps  # Frames per second
    total_frames = int(duration * fps)

    # Final image array
    final_image_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Process each row (y) of the video
    for y in range(height):
        pixel_rows = []

        # Extract pixel row from each frame for this y
        for frame_number in range(total_frames):
            frame = video.get_frame(frame_number / fps)
            pixel_row = frame[y, :, :]  # Extract the pixel row at height y
            if pixel_row.shape == (width, 3):  # Ensure each pixel row is correct width x 3
                pixel_rows.append(pixel_row.reshape(1, width, 3))

        # Combine rows into a single image and calculate median
        time_slice_image = np.vstack(pixel_rows)
        median_row = np.median(time_slice_image, axis=0).astype(np.uint8)
        final_image_array[y, :, :] = median_row

    # Convert to PIL Image for saving
    final_image_pil = Image.fromarray(final_image_array)

    # Save the final image
    final_image_pil.save(output_image_path)

# Example usage
video_path = '1900_tracked_crop2-5_fixed.mp4'
output_image_path = 'final_median_time_slice_image_2-5_fixed.png'
create_median_time_slice_image(video_path, output_image_path)

