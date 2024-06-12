# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 05:37:24 2023

@author: yuuki
"""

from moviepy.editor import VideoFileClip
import numpy as np
from PIL import Image

def create_time_slice_image(video_path, output_image_path, y):
    # Load the video
    video = VideoFileClip(video_path)

    # Video properties
    duration = video.duration  # Duration in seconds
    fps = video.fps  # Frames per second
    width, height = video.size  # Width and height of the video
    
    y = height // 2

    # Calculate the total number of frames
    total_frames = int(duration * fps)
    
    # Initialize an empty array to store the pixel rows
    pixel_rows = []

    # Extract pixel row from each frame
    for frame_number in range(total_frames):
        frame = video.get_frame(frame_number / fps)
        pixel_row = frame[y, :, :]  # Extract the pixel row at height y
        if pixel_row.shape == (width, 3):  # Ensure each pixel row is 1920x3
            pixel_rows.append(pixel_row.reshape(1, width, 3))
        else:
            print(f"Frame {frame_number}: Incorrect pixel row shape: {pixel_row.shape}")

    # Combine all rows into a single image
    final_image = np.vstack(pixel_rows)
    
    # Log the final image size
    print(f"Final image size: {final_image.shape}")

    # Convert to PIL Image for saving
    final_image_pil = Image.fromarray(final_image)

    # Save the final image
    final_image_pil.save(output_image_path)

# Example usage
video_path = '1900-edit720p25fps-black001-fixed.mp4'
output_image_path = 'time_slice_image_001.png'
y = 720  # Desired pixel row height
create_time_slice_image(video_path, output_image_path, y)

