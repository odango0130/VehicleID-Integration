# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:10:33 2023

@author: odayuki
"""

from moviepy.editor import VideoFileClip

def reduce_resolution(video_path, output_video_path, resize_factor=0.5):
    # Load the video
    video = VideoFileClip(video_path)
    
    # Resize the video
    resized_video = video.resize(resize_factor)

    # Write the resized video to a new file
    resized_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

# Example usage
video_path = '1900_tracked_crop2-5.mp4'
output_video_path = '1900_tracked_crop2-5_resized.mp4'
reduce_resolution(video_path, output_video_path, resize_factor=0.2)