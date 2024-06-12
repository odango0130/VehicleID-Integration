# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:26:47 2023

@author: odayuki
"""

import cv2

# 動画ファイルのパスを指定
video_path = '1900_tracked_crop2-5_fixed.mp4'  # ここに動画ファイルのパスを入力

# VideoCaptureオブジェクトを作成
cap = cv2.VideoCapture(video_path)

# 動画が正常に開かれたか確認
if not cap.isOpened():
    print("エラー：動画を開けませんでした。")
else:
    # 動画の総フレーム数を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"動画の総フレーム数: {total_frames}")

# VideoCaptureオブジェクトを解放
cap.release()
