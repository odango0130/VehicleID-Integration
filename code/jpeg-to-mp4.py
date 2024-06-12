# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 17:28:31 2024

@author: yuuki
"""

import cv2
import os

# 画像が保存されているディレクトリ
image_folder = 'D:\\ForMe\\research\\Insight-MVT_Annotation_Test\\MVI_39361'
# 出力する動画のファイル名
video_name = 'MVI_39361.mp4'

# img00001.jpg, img00002.jpg, ... のようなファイル名に対応
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort(key=lambda x: int(x.replace('img', '').split('.')[0]))

# 画像から動画の設定を取得
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# ビデオライターの設定 (MP4Vコーデックを使用)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter(video_name, fourcc, 25, (width, height))

# 各画像を動画に追加
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()