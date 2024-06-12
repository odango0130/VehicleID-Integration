# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 09:20:22 2024

@author: yuuki

"""



import cv2

"""
cap = cv2.VideoCapture(r"C:\ForMe\1900_tracked_crop2-5_resized.mp4")

totalframecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("このビデオの合計フレーム数は：" , totalframecount)
"""


import numpy as np

"""
# 予測関数の定義
def predict_lost_bbox_with_least_squares(lost_bboxes, frame_number, history_length=5):
    for lost_id, bboxes in lost_bboxes.items():
        if len(bboxes) >= history_length:
            past_coords = np.array([b['bbox'][:2] for b in bboxes[-history_length:]])
            past_frames = np.array([b['frame_number'] for b in bboxes[-history_length:]]).reshape(-1, 1)

            A = np.hstack([past_frames, np.ones(past_frames.shape)])
            vx, vy = np.linalg.lstsq(A, past_coords, rcond=None)[0][:, 0]

            predicted_x = bboxes[-1]['bbox'][0] + vx
            predicted_y = bboxes[-1]['bbox'][1] + vy
            predicted_bbox = [predicted_x, predicted_y, bboxes[-1]['bbox'][2] + vx, bboxes[-1]['bbox'][3] + vy]

            lost_bboxes[lost_id].append({'bbox': predicted_bbox, 'frame_number': frame_number})
"""

def predict_lost_bbox_with_least_squares(lost_bboxes, frame_number, history_length=5):
    for lost_id, bboxes in lost_bboxes.items():
        if len(bboxes) >= history_length:
            past_coords = np.array([b['bbox'][:2] for b in bboxes[-history_length:]])
            past_frames = np.array([b['frame_number'] for b in bboxes[-history_length:]]).reshape(-1, 1)

            A = np.hstack([past_frames, np.ones(past_frames.shape)])
            
            # x座標とy座標の速度を別々に計算
            vx, intercept_x = np.linalg.lstsq(A, past_coords[:, 0], rcond=None)[0]
            vy, intercept_y = np.linalg.lstsq(A, past_coords[:, 1], rcond=None)[0]

            # 次のフレームでの座標を予測
            predicted_x = bboxes[-1]['bbox'][0] + vx
            predicted_y = bboxes[-1]['bbox'][1] + vy
            predicted_bbox = [predicted_x, predicted_y, bboxes[-1]['bbox'][2] + vx, bboxes[-1]['bbox'][3] + vy]

            lost_bboxes[lost_id].append({'bbox': predicted_bbox, 'frame_number': frame_number})


# テストデータの作成
lost_bboxes = {
    1: [
        {'bbox': [100, 100, 110, 110], 'frame_number': 1},
        {'bbox': [101, 101, 111, 111], 'frame_number': 2},
        {'bbox': [102, 102, 112, 112], 'frame_number': 3},
        {'bbox': [103, 103, 113, 113], 'frame_number': 4},
        {'bbox': [104, 104, 114, 114], 'frame_number': 5},
    ]
}

# 関数をテスト実行
predict_lost_bbox_with_least_squares(lost_bboxes, 6)

# 結果の表示
print(lost_bboxes)
