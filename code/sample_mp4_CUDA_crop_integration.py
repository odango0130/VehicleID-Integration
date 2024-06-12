#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2
import numpy as np
import os

from yolox.yolox_onnx import YoloxONNX
from bytetrack.mc_bytetrack import MultiClassByteTrack

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    # YOLOX parameters
    parser.add_argument(
        "--yolox_model",
        type=str,
        default='model/yolox_nano.onnx',
    )
    parser.add_argument(
        '--input_shape',
        type=str,
        default="416,416",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.3,
        help='Class confidence',
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.45,
        help='NMS IoU threshold',
    )
    parser.add_argument(
        '--nms_score_th',
        type=float,
        default=0.1,
        help='NMS Score threshold',
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )

    # motpy parameters
    parser.add_argument(
        "--track_thresh",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--track_buffer",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--min_box_area",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--mot20",
        action="store_true",
    )

    args = parser.parse_args()

    return args


class dict_dot_notation(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def main():
    # 追跡対象のIDを手動で指定
    tracking_target_id = "2_5"
    # トラッキング開始フラグ
    tracking_started = False
    # 追跡対象のIDバウンディングボックスの最初の横幅
    initial_bbox_width = None


    
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.movie is not None:
        cap_device = args.movie

    # YOLOX parameters
    model_path = args.yolox_model
    input_shape = tuple(map(int, args.input_shape.split(',')))
    score_th = args.score_th
    nms_th = args.nms_th
    nms_score_th = args.nms_score_th
    with_p6 = args.with_p6

    # ByteTrack parameters
    track_thresh = args.track_thresh
    track_buffer = args.track_buffer
    match_thresh = args.match_thresh
    min_box_area = args.min_box_area
    mot20 = args.mot20

    # カメラ準備 ###############################################################
    cap = cv2.VideoCapture(cap_device)
    #setの部分をいじりたくないけどcap_widthとcap_heightを手入力したくないから下2行追加
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    


    # 出力ファイル
    if args.movie is not None:
        outpath = os.path.splitext(args.movie)[0] + "_tracked_crop.mp4"
    else:
        outpath = "camera.mp4"

    writer = cv2.VideoWriter(
        outpath,
        cv2.VideoWriter_fourcc(*'mp4v'),
        cap_fps, (int(cap_width), int(cap_height))
    )
    
    

    # モデルロード #############################################################
    yolox = YoloxONNX(
        model_path=model_path,
        input_shape=input_shape,
        class_score_th=score_th,
        nms_th=nms_th,
        nms_score_th=nms_score_th,
        with_p6=with_p6,
        providers=['CUDAExecutionProvider'],
    )

    # ByteTrackerインスタンス生成
    tracker = MultiClassByteTrack(
        fps=cap_fps,
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        min_box_area=min_box_area,
        mot20=mot20,
    )

    # トラッキングID保持用変数
    track_id_dict = {}

    # COCOクラスリスト読み込み
    with open('coco_classes.txt', 'rt') as f:
        coco_classes = f.read().rstrip('\n').split('\n')

    # トラッカー別のフレーム保持用
    track_id_frames = {}
    
    # 画素中央値処理に引き渡すフレーム保持用
    frames = []
    
    # 画素中央値処理後に中心を戻すための元の中心座標保持用
    precenters_x = []
    precenters_y = []
    scale_factors = []



    while True:
        start_time = time.time()

        # カメラキャプチャ ################################################
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # 推論実施 ########################################################
        # Object Detection
        bboxes, scores, class_ids = yolox.inference(frame)
        

        # Multi Object Tracking
        t_ids, t_bboxes, t_scores, t_class_ids = tracker(
            frame,
            bboxes,
            scores,
            class_ids,
        )

    
       #-------------------車中心画像作成---------------------------------#
        # 特定のトラッキングIDのバウンディングボックスを探す
        target_bbox = None
                
                
        for tracker_id, bbox in zip(t_ids, t_bboxes):
            if tracker_id == tracking_target_id:
                target_bbox = bbox
                if not tracking_started:
                    tracking_started = True  # トラッキング開始
                    initial_bbox_width = target_bbox[2] - target_bbox[0]
                break
        
        elapsed_time = time.time() - start_time

        
        if target_bbox is None and not tracking_started:
            # トラッキング対象がまだ見つかっていない場合は、何もしない
            continue
        elif target_bbox is None:
            # トラッキング対象が見失った場合、録画を停止
            print(f"Tracking target ID {tracking_target_id} lost. Ending recording.")
            break

        # バウンディングボックスの中心を計算
        center_x = int((target_bbox[0] + target_bbox[2]) / 2)
        center_y = int((target_bbox[1] + target_bbox[3]) / 2)

        # 画像をクロップして中心に配置
        """
        cropped_image = cv2.getRectSubPix(
            debug_image, (cap_width, cap_height), (center_x, center_y)
        )
        """

        # パディング
        top_border = max(cap_height - center_y, 0)
        bottom_border = max(center_y, 0)
        left_border = max(cap_width - center_x, 0)
        right_border = max(center_x, 0)
        
        # 元の中心の位置を記録
        precenter_x = cap_width/2 + left_border
        precenter_y = cap_height/2 + top_border
    
        # 境界を画像に追加
        bordered_image = cv2.copyMakeBorder(
            debug_image, top_border, bottom_border, left_border, right_border,
            cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # 現在のバウンディングボックスの幅を計算
        current_bbox_width = target_bbox[2] - target_bbox[0]
        
        # スケーリング係数を計算（初期サイズと現在サイズの比）
        scale_factor = current_bbox_width / initial_bbox_width
        
        # 拡大縮小後の画像の解像度
        scaled_width = int(cap_width*2 * scale_factor)
        scaled_height = int(cap_height*2 * scale_factor)
        
        if scale_factor < 1:            # 現在ののbboxが初期より小さくなったとき
            scaled_image = cv2.getRectSubPix(           # 外側の画素を削って全体に対するbboxの割合を初期と同じに
                bordered_image, (scaled_width, scaled_height), (center_x, center_y)
            )
        elif scale_factor > 1:          # 今のbboxが初期より大きくなったとき
            extend_width = scaled_width - cap_width*2
            extend_height = scaled_height - cap_height*2
            extend_top = int(extend_height*0.5)         #外側にパディングをつけて全体に対する…
            extend_bottom = int(extend_height*0.5)
            extend_left = int(extend_width*0.5)
            extend_right = int(extend_width*0.5)
            scaled_image =  cv2.copyMakeBorder(
                bordered_image, extend_top, extend_bottom, extend_left, extend_right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif scale_factor == 1:
            scaled_image = bordered_image
            
        # 画像をリサイズする解像度を指定
        desired_width, desired_height = int(cap_width*2), int(cap_height*2) 
        
        # フレームのリサイズ（指定された解像度に変更）
        resized_frame = cv2.resize(scaled_image, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)
            
            
            
            
        print(resized_frame.shape)
            

        # トラッキングが開始している場合のみ書き込み
        if tracking_started:
            frames.append(resized_frame)
            precenters_x.append(precenter_x)
            precenters_y.append(precenter_y)
            scale_factors.append(scale_factor)
            #writer.write(resized_frame)
        #---------------------------------------------------------#

    print(scale_factors)
    frames_array = np.array(frames)
    create_rolling_median_video(frames_array, 12, cap_fps)
    cap.release()
    writer.release()

def create_rolling_median_video(frames, per_median, fps):
    frame_count, frame_height, frame_width, _ = frames.shape
    # ビデオライターを初期化
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('rolling_median_video.mp4', fourcc, fps, (frame_width, frame_height))
    
    # 中央値を計算するフレーム数
    frame_per_median = per_median
    
    # 中央値画像を計算してビデオに追加するループ
    for i in range(0, frame_count - frame_per_median):
        median_frame = compute_median_for_frames(frames, i, i + frame_per_median)
        out.write(median_frame)
    
    # リソースを解放
    out.release()
    
def compute_median_for_frames(frames, start, end):
    # 指定されたフレーム範囲を抽出
    selected_frames = frames[start:end]

    # 中央値を計算
    median_frame = np.median(selected_frames, axis=0).astype(np.uint8)

    return median_frame
    



## 以下使用しない関数 ##

def get_id_color(index):
    temp_index = abs(int(index)) * 3
    color = ((37 * temp_index) % 255, (17 * temp_index) % 255,
             (29 * temp_index) % 255)
    return color


def write_track_id_wise_video(
    track_id_frames_dict,
    base_dir, cap_fps
):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for track_id, track_frames in track_id_frames_dict.items():
        width, height = 0, 0
        for frame in track_frames:
            height = max(frame.shape[0], height)
            width = max(frame.shape[1], width)

        writer = cv2.VideoWriter(
            f"{base_dir}/{track_id}.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'),
            cap_fps, (width, height)
        )

        for frame in track_frames:
            pad_left = (width - frame.shape[1])//2
            pad_right = width - frame.shape[1] - pad_left
            pad_top = (height - frame.shape[0])//2
            pad_bottom = height - frame.shape[0] - pad_top
            frame_pad = cv2.copyMakeBorder(frame, 
                            pad_top, pad_bottom, pad_left, pad_right, 
                            cv2.BORDER_CONSTANT, (0, 0, 0))
            writer.write(frame_pad)
        writer.release()


def draw_debug(
    image,
    elapsed_time,
    score_th,
    trakcer_ids,
    bboxes,
    scores,
    class_ids,
    track_id_dict,
    coco_classes,
    tracker_id_frames,
):
    debug_image = copy.deepcopy(image)

    for tracker_id, bbox, score, class_id in zip(trakcer_ids, bboxes, scores,
                                                 class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if score_th > score:
            continue

        color = get_id_color(int(track_id_dict[tracker_id]))

        # バウンディングボックス
        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            color,
            thickness=2,
        )

        # トラックID、スコア
        score_txt = str(round(score, 2))
        text = 'Track ID:%s(%s)' % (int(track_id_dict[tracker_id]), score_txt)
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness=2,
        )
        # クラスID
        text = 'Class ID:%s(%s)' % (class_id, coco_classes[class_id])
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness=2,
        )

        # ID別のフレームを記録
        crop = debug_image[max(y1-30,0):y2, x1:x2, :]
        if not tracker_id in tracker_id_frames.keys():
            tracker_id_frames[tracker_id] = [crop]
        else:
            tracker_id_frames[tracker_id].append(crop)

    # 推論時間
    text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000)
    text = text + 'ms'
    debug_image = cv2.putText(
        debug_image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    return debug_image


## ここまで使用しない関数 ##



if __name__ == '__main__':
    main()
