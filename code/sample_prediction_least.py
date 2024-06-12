#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2
import numpy as np
import os
from sklearn.linear_model import LinearRegression


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
    # 変数の初期化
    frame_number = 0
    active_bboxes = {}
    lost_bboxes = {}
    id_mapping = {}

    # 距離の閾値（ユーザーが指定）
    distanceThreshold = 150  # 例として50ピクセルを設定
    discardFrames = 50  # 例として10フレームを設定

    
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
        outpath = os.path.splitext(args.movie)[0] + "_predict-2.mp4"
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

        # 以降の処理（フロー1~7） ##########################################
        frame_number += 1
        
        # 2. ID置き換え
        t_ids = [id_mapping.get(t_id, t_id) for t_id in t_ids]
        
        # 6. 新たなIDの紐づけ判定
        match_new_id(t_ids, t_bboxes, active_bboxes, lost_bboxes, id_mapping, frame_number, distanceThreshold)
        
        
        # 3. lost判定と4. lostした対象のbbox座標予測
        update_bboxes(t_ids, t_bboxes, active_bboxes, lost_bboxes, frame_number, id_mapping, discardFrames)
        predict_lost_bbox_with_least_squares(lost_bboxes, frame_number)
        
        # 5. 再発見したlostを削除
        # この部分はupdate_bboxes関数内で処理
        

        # 7. デバッグ描画
        # draw_debug関数をここで呼び出す
        # debug_image = draw_debug(debug_image, lost_bboxes)
        debug_image = draw_debug2(debug_image, active_bboxes)
        # 書き込み
        writer.write(debug_image)
        
        # 処理時間計測 ######################################################
        elapsed_time = time.time() - start_time
        print(f"frame: {frame_number}, elapsed time: {elapsed_time:.3f} seconds")
        print(id_mapping)
        #---------------------------------------------------------#
    cap.release()
    writer.release()
    
    
def update_bboxes(t_ids, t_bboxes, active_bboxes, lost_bboxes, frame_number, id_mapping, discardFrames):
    current_ids = set(t_ids)
    active_ids = set(active_bboxes.keys())
    lost_ids = set(lost_bboxes.keys())

    # active_bboxesの更新
    for active_id in active_ids - current_ids:
        # activeからlostへ移動
        lost_bboxes[active_id] = active_bboxes.pop(active_id)

    # lost_bboxesの更新
    for lost_id in list(lost_ids):
        if frame_number - lost_bboxes[lost_id][-1]['frame_number'] > discardFrames:
            # 古いレコードを削除
            del lost_bboxes[lost_id]

    # 新しいbboxの追加
    for t_id, bbox in zip(t_ids, t_bboxes):
        if t_id in id_mapping:
            t_id = id_mapping[t_id]

        if t_id in active_bboxes:
            active_bboxes[t_id].append({'bbox': bbox, 'frame_number': frame_number})
        else:
            active_bboxes[t_id] = [{'bbox': bbox, 'frame_number': frame_number}]


def predict_lost_bbox_with_least_squares(lost_bboxes, frame_number, history_length=5):
    for lost_id, bboxes in lost_bboxes.items():
        if len(bboxes) >= history_length:
            # 過去の座標とフレーム番号を取得
            past_coords = np.array([b['bbox'][:2] for b in bboxes[-history_length:]])
            past_frames = np.array([b['frame_number'] for b in bboxes[-history_length:]]).reshape(-1, 1)

            # 最小二乗法でパラメータを推定
            A = np.hstack([past_frames, np.ones(past_frames.shape)])
            vx, vy = np.linalg.lstsq(A, past_coords, rcond=None)[0][:, 0]

            # 次のフレームでの座標を予測
            predicted_x = bboxes[-1]['bbox'][0] + vx
            predicted_y = bboxes[-1]['bbox'][1] + vy
            predicted_bbox = [predicted_x, predicted_y, bboxes[-1]['bbox'][2] + vx, bboxes[-1]['bbox'][3] + vy]

            lost_bboxes[lost_id].append({'bbox': predicted_bbox, 'frame_number': frame_number})

def match_new_id(t_ids, t_bboxes, active_bboxes, lost_bboxes, id_mapping, frame_number, distanceThreshold):
    for t_id, bbox in zip(t_ids, t_bboxes):
        print("t_id-for")
        if t_id not in active_bboxes:       #新しいIDか
            min_distance = float('inf')
            matched_id = None
            print("lost_id-for")
            for lost_id, bboxes in lost_bboxes.items():     #lostID全てをforで回す
                distance = np.linalg.norm(np.array(bbox[:2]) - np.array(bboxes[-1]['bbox'][:2]))
                if distance < min_distance and distance < distanceThreshold:    #lost対象と近いかと閾値より近いか
                    print("distance-if")
                    min_distance = distance
                    matched_id = lost_id

            if matched_id is not None:      #紐づけが行えたら登録
                print("matching-success")
                id_mapping[t_id] = matched_id
                active_bboxes[matched_id] = [{'bbox': bbox, 'frame_number': frame_number}]
                del lost_bboxes[matched_id]

def draw_debug(image, lost_bboxes):
    """
    デバッグ用の画像に、失われた（lost）バウンディングボックスの左上の座標を描画する関数

    :param image: 描画対象の画像
    :param lost_bboxes: 失われたバウンディングボックスの辞書（キーはID）
    """
    for object_id, bbox_info in lost_bboxes.items():
        # 最新のバウンディングボックスを取得
        bbox = bbox_info[-1]['bbox']
        top_left = (int(bbox[0]), int(bbox[1]))

        # 左上の座標を描画
        cv2.circle(image, top_left, 5, (0, 0, 255), -1)

        # IDを描画
        cv2.putText(image, str(object_id), (top_left[0] + 10, top_left[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image

def draw_debug2(image, bboxes):
    """
    デバッグ用の画像にバウンディングボックスとIDを描画する関数

    :param image: 描画対象の画像
    :param bboxes: トラッキング中のバウンディングボックスの辞書（キーはID）
    """
    for object_id, bbox_info in bboxes.items():
        # 最新のバウンディングボックスを取得
        bbox = bbox_info[-1]['bbox']
        top_left = (int(bbox[0]), int(bbox[1]))
        bottom_right = (int(bbox[2]), int(bbox[3]))

        # バウンディングボックスを描画
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        # IDを描画
        cv2.putText(image, str(object_id), (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image



    
    
def create_center_adjust_rolling_video(
        frames, fps, scale_factors, precenters_x, precenters_y,
        cap_width, cap_height,
        desired_width, desired_height,
        per_median
        ):
    frame_count, frame_height, frame_width, _ = frames.shape
    # 復元用に各要素の逆数を計算
    inverse_scaled_factors = [1/x for x in scale_factors]  
    # ビデオライターを初期化
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('center_adjust_video.mp4', fourcc, fps, (frame_width, frame_height))
    
    # 全フレーム中央値画像
    #median_frame = compute_median_ignore_zeros_color(frames, 1, frame_count)
    #median_frames = []
    print("pre-frames")
    # バッファ中央値フレーム作成
    median_frames = create_rolling_median_frames(frames, per_median, fps)
    print("after-frames")
    
    # median_frameの中心座標
    mimage_center_x = cap_width
    mimage_center_y = cap_height
    
    skip_start = per_median/2  # スキップする最初の要素数
    skip_end = per_median/2    # 無視する最後の要素数

    index = 0
    
    print(len(median_frames))
    
    # 拡大縮小実行for文
    for scale_factor, center_x, center_y in zip(inverse_scaled_factors, precenters_x, precenters_y):
        if index < skip_start or index > len(median_frames)-1:
            index += 1
            print("例外処理")
            continue
        # 拡大縮小後の画像の解像度
        scaled_width = int(cap_width*2 * scale_factor)
        scaled_height = int(cap_height*2 * scale_factor)
        # 拡大縮小を行い
        if scale_factor < 1:            # 現在ののbboxが初期より小さくなったとき
            scaled_image = cv2.getRectSubPix(           # 外側の画素を削って全体に対するbboxの割合を初期と同じに
                median_frames[index], (scaled_width, scaled_height), (mimage_center_x, mimage_center_y)
            )
        elif scale_factor > 1:          # 今のbboxが初期より大きくなったとき
            extend_width = scaled_width - cap_width*2
            extend_height = scaled_height - cap_height*2
            extend_top = int(extend_height*0.5)         #外側にパディングをつけて全体に対する…
            extend_bottom = int(extend_height*0.5)
            extend_left = int(extend_width*0.5)
            extend_right = int(extend_width*0.5)
            scaled_image =  cv2.copyMakeBorder(
                median_frames[index], extend_top, extend_bottom, extend_left, extend_right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif scale_factor == 1:
            scaled_image = median_frames[index]
        # フレームのリサイズ（指定された解像度に変更）
        resized_frame = cv2.resize(scaled_image, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)
        cropped_frame = get_rect_with_black_border(resized_frame, (center_x, center_y), (desired_width, desired_height))
        #############################
        #print("resized")           #
        #print(resized_frame.shape) #
        #print("cropped")           #
        #print(cropped_frame.shape) #
        #############################
        out.write(cropped_frame)
        print("フレーム追加")
        print(index)
        index += 1
        
    
    
    
def create_center_adjust_video(
        frames, fps, scale_factors, precenters_x, precenters_y,
        cap_width, cap_height,
        desired_width, desired_height
        ):
    frame_count, frame_height, frame_width, _ = frames.shape
    # 復元用に各要素の逆数を計算
    inverse_scaled_factors = [1/x for x in scale_factors]  
    # ビデオライターを初期化
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('center_adjust_video.mp4', fourcc, fps, (frame_width, frame_height))
    # 全フレーム中央値画像
    median_frame = compute_median_ignore_zeros_color(frames, 1, frame_count)
    median_frames = []
    # median_frameの中心座標
    mimage_center_x = cap_width
    mimage_center_y = cap_height
    # 拡大縮小実行for文
    for scale_factor, center_x, center_y in zip(inverse_scaled_factors, precenters_x, precenters_y):
        # 拡大縮小後の画像の解像度
        scaled_width = int(cap_width*2 * scale_factor)
        scaled_height = int(cap_height*2 * scale_factor)
        # 拡大縮小実行
        if scale_factor < 1:            # 現在ののbboxが初期より小さくなったとき
            scaled_image = cv2.getRectSubPix(           # 外側の画素を削って全体に対するbboxの割合を初期と同じに
                median_frame, (scaled_width, scaled_height), (mimage_center_x, mimage_center_y)
            )
        elif scale_factor > 1:          # 今のbboxが初期より大きくなったとき
            extend_width = scaled_width - cap_width*2
            extend_height = scaled_height - cap_height*2
            extend_top = int(extend_height*0.5)         #外側にパディングをつけて全体に対する…
            extend_bottom = int(extend_height*0.5)
            extend_left = int(extend_width*0.5)
            extend_right = int(extend_width*0.5)
            scaled_image =  cv2.copyMakeBorder(
                median_frame, extend_top, extend_bottom, extend_left, extend_right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif scale_factor == 1:
            scaled_image = median_frame
        # フレームのリサイズ（指定された解像度に変更）
        resized_frame = cv2.resize(scaled_image, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)
        cropped_frame = get_rect_with_black_border(resized_frame, (center_x, center_y), (desired_width, desired_height))
        #############################
        #print("resized")
        #print(resized_frame.shape)
        #print("cropped")
        #print(cropped_frame.shape)
        #############################
        out.write(cropped_frame)
        
        
def create_rolling_median_frames(frames, per_median, fps):
    print("create_frames")
    frame_count, frame_height, frame_width, _ = frames.shape
    
    rolling_frames = []
    
    # 中央値を計算するフレーム数
    frame_per_median = per_median
    
    # 中央値画像を計算してビデオに追加するループ
    for i in range(0, frame_count - frame_per_median):
        median_frame = compute_median_ignore_zeros_rgb(frames, i, i + frame_per_median)
        rolling_frames.append(median_frame)
    return rolling_frames



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
    


def compute_median_ignore_zeros_rgb(frames, start, end):
    print("compute_median_rgb")
    print(start)
    # 指定されたフレーム範囲を抽出
    selected_frames = frames[start:end]

    # RGBが全て0の画素を検出するマスクを作成
    # (高さ, 幅, チャンネル数) の形状で、各チャンネルについて全て0かどうかをチェック
    mask = np.all(selected_frames == 0, axis=-1)
    

    # マスクされた配列を作成
    # マスクは (高さ, 幅) の形状なので、チャンネル次元に沿って繰り返す
    masked_frames = np.ma.array(selected_frames, mask=np.repeat(mask[:, :, :, np.newaxis], 3, axis=3))
    print("masked_frames")
    print(masked_frames)

    # masked_arrayを使用して中央値を計算
    # axis=0 でフレーム次元に沿って計算
    median_frame = np.ma.median(masked_frames, axis=0).filled(0)

    return median_frame.astype(np.uint8)

# この関数は、RGB全てが0の画素を無視して各画素ごとの中央値を計算します。
# このコードも、実際のデータ構造に応じて調整が必要な場合があります。


    
def compute_median_ignore_zeros_color(frames, start, end):
    print("compute_median")
    print(start)
    # 指定されたフレーム範囲を抽出
    selected_frames = frames[start:end]

    # 各位置の0でない値を格納するリストを作成
    def custom_median(arr):
        non_zero_arr = arr[arr != 0]
        if non_zero_arr.size == 0:
            return 0  # 全ての値が0の場合は0を返す
        else:
            return np.median(non_zero_arr)

    # 各位置ごとに中央値を計算
    median_frame = np.apply_along_axis(custom_median, axis=0, arr=selected_frames)

    return median_frame.astype(np.uint8)

    
    
def compute_median_for_frames(frames, start, end):
    # 指定されたフレーム範囲を抽出
    selected_frames = frames[start:end]

    # 中央値を計算
    median_frame = np.median(selected_frames, axis=0).astype(np.uint8)

    return median_frame


def get_rect_with_black_border(image, center, size):
    img_height, img_width = image.shape[:2]

    # 矩形の左上と右下の座標を計算
    x1 = int(center[0] - size[0] / 2)
    y1 = int(center[1] - size[1] / 2)
    x2 = x1 + size[0]
    y2 = y1 + size[1]

    # 切り出す範囲を画像の範囲内に調整
    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(img_width, x2)
    y2_clip = min(img_height, y2)

    # 調整された範囲で矩形領域を切り出し
    cropped = image[y1_clip:y2_clip, x1_clip:x2_clip]

    # はみ出た部分のサイズを計算
    top = max(0, -y1)
    bottom = max(0, y2 - img_height)
    left = max(0, -x1)
    right = max(0, x2 - img_width)

    # 黒いボーダーを追加
    cropped = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return cropped


"""
def get_rect_with_black_border(image, center, size):
    # 画像のサイズを取得
    img_height, img_width = image.shape[:2]

    # 矩形の左上と右下の座標を計算
    x1 = int(center[0] - size[0] / 2)
    y1 = int(center[1] - size[1] / 2)
    x2 = x1 + size[0]
    y2 = y1 + size[1]

    # 矩形が画像内に完全に収まるように調整
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_width, x2), min(img_height, y2)

    # 矩形領域を切り出し
    cropped = image[y1:y2, x1:x2]

    # はみ出た部分を黒で埋める
    if cropped.shape[0] != size[1] or cropped.shape[1] != size[0]:
        cropped = cv2.copyMakeBorder(cropped, top=max(0, -y1), bottom=max(0, y2 - img_height),
                                     left=max(0, -x1), right=max(0, x2 - img_width),
                                     borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return cropped
"""



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


def draw_debug_old(
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
