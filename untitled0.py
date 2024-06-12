# -*- coding: utf-8 -*-
"""
Created on Mon May  8 02:48:48 2023

@author: yuuki
"""

import cv2
import time

movie = cv2.VideoCapture('./movie/-25816.mp4')

red = (0, 0, 255) # 枠線の色
before = None # 前回の画像を保存する変数
fps = int(movie.get(cv2.CAP_PROP_FPS)) #動画のFPSを取得
size = () #動画の縦横サイズを入れる変数
width = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)) #横幅
height = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT)) #高さ
size = (width,height)
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #保存形式
print(size)
writer = cv2.VideoWriter('outtest2.mp4', fmt, fps, size) #ライター作成

while True:
    # 画像を取得
    ret, frame = movie.read()
    # 再生が終了したらループを抜ける
    if ret == False: break
    # 白黒画像に変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if before is None:
        before = gray.astype("float")
        continue
    #現在のフレームと移動平均との差を計算
    cv2.accumulateWeighted(gray, before, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(before))
    #frameDeltaの画像を２値化
    thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
    #輪郭のデータを得る
    contours = cv2.findContours(thresh,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE)[0]

    # 差分があった点を画面に描く
    for target in contours:
        x, y, w, h = cv2.boundingRect(target)
        if w < 30: continue # 小さな変更点は無視
        cv2.rectangle(frame, (x, y), (x+w, y+h), red, 2)

    #ウィンドウでの再生速度を元動画と合わせる
    time.sleep(1/fps)
    # ウィンドウで表示
    cv2.imshow('target_frame', frame)
    # 画像を1フレーム分として書き込み
    writer.write(frame)
    # Enterキーが押されたらループを抜ける
    if cv2.waitKey(1) == 13: break

writer.release()
movie.release()
cv2.destroyAllWindows() # ウィンドウを破棄