#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
影视解说项目 - 第一步：视频预处理与结构分析（修复版）
适配 PaddleOCR v2.7+，解决 use_gpu 和 use_angle_cls 弃用问题
"""

import os
import json
import cv2
import numpy as np
import ffmpeg
from scenedetect import detect, ContentDetector

# ========================
# 配置区
# ========================
VIDEO_PATH = "input.mp4"
OUTPUT_DIR = "output_step1"
EXTRACT_SUBTITLES = False
SCENE_MIN_DURATION = 1.0
THRESHOLD = 27

# ========================
# 为 PaddleOCR 设置环境（避免卡在模型源检查）
# ========================
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# 延迟导入 PaddleOCR，仅在需要时加载
# 替换 get_ocr_engine 函数为：
def get_ocr_engine():
    try:
        from paddleocr import PaddleOCR
        # 只保留 lang，其他全靠默认
        ocr = PaddleOCR(lang="ch")
        return ocr
    except Exception as e:
        print(f"⚠️ PaddleOCR 初始化失败: {e}")
        return None
    
# ========================
# 工具函数（保持不变）
# ========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def extract_audio_segment(input_video, start_time, end_time, output_audio):
    try:
        (
            ffmpeg
            .input(input_video, ss=start_time, to=end_time)
            .output(output_audio, ac=1, ar=16000, loglevel="quiet")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise

def get_best_frame(video_path, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)
    best_score = -1
    best_frame = None
    total_frames = end_frame - start_frame
    step = max(1, total_frames // 30)

    for i in range(start_frame, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if score > best_score:
            best_score = score
            best_frame = frame.copy()

    cap.release()
    return best_frame

def extract_subtitle_from_frame(frame, ocr_engine):
    h, w = frame.shape[:2]
    roi = frame[int(0.82 * h):int(0.98 * h), :]  # 字幕区域
    result = ocr_engine.predict(roi)  # ✅ 稳定调用方式
    
    if not result or len(result) == 0:
        return ""
    
    texts = []
    # 新版返回结构: [[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ('text', confidence)], ...]
    for item in result:
        if isinstance(item, list) and len(item) >= 2:
            text_info = item[1]
            if isinstance(text_info, tuple) and len(text_info) == 2:
                text, conf = text_info
                if isinstance(conf, (int, float)) and conf > 0.8:
                    texts.append(str(text))
    return " ".join(texts).strip()

# ========================
# 主流程
# ========================




def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"输入视频不存在: {VIDEO_PATH}")

    ensure_dir(OUTPUT_DIR)
    audio_dir = os.path.join(OUTPUT_DIR, "audio")
    frames_dir = os.path.join(OUTPUT_DIR, "frames")
    subtitles_dir = os.path.join(OUTPUT_DIR, "subtitles")
    ensure_dir(audio_dir)
    ensure_dir(frames_dir)
    if EXTRACT_SUBTITLES:
        ensure_dir(subtitles_dir)

    # 初始化 OCR（仅在需要时）
    ocr_engine = None
    if EXTRACT_SUBTITLES:
        print("初始化 PaddleOCR（首次运行会下载模型）...")
        ocr_engine = get_ocr_engine()
        if ocr_engine is None:
            print("❌ 跳过字幕 OCR")
            EXTRACT_SUBTITLES_GLOBAL = False
        else:
            EXTRACT_SUBTITLES_GLOBAL = True
    else:
        EXTRACT_SUBTITLES_GLOBAL = False

    # 场景检测
    print("正在进行场景分割...")
    scene_list = detect(
        VIDEO_PATH,
        detector=ContentDetector(
            threshold=THRESHOLD,
            min_scene_len=int(SCENE_MIN_DURATION * 30)
        )
    )

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    scenes_meta = []
    print(f"共检测到 {len(scene_list)} 个场景，开始处理...")

    for idx, (start_frame, end_frame) in enumerate(scene_list):
        start_time = start_frame.get_frames() / fps
        end_time = end_frame.get_frames() / fps
        duration = end_time - start_time

        if duration < SCENE_MIN_DURATION:
            continue

        scene_id = f"{idx:04d}"
        print(f"处理场景 {scene_id} ({start_time:.1f}s - {end_time:.1f}s)")

        # 提取音频
        audio_path = os.path.join(audio_dir, f"scene_{scene_id}.wav")
        extract_audio_segment(VIDEO_PATH, start_time, end_time, audio_path)

        # 提取最佳帧
        best_frame = get_best_frame(VIDEO_PATH, start_frame.get_frames(), end_frame.get_frames())
        frame_path = ""
        if best_frame is not None:
            frame_path = os.path.join(frames_dir, f"scene_{scene_id}.jpg")
            cv2.imwrite(frame_path, best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # OCR 字幕
        subtitle_text = ""
        if EXTRACT_SUBTITLES_GLOBAL and best_frame is not None:
            subtitle_text = extract_subtitle_from_frame(best_frame, ocr_engine)

        scenes_meta.append({
            "scene_id": idx,
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
            "duration": round(duration, 3),
            "audio_path": audio_path,
            "frame_path": frame_path,
            "subtitle_text": subtitle_text
        })

    meta_path = os.path.join(OUTPUT_DIR, "scenes.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(scenes_meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 第一步完成！结果保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()