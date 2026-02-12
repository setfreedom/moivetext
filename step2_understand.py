#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
影视解说项目 - 第二步：多模态内容理解（ASR + 视觉描述）
输入：output_step1/scenes.json + audio/ + frames/
输出：output_step2/scenes_enhanced.json
"""

import os
import json
import torch
from PIL import Image
from faster_whisper import WhisperModel
from transformers import BlipProcessor, BlipForConditionalGeneration

# ========================
# 配置
# ========================
INPUT_META = "output_step1/scenes.json"
OUTPUT_DIR = "output_step2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设备自动选择
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"🚀 使用设备: {device} ({compute_type})")

# ========================
# 初始化模型（懒加载，避免无文件时报错）
# ========================
print("正在加载 Whisper 模型 (large-v3)...")
whisper_model = WhisperModel(
    r"G:\models\faster-whisper-large-v3",  # 👈 本地路径
    device=device,
    compute_type=compute_type,
    local_files_only=True  # 强制离线
)

print("正在加载 BLIP 视觉描述模型...")
blip_processor = BlipProcessor.from_pretrained(
    r"G:\models\blip-image-captioning-large",
    local_files_only=True
)
blip_model = BlipForConditionalGeneration.from_pretrained(
    r"G:\models\blip-image-captioning-large",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    local_files_only=True
).to(device)

# ========================
# 工具函数
# ========================

def transcribe_audio(audio_path):
    """使用 Whisper 转录音频（中文优化）"""
    try:
        segments, _ = whisper_model.transcribe(
            audio_path,
            language="zh",          # 强制中文
            beam_size=5,
            vad_filter=True,        # 启用语音活动检测（去静音）
            temperature=0.0         # 确定性输出
        )
        text = "".join([seg.text for seg in segments]).strip()
        return text if text else ""
    except Exception as e:
        print(f"⚠️ ASR 失败 ({audio_path}): {e}")
        return ""

def generate_caption(image_path):
    """使用 BLIP 生成图像描述"""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = blip_processor(images=image, return_tensors="pt").to(device)
        if device == "cuda":
            inputs = {k: v.half() for k, v in inputs.items()}
        
        with torch.no_grad():
            output = blip_model.generate(**inputs, max_length=50, num_beams=5)
        
        caption = blip_processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        print(f"⚠️ 视觉描述失败 ({image_path}): {e}")
        return ""

# ========================
# 主流程
# ========================

def main():
    if not os.path.exists(INPUT_META):
        raise FileNotFoundError(f"未找到第一步输出: {INPUT_META}")

    with open(INPUT_META, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    print(f"共加载 {len(scenes)} 个场景，开始多模态理解...")

    enhanced_scenes = []
    for i, scene in enumerate(scenes):
        print(f"[{i+1}/{len(scenes)}] 处理场景 {scene['scene_id']}...")

        # 1. ASR 转录
        asr_text = ""
        if os.path.exists(scene["audio_path"]):
            asr_text = transcribe_audio(scene["audio_path"])

        # 2. 视觉描述
        vision_caption = ""
        if os.path.exists(scene["frame_path"]):
            vision_caption = generate_caption(scene["frame_path"])

        # 3. 融合上下文（简单拼接，后续可优化）
        combined = []
        if vision_caption:
            combined.append(vision_caption)
        if asr_text:
            combined.append(asr_text)
        combined_context = "。".join(combined) + ("。" if combined else "")

        # 4. 保存
        enhanced_scenes.append({
            "scene_id": scene["scene_id"],
            "start_time": scene["start_time"],
            "end_time": scene["end_time"],
            "duration": scene["duration"],
            "asr_text": asr_text,
            "vision_caption": vision_caption,
            "combined_context": combined_context
        })

    # 保存结果
    output_path = os.path.join(OUTPUT_DIR, "scenes_enhanced.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enhanced_scenes, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 第二步完成！结果保存至: {output_path}")
    print(f"   - 示例 ASR: {enhanced_scenes[0]['asr_text'][:50]}...")
    print(f"   - 示例视觉: {enhanced_scenes[0]['vision_caption']}")

if __name__ == "__main__":
    main()