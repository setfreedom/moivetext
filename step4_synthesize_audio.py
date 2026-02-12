#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
影视解说项目 - 第四步：使用 ModelScope 内置的 CosyVoice 合成语音（正确方式）
"""

import os
import re
import torch
import soundfile as sf

# 强制使用 CPU（无 GPU）
torch.set_num_threads(4)

def split_sentences(text):
    sentences = re.split(r'(?<=[。？！…])\s*', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def main():
    INPUT_SCRIPT = "output_step3/movie_script.txt"
    OUTPUT_DIR = "output_step4/audio"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_SCRIPT, "r", encoding="utf-8") as f:
        script = f.read()
    
    sentences = split_sentences(script)
    print(f"共切分为 {len(sentences)} 句")

    # ✅ 关键：直接导入 CosyVoice 模型类
    from modelscope.models.audio.tts.cosyvoice import CosyVoiceModel
    from modelscope.pipelines.audio.tts_pipeline import TextToSpeechPipeline

    print("正在加载 CosyVoice 模型...")
    model = CosyVoiceModel.from_pretrained('iic/CosyVoice-300M')
    pipeline = TextToSpeechPipeline(model=model, device='cpu')

    for i, sentence in enumerate(sentences, 1):
        print(f"[{i}/{len(sentences)}] 合成: {sentence[:40]}...")
        try:
            # 调用 pipeline
            result = pipeline(input=sentence, voice='中文女')
            audio = result['output_wav']
            sf.write(os.path.join(OUTPUT_DIR, f"audio_{i:03d}.wav"), audio, 22050)
            print(f"✅ 保存成功")
        except Exception as e:
            print(f"❌ 失败: {e}")

    print("🎉 语音合成完成！")

if __name__ == "__main__":
    main()