#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
影视解说项目 - 第三步（API版）：调用 Qwen-Max 生成解说稿
输入：output_step2/scenes_enhanced.json
输出：output_step3/movie_script.txt
"""

import os
import json
import dashscope
from dashscope import Generation

# ========================
# 配置
# ========================
INPUT_META = "output_step2/scenes_enhanced.json"
OUTPUT_DIR = "output_step3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 替换为你的 DashScope API Key
dashscope.api_key = "sk-c7ee0227f323467c85d52047b2766731"  # 👈 在这里填你的 KEY

def build_context(scenes, max_scenes=50):
    texts = []
    for scene in scenes[:max_scenes]:
        if scene.get("combined_context"):
            start = scene["start_time"]
            texts.append(f"[{start:.1f}s] {scene['combined_context']}")
    return "\n".join(texts)

def generate_script_with_qwen(context):
    prompt = f"""你是一位资深影视解说博主，擅长用生动、紧凑、有深度的语言解说电影。请根据以下带时间戳的剧情片段，生成一篇800-1200字的中文解说稿。

要求：
1. 开头要有吸引人的钩子（如悬念、反问、金句）
2. 按时间顺序梳理主线，突出关键转折和人物动机
3. 语言口语化，带情绪张力（可用“你敢信？”、“更绝的是...”等）
4. 结尾升华主题或留下思考
5. 不要出现“视频中”、“画面显示”等元描述

剧情片段：
{context}

现在，请开始你的解说："""

    response = Generation.call(
        model="qwen-max",          # 或 qwen-plus（性价比更高）
        prompt=prompt,
        seed=1234,
        temperature=0.7,
        result_format="text"
    )
    
    if response.status_code == 200:
        return response.output.text.strip()
    else:
        raise RuntimeError(f"API 调用失败: {response}")

def main():
    with open(INPUT_META, "r", encoding="utf-8") as f:
        scenes = json.load(f)
    
    print(f"共加载 {len(scenes)} 个场景，构建剧情上下文...")
    context = build_context(scenes)
    
    print("正在调用 Qwen-Max 生成解说文案...")
    script = generate_script_with_qwen(context)
    
    output_path = os.path.join(OUTPUT_DIR, "movie_script.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(script)
    
    print(f"\n✅ 第三步完成！解说稿已保存至: {output_path}")
    print("\n--- 预览开头 ---")
    print(script[:500] + "...\n")

if __name__ == "__main__":
    main()