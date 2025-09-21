from __future__ import annotations
import os
from typing import Optional, Dict, Any
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
import video_analyzer_core as core

app = FastMCP("video-analyzer")

def _get_api_key(explicit: Optional[str]) -> str:
    key = explicit or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key 未提供，請傳入 openai_api_key 或設定環境變數 OPENAI_API_KEY")
    return key

@app.tool()
def analyze_video(video_path: str, openai_api_key: Optional[str] = None, include_logs: bool = True) -> Any:
    """
    完整流程：自動判斷是否有音訊、擷取影格（記憶體）、音訊（記憶體）→ 轉錄 → 融合 → 回傳描述。
    include_logs=True 時，回傳 {result, logs}；False 時僅回傳 result。
    """
    api_key = _get_api_key(openai_api_key)
    core.start_log_capture()
    try:
        result = core.analyze_video(api_key, video_path)
        logs = core.get_captured_logs()
    finally:
        core.end_log_capture()
    return {"result": result, "logs": logs} if include_logs else result

@app.tool()
def image_analysis(video_path: str, openai_api_key: Optional[str] = None, include_logs: bool = True, model: str = "gpt-4o") -> Any:
    """
    只進行圖像分析。include_logs=True 時，回傳 {result, logs}。
    """
    api_key = _get_api_key(openai_api_key)
    client = OpenAI(api_key=api_key)
    core.start_log_capture()
    try:
        result = core.image_analysis(video_path, client, model=model)
        logs = core.get_captured_logs()
    finally:
        core.end_log_capture()
    return {"result": result, "logs": logs} if include_logs else result

@app.tool()
def audio_transcribe(video_path: str, openai_api_key: Optional[str] = None, include_logs: bool = True) -> Any:
    """
    只進行音訊轉錄（記憶體）。include_logs=True 時，回傳 {result, logs}。
    """
    api_key = _get_api_key(openai_api_key)
    client = OpenAI(api_key=api_key)
    core.start_log_capture()
    try:
        audio_present, wav_bytes = core.detect_and_extract_audio_in_memory(video_path)
        if not audio_present:
            result = "未偵測到音訊或無法確認音訊"
        elif wav_bytes is None:
            result = "偵測到音訊但記憶體萃取失敗"
        else:
            result = core.audio_analysis_from_memory(wav_bytes, client)
        logs = core.get_captured_logs()
    finally:
        core.end_log_capture()
    return {"result": result, "logs": logs} if include_logs else result

@app.tool()
def combine_image_and_audio(image_desc: str, audio_text: str, openai_api_key: Optional[str] = None, include_logs: bool = True, model: str = "gpt-4o") -> Any:
    """
    輸入兩段文字，融合成完整描述。include_logs=True 時，回傳 {result, logs}。
    """
    api_key = _get_api_key(openai_api_key)
    client = OpenAI(api_key=api_key)
    core.start_log_capture()
    try:
        result = core.combine_image_and_audio(image_desc, audio_text, client, model=model)
        logs = core.get_captured_logs()
    finally:
        core.end_log_capture()
    return {"result": result, "logs": logs} if include_logs else result

if __name__ == "__main__":
    app.run()