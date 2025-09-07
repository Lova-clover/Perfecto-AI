# polly_tts.py
from __future__ import annotations
import os
import subprocess
from typing import List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

try:
    import streamlit as st
    _SECRETS = getattr(st, "secrets", {})
except Exception:
    _SECRETS = {}

# 프로젝트에서 쓰는 음성 키 → Polly VoiceId 매핑
TTS_POLLY_VOICES = {
    "korean_female1": "Seoyeon",
    "korean_male1":   "Joon",
    # 필요시 추가
}

def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    if _SECRETS and name in _SECRETS:
        return _SECRETS.get(name)
    return os.getenv(name, default)

def _get_polly_client():
    region = _get_env("AWS_REGION_NAME") or _get_env("AWS_REGION") or "ap-northeast-2"
    key    = _get_env("AWS_ACCESS_KEY_ID")
    sec    = _get_env("AWS_SECRET_ACCESS_KEY")

    cfg = Config(retries={"max_attempts": 5, "mode": "standard"})
    if key and sec:
        return boto3.client(
            "polly",
            region_name=region,
            aws_access_key_id=key,
            aws_secret_access_key=sec,
            config=cfg,
        )
    else:
        return boto3.client("polly", region_name=region, config=cfg)

def synthesize_lines_polly(
    ssml_lines: List[str],
    voice_key: str = "korean_female1",
    out_dir: str = "assets/_tts_chunks",
    format_ext: str = "mp3",
) -> List[str]:
    """
    각 SSML 라인을 Polly로 합성하여 파일로 저장하고 경로 리스트 반환.
    """
    if not ssml_lines:
        return []

    os.makedirs(out_dir, exist_ok=True)
    voice_id = TTS_POLLY_VOICES.get(voice_key) or voice_key or "Seoyeon"
    polly = _get_polly_client()

    paths: List[str] = []
    for i, ssml in enumerate(ssml_lines):
        out_path = os.path.join(out_dir, f"tts_{i:03d}.{format_ext}")
        try:
            resp = polly.synthesize_speech(
                VoiceId=voice_id,
                OutputFormat="mp3" if format_ext.lower() == "mp3" else "ogg_vorbis",
                Text=ssml,
                TextType="ssml",
            )
            audio_stream = resp.get("AudioStream")
            if not audio_stream:
                raise RuntimeError("No AudioStream in Polly response")
            with open(out_path, "wb") as f:
                f.write(audio_stream.read())
        except (BotoCoreError, ClientError, Exception):
            _write_silence_mp3(out_path, 0.6)  # 실패 시 0.6s 무음
        paths.append(out_path)
    return paths

def _write_silence_mp3(out_path: str, duration_sec: float = 0.6):
    """
    ffmpeg로 무음 mp3 생성 (moviepy 없이).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono",
        "-t", str(max(0.05, float(duration_sec))),
        "-c:a", "libmp3lame", "-b:a", "128k",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        # ffmpeg 실패 시라도 빈 파일 생성 (후속 단계에서 길이 폴백)
        open(out_path, "wb").close()

def mix_polly(chunk_paths: List[str], out_path: str) -> str:
    """
    ffmpeg concat demuxer로 MP3들을 무손실로 이어붙임.
    (모두 동일 코덱/샘플레이트인 Polly 출력이라 -c copy 가능)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # concat용 임시 리스트 파일 작성
    list_path = out_path + ".txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in chunk_paths:
            if not p:
                continue
            # 경로에 ' 가 있어도 안전하게 처리
            f.write(f"file '{os.path.abspath(p).replace(\"'\", \"'\\\\''\")}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        # copy 실패 시 재인코딩으로 폴백
        cmd2 = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c:a", "libmp3lame", "-b:a", "192k",
            out_path,
        ]
        subprocess.run(cmd2, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        try:
            os.remove(list_path)
        except Exception:
            pass
    return out_path
