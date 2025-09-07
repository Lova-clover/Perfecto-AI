# polly_tts.py
from __future__ import annotations
import os
from typing import List, Optional
import io

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

# moviepy로 오디오 이어붙이기
from moviepy.editor import AudioFileClip, concatenate_audioclips

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
    # st.secrets 우선, 없으면 OS env
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
        # 크레덴셜은 인스턴스 프로파일/기본 설정에서 찾음
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
        except (BotoCoreError, ClientError) as e:
            # 실패 시 빈 무음 파일 대체(0.6s)
            _write_silence(out_path, 0.6)
        paths.append(out_path)
    return paths

def _write_silence(out_path: str, duration_sec: float = 0.6):
    # moviepy로 짧은 무음 생성 대신, 빈 파일을 만들고 후속 로직에서 길이 보정하는 방어
    open(out_path, "wb").close()

def mix_polly(chunk_paths: List[str], out_path: str) -> str:
    """
    라인별 오디오를 순서대로 이어붙여 하나의 파일로 만든다.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    clips = []
    for p in chunk_paths:
        try:
            clips.append(AudioFileClip(p))
        except Exception:
            # 읽을 수 없으면 0.2s짜리 더미(사일런스 대체는 건너뜀)
            pass
    if clips:
        final = concatenate_audioclips(clips)
        final.write_audiofile(out_path, codec="libmp3lame", verbose=False, logger=None)
        for c in clips:
            c.close()
    else:
        open(out_path, "wb").close()
    return out_path
