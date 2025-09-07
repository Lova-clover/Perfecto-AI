# runner.py
# -*- coding: utf-8 -*-
"""
비-UI(배치) 실행 러너.
- Polly만 사용
- LLM 호출: 절 분절 1회, SSML 1회(내부), 문장 키워드 1회
- 영상 합성까지 수행(옵션: 업로드)
"""

import os
import re
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# 프로젝트 모듈
from persona import generate_response_from_persona
from ssml_converter import breath_linebreaks_batch
from generate_timed_segments import generate_subtitle_from_script
from keyword_generator import generate_image_keywords_per_line_batch
from image_generator import generate_images_for_topic
from video_maker import create_video_with_segments
from upload import upload_to_youtube

load_dotenv()


def _split_to_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _distribute_sentence_keywords_to_segments(
    keywords: List[str],
    n_segments: int
) -> List[str]:
    if n_segments <= 0:
        return []
    if not keywords:
        return ["abstract background"] * n_segments

    s_cnt = max(1, len(keywords))
    base = n_segments // s_cnt
    rem = n_segments % s_cnt
    per_sentence_counts = [base + (1 if i < rem else 0) for i in range(s_cnt)]

    expanded = []
    for i, k in enumerate(keywords):
        expanded.extend([k] * per_sentence_counts[i])

    if len(expanded) < n_segments:
        expanded.extend([keywords[-1]] * (n_segments - len(expanded)))
    elif len(expanded) > n_segments:
        expanded = expanded[:n_segments]
    return expanded


def run_job(job: Dict[str, Any]) -> Optional[str]:
    """
    Args:
        job: {
          "script_text": str (직접 입력 대본)  또는
          "persona_prompt": str (페르소나로 대본 생성),
          "polly_voice_key": "korean_female1",
          "bgm_path": str (선택),
          "upload": bool,
          "youtube_title": str,
          "youtube_description": str
        }
    Returns:
        최종 비디오 경로 또는 None
    """
    print("🎬 [runner] 시작")

    script_text = (job.get("script_text") or "").strip()
    persona_prompt = (job.get("persona_prompt") or "").strip()

    if not script_text and persona_prompt:
        print("🤖 [runner] 페르소나로 대본 생성...")
        try:
            script_text = generate_response_from_persona(persona_prompt).strip()
        except Exception as e:
            print(f"❌ 페르소나 생성 실패: {e}")
            return None

    if not script_text:
        print("❌ [runner] 입력 대본이 없습니다.")
        return None

    # 1차 문장 분절 (LLM 없음)
    sentence_units = _split_to_sentences(script_text)
    if not sentence_units:
        sentence_units = [script_text]
    print(f"🧩 [runner] 1차 분절(문장): {len(sentence_units)}개")

    # 2차 절 분절 (LLM 1회)
    print("🫁 [runner] 2차 분절(절) LLM 호출...")
    try:
        clause_lines = breath_linebreaks_batch(script_text)
    except Exception as e:
        print(f"❌ 절 분절 실패: {e}")
        clause_lines = sentence_units[:]
    print(f"🫁 [runner] 절 라인 수: {len(clause_lines)}")

    # 세그먼트/오디오/ASS (SSML LLM 1회는 내부 처리)
    ass_path = os.path.join("assets", "auto", "subtitles.ass")
    polly_voice_key = job.get("polly_voice_key", "korean_female1")

    print("🗣️ [runner] 세그먼트/ASS 생성...")
    try:
        segments, audio_clips, ass_path = generate_subtitle_from_script(
            script_text=script_text,
            ass_path=ass_path,
            provider="polly",
            template="default",
            polly_voice_key=job.get("polly_voice_key", "korean_female1"),
            strip_trailing_punct_last=True,
            pre_split_lines=clause_lines,   # ✅ 여기서도 전달
        )
    except Exception as e:
        print(f"❌ 세그먼트 생성 실패: {e}")
        return None
    print(f"✅ [runner] 세그먼트 수: {len(segments)}")

    # 문장별 이미지 키워드(LLM 1회) → 절 세그먼트로 균등 분배
    print("🖼️ [runner] 문장별 키워드 생성(LLM 1회)...")
    try:
        sentence_keywords_en = generate_image_keywords_per_line_batch(sentence_units)
    except Exception as e:
        print(f"❌ 키워드 생성 실패: {e}")
        sentence_keywords_en = ["abstract background"] * len(sentence_units)

    mapped_keywords = _distribute_sentence_keywords_to_segments(sentence_keywords_en, len(segments))
    print(f"🔗 [runner] 키워드 매핑 완료 (문장→세그먼트)")

    # 키워드별 이미지 1장씩 확보
    print("📦 [runner] 이미지 검색/다운로드...")
    image_paths: List[Optional[str]] = []
    for kw in mapped_keywords:
        try:
            paths = generate_images_for_topic(kw, max_results=1)
            image_paths.append(paths[0] if paths else None)
        except Exception:
            image_paths.append(None)

    # 합성
    final_audio_path = "assets/auto/_mix_audio.mp3"
    os.makedirs(os.path.join("assets", "auto"), exist_ok=True)
    save_path = os.path.join("assets", "auto", "video.mp4")
    bgm_path = job.get("bgm_path", "")

    print("🧩 [runner] 영상 합성...")
    try:
        video_path = create_video_with_segments(
            image_paths=image_paths if image_paths else [None] * len(segments),
            segments=segments,
            audio_path=final_audio_path,
            topic_title=job.get("topic", ""),
            include_topic_title=True,
            bgm_path=bgm_path,
            save_path=save_path,
            ass_path=ass_path,
        )
    except Exception as e:
        print(f"❌ 영상 합성 실패: {e}")
        return None
    print(f"✅ [runner] 영상 생성: {video_path}")

    # 업로드(옵션)
    if job.get("upload", False) and video_path and os.path.exists(video_path):
        try:
            url = upload_to_youtube(
                video_path,
                title=(job.get("youtube_title") or "AI 자동 생성 숏폼"),
                description=(job.get("youtube_description") or "Perfecto AI로 생성한 숏폼입니다.")
            )
            print(f"☁️ [runner] 업로드 완료: {url}")
            return video_path
        except Exception as e:
            print(f"⚠️ 업로드 실패: {e}")

    return video_path


if __name__ == "__main__":
    # 간단 테스트 예제
    job = {
        "script_text": "만약 지구의 산소 농도가 단 5%만 줄어든다면? 우리의 호흡은 즉시 힘들어지고 도시 전체의 전력망이 순식간에 불안정해집니다. 엘리베이터, 공장, 지하철이 동시에 멈춘다면? 생각보다 위험합니다.",
        "polly_voice_key": "korean_female1",
        "bgm_path": "",
        "upload": False,
        "youtube_title": "AI 자동 생성 숏폼",
        "youtube_description": "Perfecto AI로 생성한 숏폼입니다."
    }
    out = run_job(job)
    print("DONE:", out)
