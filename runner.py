# runner.py
import os
import re
import streamlit as st
from dotenv import load_dotenv

from persona import generate_response_from_persona
from generate_timed_segments import generate_subtitle_from_script, generate_ass_subtitle
from image_generator import generate_images_for_topic
from keyword_generator import generate_image_keywords_per_line_batch
from upload import upload_to_youtube
from video_maker import create_video_with_segments

load_dotenv()

def _split_to_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    parts = re.split(r"(?<=[.!?])\s+", text) if text else []
    return [p.strip() for p in parts if p.strip()]

def run_job(job):
    """
    전체 파이프라인 (LLM 총 3회): ① 절분절 ② SSML ③ 문장별 키워드
    """
    try:
        st.write("🎬 영상 제작 시작...")

        # 1) 입력 확보
        script_text = (job.get("script_text", "") or "").strip()
        if not script_text:
            st.error("❌ 입력 대본이 없습니다.")
            return None

        # 2) 페르소나 응답 → 최종 대본
        final_script = generate_response_from_persona(script_text)

        # 2-1) 문장 리스트(LLM 미사용) → 키워드용
        sentence_units = _split_to_sentences(final_script)
        if not sentence_units:
            sentence_units = [final_script]

        # 3) 절/SSML/오디오/세그먼트
        ass_path = os.path.join("assets", "auto", "subtitles.ass")
        segments, audio_clips, ass_path = generate_subtitle_from_script(
            final_script,
            ass_path,
            provider=job.get("tts_provider", "polly"),
            template=job.get("voice_template", "default"),
            polly_voice_key=job.get("polly_voice_key", "korean_female1"),
            strip_trailing_punct_last=True,
            pre_split_lines=None,  # 사전 절분절이 있으면 여기에 전달
        )

        # 4) 문장별 키워드(영어) - LLM 1회
        image_paths = []
        if job.get("style") != "emotional":
            try:
                keywords = generate_image_keywords_per_line_batch(sentence_units)  # 길이 = 문장 수
            except Exception as e:
                st.error(f"❌ 키워드 생성 실패: {e}")
                keywords = ["abstract background"] * len(sentence_units)

            # 4-1) 문장 키워드를 절 세그먼트에 균등 분배
            n_seg = len(segments)
            s_cnt = len(keywords) if keywords else 1
            if s_cnt <= 0:
                s_cnt = 1
                keywords = ["abstract background"]

            base = n_seg // s_cnt
            rem = n_seg % s_cnt
            per_sentence_counts = [base + (1 if i < rem else 0) for i in range(s_cnt)]
            # 예: n_seg=10, s_cnt=3 → [4,3,3]

            expanded_keywords = []
            for i, k in enumerate(keywords):
                expanded_keywords.extend([k] * per_sentence_counts[i])

            # 4-2) 키워드별로 이미지 검색
            for kw in expanded_keywords:
                paths = generate_images_for_topic(kw, max_results=1)
                image_paths.append(paths[0] if paths else None)

        # 5) 영상 합성
        video_path = os.path.join("assets", "auto", "video.mp4")
        final_audio_path = "assets/auto/_mix_audio.mp3"
        video_path = create_video_with_segments(
            image_paths=image_paths if image_paths else [None] * len(segments),
            segments=segments,
            audio_path=final_audio_path,
            topic_title=job.get("topic", ""),
            include_topic_title=True,
            bgm_path=job.get("bgm_path", ""),
            save_path=video_path,
            ass_path=ass_path,
        )

        # 6) 업로드(옵션)
        if job.get("upload", False) and video_path and os.path.exists(video_path):
            youtube_url = upload_to_youtube(
                video_path,
                title=job.get("youtube_title", "AI 자동 생성 영상"),
                description=job.get("youtube_description", "AI로 생성된 숏폼입니다.")
            )
            st.success(f"✅ 업로드 완료: {youtube_url}")
            st.session_state.youtube_link = youtube_url
        else:
            st.success("✅ 영상 생성 완료")
            st.video(video_path)
            st.session_state.final_video_path = video_path

        return video_path

    except Exception as e:
        st.error(f"❌ 영상 생성 중 오류: {e}")
        return None
