import os
import streamlit as st
from dotenv import load_dotenv
from file_handler import get_documents_from_files
from text_scraper import scrape_web_content
from best_subject_subtitle_extractor import extract_best_subject, segment_script
from persona import generate_response_from_persona
from generate_timed_segments import (
    generate_subtitle_from_script,
    generate_ass_subtitle,
    SUBTITLE_TEMPLATES,
)
from image_generator import generate_images_for_topic, generate_videos_for_topic
from keyword_generator import generate_image_keywords_per_line_batch  # 🔄 배치 키워드 함수 사용
from upload import upload_to_youtube
from video_maker import create_video_with_segments, add_subtitles_to_video


def run_job(job):
    """
    영상 제작 전체 파이프라인 실행
    - LLM 호출 최소화: (1) 분절, (2) SSML, (3) 키워드 → 총 3회
    """
    try:
        st.write("🎬 영상 제작 시작...")

        # 1. 사용자 입력 텍스트 확보
        script_text = job.get("script_text", "").strip()
        if not script_text:
            st.error("❌ 입력 대본이 없습니다.")
            return None

        # 2. 최종 대본 생성 (Persona 체인)
        final_script = generate_response_from_persona(script_text)

        # 3. 세그먼트 + SSML + 오디오 생성
        ass_path = os.path.join("assets", "auto", "subtitles.ass")
        segments, audio_clips, ass_path = generate_subtitle_from_script(
            final_script,
            ass_path,
            provider=job.get("tts_provider", "polly"),
            template=job.get("voice_template", "default"),
            polly_voice_key=job.get("polly_voice_key", "korean_female1"),
            strip_trailing_punct_last=True,
        )

        # 4. 이미지/영상 검색 (LLM은 전체 라인 배열로 1회 호출)
        image_paths = []
        if job.get("style") != "emotional":
            try:
                line_texts = [seg["text"] for seg in segments] if segments else []
                # 🔄 전체 라인 한번에 LLM 호출
                image_keywords = generate_image_keywords_per_line_batch(line_texts)

                # 각 라인별로 Pexels 검색 (LLM은 더 안 씀)
                for kw in image_keywords:
                    paths = generate_images_for_topic(kw, max_results=1)
                    if paths:
                        image_paths.append(paths[0])
                    else:
                        image_paths.append(None)
            except Exception as e:
                st.error(f"❌ 이미지 키워드 생성/검색 실패: {e}")
                image_paths = [None] * len(segments)

        # 5. 영상 합성 (자막은 마지막에 덧씌움)
        video_path = os.path.join("assets", "auto", "video.mp4")
        final_audio_path = "assets/auto/_mix_audio.mp3"  # generate_subtitle_from_script에서 생성
        video_path = create_video_with_segments(
            image_paths=image_paths,
            segments=segments,
            audio_path=final_audio_path,
            topic_title=job.get("topic", ""),
            include_topic_title=True,
            bgm_path=job.get("bgm_path", ""),
            save_path=video_path,
            ass_path=ass_path,
        )

        # 6. 유튜브 업로드 (옵션)
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
