# main.py
# -*- coding: utf-8 -*-
"""
Streamlit UI 엔드투엔드 파이프라인 (Polly 전용)
- 1차 분절: 문장 단위 (LLM 미사용, 구두점 기반)
- 2차 분절: 절/호흡 단위 (LLM 1회, ssml_converter.breath_linebreaks_batch)
- SSML: 절 배열을 그대로 LLM 1회 호출해 라인별 SSML 생성
- ASS: 절 배열 → ASS (물음표 외 특수문자 제거, 피치 임계 이상 색상)
- 이미지 키워드: '문장' 기준 LLM 1회 (영어 키워드), 절 세그먼트에 균등 매핑
- TTS: Polly만 사용
"""

import os
import re
import json
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

# ===== 프로젝트 모듈 =====
from persona import generate_response_from_persona  # (있을 경우 사용)
from ssml_converter import breath_linebreaks_batch  # 절 분절 (LLM 1회)
from generate_timed_segments import (
    generate_subtitle_from_script,  # 절→SSML→TTS→세그먼트→ASS
    generate_ass_subtitle,          # 필요 시 단독 호출 가능(여기선 내부에서 처리됨)
    SUBTITLE_TEMPLATES,
    _auto_split_for_tempo,          # 호환용(현재 미사용)
    dedupe_adjacent_texts,          # 호환용(현재 미사용)
)
from keyword_generator import generate_image_keywords_per_line_batch  # 문장→영어 키워드(LLM 1회)
from polly_tts import TTS_POLLY_VOICES
from image_generator import generate_images_for_topic
from video_maker import create_video_with_segments
from upload import upload_to_youtube  # 업로드 옵션

# RAG/수집 모듈(프로젝트에 존재) - 여기선 직접 사용하지 않지만 호환을 위해 남겨둠
try:
    from RAG.rag_pipeline import get_retriever_from_source  # noqa: F401
    from RAG.chain_builder import get_conversational_rag_chain, get_default_chain  # noqa: F401
    from best_subtitle_extractor import load_best_subtitles_documents  # noqa: F401
    from text_scraper import get_links, clean_html_parallel, filter_noise  # noqa: F401
except Exception:
    pass

load_dotenv()


# =========================
# 유틸
# =========================
def _split_to_sentences(text: str) -> List[str]:
    """구두점 기반 1차 분절(LLM 미사용): 문장 단위로만 분할."""
    text = (text or "").strip()
    if not text:
        return []
    # 문장 종결부호(. ! ?) 기준 + 뒤 공백으로 분절
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = [p.strip() for p in parts if p.strip()]
    return out


def _distribute_sentence_keywords_to_segments(
    keywords: List[str],
    n_segments: int
) -> List[str]:
    """
    문장 키워드 배열(길이 S)을 절 세그먼트 개수(N)에 균등 분배하여 길이 N의 키워드 배열 생성
    예) N=10, S=3 -> 분배 [4,3,3]
    """
    if n_segments <= 0:
        return []
    if not keywords:
        return ["abstract background"] * n_segments

    s_cnt = max(1, len(keywords))
    base = n_segments // s_cnt
    rem = n_segments % s_cnt
    per_sentence_counts = [base + (1 if i < rem else 0) for i in range(s_cnt)]
    # 확장
    expanded = []
    for i, k in enumerate(keywords):
        expanded.extend([k] * per_sentence_counts[i])
    # 혹시 모자라거나 넘치면 보정
    if len(expanded) < n_segments:
        expanded.extend([keywords[-1]] * (n_segments - len(expanded)))
    elif len(expanded) > n_segments:
        expanded = expanded[:n_segments]
    return expanded


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Perfecto AI — Polly-only Pipeline", layout="wide")
st.title("🎬 Perfecto AI — Polly 전용 파이프라인")

with st.sidebar:
    st.subheader("TTS (Polly)")
    polly_voice_key = st.selectbox(
        "Polly Voice",
        options=list(TTS_POLLY_VOICES.keys()) or ["korean_female1"],
        index=0,
        help="프로젝트에서 정의한 Polly 음성 프리셋 키",
    )
    add_bgm = st.checkbox("배경음악 추가", value=False)
    uploaded_bgm_file = None
    if add_bgm:
        uploaded_bgm_file = st.file_uploader("BGM 업로드 (.mp3, .wav)", type=["mp3", "wav"])

    allow_upload = st.checkbox("완료 시 유튜브 업로드", value=False)
    yt_title = st.text_input("YouTube 제목", value="AI 자동 생성 숏폼")
    yt_desc = st.text_area("YouTube 설명", value="Perfecto AI로 생성한 숏폼입니다.", height=80)


st.markdown("### 1) 대본 입력 또는 생성")
tab1, tab2 = st.tabs(["직접 입력", "페르소나로 생성"])

with tab1:
    user_script = st.text_area(
        "최종 대본(한국어 권장)",
        value="만약 지구의 산소 농도가 단 5%만 줄어든다면? 우리의 호흡은 즉시 힘들어지고 도시 전체의 전력망이 순식간에 불안정해집니다. 엘리베이터, 공장, 지하철이 동시에 멈춘다면? 생각보다 위험합니다.",
        height=180
    )

with tab2:
    persona_prompt = st.text_area(
        "페르소나 프롬프트",
        value="너는 유튜브 쇼츠용 과학 콘텐츠 작가다. 150~200단어 대본을 한국어로 만들어라. 훅은 강렬한 질문으로 시작.",
        height=160
    )
    if st.button("🤖 페르소나로 대본 생성", use_container_width=True):
        try:
            gen = generate_response_from_persona(persona_prompt).strip()
        except Exception as e:
            st.error(f"대본 생성 실패: {e}")
            gen = ""
        if gen:
            user_script = gen
            st.success("✅ 대본 생성 완료 (좌측 탭의 텍스트 박스에 반영되지 않을 수 있으니 아래 미리보기 확인)")
            st.code(gen, language="markdown")

st.divider()

st.markdown("### 2) 영상 만들기")
colA, colB = st.columns([1, 1])
with colA:
    make_btn = st.button("🎥 영상 만들기", use_container_width=True, type="primary")
with colB:
    st.info("파이프라인: 문장분절 → 절분절(LLM) → SSML(LLM) → Polly TTS → 세그먼트/ASS → 이미지 매칭 → 렌더")


# =========================
# 파이프라인 실행
# =========================
if make_btn:
    final_script = (user_script or "").strip()
    if not final_script:
        st.error("❌ 대본이 비어 있습니다.")
        st.stop()

    # BGM 파일 저장 (선택)
    bgm_path = ""
    if uploaded_bgm_file:
        os.makedirs("assets", exist_ok=True)
        bgm_path = os.path.join("assets", uploaded_bgm_file.name)
        with open(bgm_path, "wb") as f:
            f.write(uploaded_bgm_file.read())
        st.success(f"🔊 배경음악 업로드: {os.path.basename(bgm_path)}")

    # 1) 1차 분절(문장) — LLM 미사용
    sentence_lines = _split_to_sentences(final_script)
    if not sentence_lines:
        sentence_lines = [final_script]
    st.write(f"🧩 1차 분절(문장) 개수: {len(sentence_lines)}")

    # 2) 2차 분절(절) — LLM 1회
    st.write("🫁 절(호흡) 단위로 분절 중... (LLM 1회)")
    try:
        clause_lines = breath_linebreaks_batch(final_script)  # LLM 1회
    except Exception as e:
        st.error(f"절 분절 실패(LLM): {e}")
        clause_lines = sentence_lines[:]  # 폴백
    st.write(f"🫁 2차 분절(절) 개수: {len(clause_lines)}")
    with st.expander("절 분절 미리보기", expanded=False):
        st.write(clause_lines)

    # 3) 절→SSML→TTS→세그먼트→ASS (LLM: SSML 1회는 내부 처리)
    ass_path = os.path.join("assets", "auto", "subtitles.ass")
    st.write("🗣️ TTS/세그먼트/ASS 생성 중...")
    try:
        segments, audio_clips, ass_path = generate_subtitle_from_script(
            final_script,
            ass_path,
            provider="polly",
            template="default",              # 호환용 인자
            polly_voice_key=polly_voice_key,
            strip_trailing_punct_last=True,
            pre_split_lines=clause_lines,    # 전달하면 내부에서 절 LLM 재호출 없음
        )
    except Exception as e:
        st.error(f"세그먼트 생성 실패: {e}")
        st.stop()

    st.success(f"✅ 세그먼트 생성 완료 (총 {len(segments)}개)")
    with st.expander("세그먼트 미리보기", expanded=False):
        st.json(segments[:10])  # 처음 10개만 예시 표시

    # 4) 문장별 영어 키워드(LLM 1회) → 절 세그먼트에 균등 매핑
    st.write("🖼️ 문장별 이미지 키워드 생성 중... (LLM 1회)")
    try:
        sentence_keywords_en = generate_image_keywords_per_line_batch(sentence_lines)  # len == 문장 수
    except Exception as e:
        st.error(f"키워드 생성 실패: {e}")
        sentence_keywords_en = ["abstract background"] * len(sentence_lines)

    mapped_keywords = _distribute_sentence_keywords_to_segments(sentence_keywords_en, len(segments))
    with st.expander("문장 키워드 → 세그먼트 매핑(상위 15)", expanded=False):
        st.table(
            [{"seg_idx": i, "keyword_en": mapped_keywords[i], "text": segments[i]["text"]} for i in range(min(15, len(segments)))]
        )

    # 5) 키워드별 이미지 1장씩 확보
    st.write("📦 이미지 검색/다운로드...")
    image_paths: List[Optional[str]] = []
    for kw in mapped_keywords:
        try:
            paths = generate_images_for_topic(kw, max_results=1)  # 프로젝트 내 이미지 검색 함수
            image_paths.append(paths[0] if paths else None)
        except Exception:
            image_paths.append(None)

    # 6) 영상 합성
    st.write("🧩 영상 합성 중...")
    os.makedirs(os.path.join("assets", "auto"), exist_ok=True)
    final_audio_path = "assets/auto/_mix_audio.mp3"
    out_video = os.path.join("assets", "auto", "video.mp4")

    try:
        video_path = create_video_with_segments(
            image_paths=image_paths if image_paths else [None] * len(segments),
            segments=segments,
            audio_path=final_audio_path,
            topic_title="",
            include_topic_title=True,
            bgm_path=bgm_path,
            save_path=out_video,
            ass_path=ass_path,
        )
        st.success("✅ 영상 생성 완료")
        st.video(video_path)
        st.session_state.final_video_path = video_path
    except Exception as e:
        st.error(f"영상 합성 실패: {e}")
        st.stop()

    # 7) 업로드(옵션)
    if allow_upload:
        try:
            url = upload_to_youtube(
                video_path,
                title=yt_title.strip() or "AI 자동 생성 숏폼",
                description=yt_desc.strip() or "Perfecto AI로 생성한 숏폼입니다."
            )
            st.success(f"☁️ 업로드 완료: {url}")
            st.session_state.youtube_link = url
        except Exception as e:
            st.error(f"업로드 실패: {e}")
