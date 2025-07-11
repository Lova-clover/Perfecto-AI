import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from rag_pipeline import get_retriever_from_source, get_document_chain, get_default_chain
from web_ingest import full_web_ingest
from image_generator import generate_images_for_topic
from elevenlabs_tts import generate_tts, TTS_TEMPLATES
from whisper_asr import transcribe_audio_with_timestamps, generate_ass_subtitle, SUBTITLE_TEMPLATES
from video_maker import create_video_with_segments, add_subtitles_to_video
from deep_translator import GoogleTranslator
import os
import requests # 기본 이미지 다운로드를 위해 추가

# API 키 로드
load_dotenv()

# --- 앱 기본 설정 ---
st.set_page_config(page_title="Multimodal RAG Chatbot", page_icon="🤖")
st.title("🤖 멀티모달 파일/URL 분석 RAG 챗봇")
st.markdown(
    """
안녕하세요! 이 챗봇은 웹사이트 URL이나 업로드된 파일(PDF, DOCX, TXT)의 내용을 분석하고 답변합니다.
"""
)

# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "당신은 문서 분석 전문가 AI 어시스턴트이자, 숏폼(Short-form) 영상 제작을 위한 스크립트 전문가입니다. 주어진 문서의 텍스트와 테이블을 정확히 이해하고, 짧고 간결하며 핵심 내용을 담은 쇼츠 영상 스크립트를 제작해주세요. 스크립트 외에는 어떤 답변도 해서는 안됩니다. 또한 마크다운과 같은 기호는 전부 제거해주세요."
if "last_user_query" not in st.session_state:
    st.session_state.last_user_query = ""
if "video_topic" not in st.session_state:
    st.session_state.video_topic = "" # 영상 주제 초기화
if "edited_script_content" not in st.session_state:
    st.session_state.edited_script_content = "" # 수정 가능한 스크립트 내용 초기화
if "selected_tts_template" not in st.session_state:
    st.session_state.selected_tts_template = "educational"
if "selected_subtitle_template" not in st.session_state:
    st.session_state.selected_subtitle_template = "educational"
if "bgm_path" not in st.session_state:
    st.session_state.bgm_path = None


# --- 사이드바 UI ---
with st.sidebar:
    st.header("⚙️ 설정")
    st.divider()
    st.subheader("🤖 AI 페르소나 설정")
    prompt_input = st.text_area(
        "AI의 역할을 설정해주세요.", value=st.session_state.system_prompt, height=150
    )
    if st.button("페르소나 적용"):
        st.session_state.system_prompt = prompt_input
        st.toast("AI 페르소나가 적용되었습니다.")
    st.divider()
    st.subheader("🔎 분석 대상 설정")
    url_input = st.text_input("검색 키워드 입력", placeholder="ex) 인공지능 윤리")
    uploaded_files = st.file_uploader(
        "파일 업로드 (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )
    st.info("LlamaParse는 테이블, 텍스트가 포함된 문서 분석에 최적화되어 있습니다.", icon="ℹ️")
    
    if st.button("분석 시작"):
        st.session_state.messages = []
        st.session_state.retriever = None
        st.session_state.video_topic = "" # 분석 시작 시 영상 주제 초기화
        st.session_state.edited_script_content = "" # 분석 시작 시 스크립트 내용 초기화

        source_type = None
        source_input = None

        if uploaded_files:
            source_type = "Files"
            source_input = uploaded_files

        elif url_input:
            with st.spinner("웹페이지를 수집하고 벡터화하는 중입니다..."):
                text_path, index_dir, error = full_web_ingest(url_input)
                if not error:
                    source_type = "FAISS"
                    source_input = index_dir  # 폴더 경로
                else:
                    st.error(f"웹페이지 수집 및 벡터화 중 오류 발생: {error}")
        else:
            st.warning("검색 키워드 또는 파일을 입력해주세요.")

        if source_input and source_type:
            st.session_state.retriever = get_retriever_from_source(source_type, source_input)
            if st.session_state.retriever:
                st.success("분석이 완료되었습니다! 이제 질문해보세요.")
            else:
                st.error("문서 처리 중 오류가 발생했습니다. 다시 시도해주세요.")


    st.divider()
    # --- 영상 제작 설정 섹션 ---
    st.subheader("💡 영상 제작 설정")

    # 영상 주제 입력 필드
    st.session_state.video_topic = st.text_input(
        "영상 주제를 입력하거나 수정하세요:",
        value=st.session_state.video_topic # 세션 상태에서 가져옴
    )

    # 스크립트 내용 (수정 가능) 텍스트 영역
    st.session_state.edited_script_content = st.text_area(
        "스크립트 내용 (여기서 수정하세요):",
        value=st.session_state.edited_script_content, # 세션 상태에서 가져옴
        height=300
    )

    # TTS 템플릿 선택
    selected_tts_template = st.selectbox(
        "음성 템플릿 선택",
        options=list(TTS_TEMPLATES.keys()),
        index=list(TTS_TEMPLATES.keys()).index(st.session_state.selected_tts_template)
    )
    st.session_state.selected_tts_template = selected_tts_template

    # 자막 템플릿 선택
    selected_subtitle_template = st.selectbox(
        "자막 템플릿 선택",
        options=list(SUBTITLE_TEMPLATES.keys()),
        index=list(SUBTITLE_TEMPLATES.keys()).index(st.session_state.selected_subtitle_template)
    )
    st.session_state.selected_subtitle_template = selected_subtitle_template

    # BGM 파일 업로드 (선택 사항)
    uploaded_bgm_file = st.file_uploader("BGM 파일 업로드 (선택 사항, .mp3)", type=["mp3"])
    if uploaded_bgm_file:
        bgm_save_path = "assets/bgm.mp3" # 임시 저장 경로
        os.makedirs(os.path.dirname(bgm_save_path), exist_ok=True)
        with open(bgm_save_path, "wb") as f:
            f.write(uploaded_bgm_file.read())
        st.session_state.bgm_path = bgm_save_path
        st.success("BGM 파일 업로드 완료!")
    else:
        st.session_state.bgm_path = None


    if st.button("영상 만들기"):
        # 사용자가 수정한 스크립트 내용과 주제를 사용
        final_script_for_video = st.session_state.edited_script_content
        final_topic_for_video = st.session_state.video_topic

        if not final_script_for_video.strip():
            st.error("스크립트 내용이 비어있습니다. 스크립트를 입력하거나 생성해주세요.")
            st.stop()
        if not final_topic_for_video.strip():
            st.error("영상 주제가 비어있습니다. 주제를 입력해주세요.")
            st.stop()

        with st.spinner("✨ 영상 제작 중입니다..."):
            try:
                # --- 0-1. 추출된 토픽을 영어로 번역 (GoogleTranslator 사용) ---
                # 이제 LLM으로 주제를 추출하는 대신, 사용자가 입력한 주제를 바로 사용합니다.
                st.write("🌐 이미지 검색어를 영어로 번역 중...")
                image_query_english = ""
                try:
                    translator = GoogleTranslator(source='ko', target='en')
                    image_query_english = translator.translate(final_topic_for_video)
                    st.success(f"이미지 검색어 번역 완료 (영어): '{image_query_english}'")
                except Exception as e:
                    st.warning(f"이미지 검색어 번역에 실패했습니다. 한국어 검색어를 그대로 사용합니다. 오류: {e}")
                    image_query_english = final_topic_for_video
                image_query_final = image_query_english 

                # --- 1. Text-to-Speech (TTS) 생성 ---
                audio_output_dir = "assets"
                os.makedirs(audio_output_dir, exist_ok=True)
                audio_path = os.path.join(audio_output_dir, "generated_audio.mp3")
                
                st.write("🗣️ 음성 파일 생성 중...")
                generate_tts(
                    text=final_script_for_video, # <--- 사용자가 수정/입력한 스크립트 사용
                    save_path=audio_path,
                    template_name=st.session_state.selected_tts_template # 선택된 템플릿 사용
                )
                st.success(f"음성 파일 생성 완료: {audio_path}")

                # --- 2. Audio Transcription (ASR) 및 Subtitle (ASS) 파일 생성 ---
                subtitle_output_dir = "assets"
                os.makedirs(subtitle_output_dir, exist_ok=True)
                ass_path = os.path.join(subtitle_output_dir, "generated_subtitle.ass")

                st.write("📝 자막 생성을 위한 음성 분석 중...")
                segments = transcribe_audio_with_timestamps(audio_path)
                generate_ass_subtitle(
                    segments=segments,
                    ass_path=ass_path,
                    template_name=st.session_state.selected_subtitle_template # 선택된 템플릿 사용
                )
                st.success(f"자막 파일 생성 완료: {ass_path}")

                # --- 3. 이미지 생성 ---
                num_images = max(1, len(segments))
                image_output_dir = "assets"
                os.makedirs(image_output_dir, exist_ok=True)
                
                st.write(f"🖼️ '{image_query_final}' 관련 이미지 {num_images}장 생성 중...")
                image_paths = generate_images_for_topic(image_query_final, num_images)
                
                if not image_paths:
                    st.warning("이미지 생성에 실패했습니다. 기본 이미지를 사용합니다.")
                    default_image_path = "assets/default_image.jpg"
                    if not os.path.exists(default_image_path):
                        try:
                            print("Downloading a placeholder image as default_image.jpg is not found.")
                            generic_image_url = "https://images.pexels.com/photos/936043/pexels-photo-936043.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2" # Example URL
                            image_data = requests.get(generic_image_url).content
                            with open(default_image_path, "wb") as f:
                                f.write(image_data)
                            print(f"✅ Placeholder image saved to: {default_image_path}")
                        except Exception as img_dl_e:
                            st.error(f"기본 이미지 다운로드에도 실패했습니다. 오류: {img_dl_e}")
                            st.stop()
                    image_paths = [default_image_path] 
                    
                st.success(f"이미지 {len(image_paths)}장 생성 완료.")

                # --- 4. 비디오 생성 (자막 제외) ---
                video_output_dir = "assets"
                os.makedirs(video_output_dir, exist_ok=True)
                temp_video_path = os.path.join(video_output_dir, "temp_video.mp4")
                final_video_path = os.path.join(video_output_dir, "final_video_with_subs.mp4")

                st.write("🎬 비디오 클립 조합 및 오디오 통합 중...")
                created_video_path = create_video_with_segments(
                    image_paths=image_paths,
                    segments=segments,
                    audio_path=audio_path,
                    topic_title=final_topic_for_video, # <--- 사용자가 수정/입력한 주제 사용
                    include_topic_title=True,
                    bgm_path=st.session_state.bgm_path, # 업로드된 BGM 경로 사용
                    save_path=temp_video_path
                )
                st.success(f"기본 비디오 생성 완료: {created_video_path}")

                # --- 5. 비디오에 자막 추가 ---
                st.write("📝 비디오에 자막 추가 중...")
                final_video_with_subs_path = add_subtitles_to_video(
                    input_video_path=created_video_path,
                    ass_path=ass_path,
                    output_path=final_video_path
                )
                st.success(f"✅ 최종 영상 생성 완료: {final_video_with_subs_path}")

                # --- 6. 결과 표시 및 다운로드 링크 제공 ---
                st.video(final_video_with_subs_path)
                with open(final_video_with_subs_path, "rb") as file:
                    st.download_button(
                        label="영상 다운로드",
                        data=file,
                        file_name="generated_multimodal_video.mp4",
                        mime="video/mp4"
                    )
                
                # Clean up temporary video file (optional)
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)

            except Exception as e:
                st.error(f"❌ 영상 생성 중 오류가 발생했습니다: {e}")
                st.exception(e)

    st.divider()
    if st.button("대화 초기화"):
        st.session_state.clear()
        st.rerun()

# 이전 대화 내용 표시
for i, message in enumerate(st.session_state["messages"]):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("참고한 출처 보기"):
                for j, source in enumerate(message["sources"]):
                    st.info(f"**출처 {j+1}**\n\n{source.page_content}")
                    st.divider()
        # "영상 만들기" 버튼은 이제 사이드바에 통합된 "영상 만들기" 버튼으로 대체됩니다.
        # 따라서 이 부분은 제거됩니다.
        # if message["role"] == "assistant" and message["content"]:
        #     if st.button("🎥 영상 만들기", key=f"generate_video_button_{i}"):
        #         # ... (영상 만들기 로직) ...
        #         pass # 이 로직은 이제 사이드바의 "영상 만들기" 버튼으로 이동합니다.

user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.last_user_query = user_input
    st.chat_message("user").write(user_input)

    try:
        chat_history = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" 
            else AIMessage(content=msg["content"])
            for msg in st.session_state.messages[:-1]
        ]
        
        if st.session_state.retriever:
            with st.chat_message("assistant"):
                with st.spinner("관련 문서를 검색하고 답변을 생성 중입니다..."):
                    retriever = st.session_state.retriever
                    source_documents = retriever.get_relevant_documents(user_input)
                    document_chain = get_document_chain(st.session_state.system_prompt)
                    
                    ai_answer = document_chain.invoke({
                        "input": user_input,
                        "chat_history": chat_history,
                        "context": source_documents
                    })
                    
                    st.markdown(ai_answer)
                    
                    st.session_state.messages.append({
                        "role": "assistant", "content": ai_answer, "sources": source_documents
                    })
                    
                    if source_documents:
                        with st.expander("참고한 출처 보기"):
                            for i, source in enumerate(source_documents):
                                st.info(f"**출처 {i+1}**\n\n{source.page_content}")
                                st.divider()
            # 챗봇이 답변을 생성한 후, 사이드바의 스크립트와 주제 필드를 자동으로 채웁니다.
            st.session_state.edited_script_content = ai_answer
            # 챗봇 답변에서 자동으로 주제를 추출하여 필드에 채웁니다.
            with st.spinner("답변에서 영상 주제를 자동으로 추출 중..."):
                topic_extraction_prompt = f"""다음 스크립트에서 이미지를 생성하기 위한 2-3개의 간결한 키워드 또는 아주 짧은 구문(최대 10단어)으로 메인 주제를 추출해주세요. 키워드/구문만 응답하세요.

                스크립트:
                {ai_answer}

                키워드/주제:"""
                topic_llm_chain = get_default_chain(system_prompt="당신은 주어진 텍스트에서 키워드를 추출하는 유용한 조수입니다.")
                extracted_topic_for_ui = topic_llm_chain.invoke({"question": topic_extraction_prompt, "chat_history": []}).strip()
                if extracted_topic_for_ui:
                    st.session_state.video_topic = extracted_topic_for_ui + "\n"
                else:
                    st.session_state.video_topic = user_input + "\n" # 추출 실패 시 사용자 질문을 기본값으로
            st.rerun() # UI 업데이트를 위해 rerun
        else:
            chain = get_default_chain(st.session_state.system_prompt)
            with st.chat_message("assistant"):
                container = st.empty()
                ai_answer = ""
                for token in chain.stream({"question": user_input, "chat_history": chat_history}):
                    ai_answer += token
                    container.markdown(ai_answer)
                st.session_state.messages.append({"role": "assistant", "content": ai_answer, "sources": []})
            # 챗봇이 답변을 생성한 후, 사이드바의 스크립트와 주제 필드를 자동으로 채웁니다.
            st.session_state.edited_script_content = ai_answer
            with st.spinner("답변에서 영상 주제를 자동으로 추출 중..."):
                topic_extraction_prompt = f"""다음 스크립트에서 이미지를 생성하기 위한 2-3개의 간결한 키워드 또는 아주 짧은 구문(최대 10단어)으로 메인 주제를 추출해주세요. 키워드/구문만 응답하세요.

                스크립트:
                {ai_answer}

                키워드/주제:"""
                topic_llm_chain = get_default_chain(system_prompt="당신은 주어진 텍스트에서 키워드를 추출하는 유용한 조수입니다.")
                extracted_topic_for_ui = topic_llm_chain.invoke({"question": topic_extraction_prompt, "chat_history": []}).strip()
                if extracted_topic_for_ui:
                    st.session_state.video_topic = extracted_topic_for_ui + "\n"
                else:
                    st.session_state.video_topic = user_input + "\n" # 추출 실패 시 사용자 질문을 기본값으로
            st.rerun() # UI 업데이트를 위해 rerun
    except Exception as e:
        st.chat_message("assistant").error(f"죄송합니다, 답변을 생성하는 중 오류가 발생했습니다.\n\n오류: {e}")
        st.session_state.messages.pop()