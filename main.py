import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from rag_pipeline import get_retriever_from_source, get_document_chain, get_default_chain, generate_topic_insights, rag_with_sources, generate_response_from_persona
from web_ingest import full_web_ingest # web_ingest는 별도로 정의되어 있어야 합니다.
from image_generator import generate_images_for_topic
from elevenlabs_tts import generate_tts, TTS_ELEVENLABS_TEMPLATES, TTS_POLLY_VOICES
from generate_timed_segments import generate_subtitle_from_script, generate_ass_subtitle, SUBTITLE_TEMPLATES
from video_maker import create_video_with_segments, add_subtitles_to_video
from deep_translator import GoogleTranslator
from file_handler import get_documents_from_files
import os
import requests # 기본 이미지 다운로드를 위해 추가
import re
import json # JSON 파싱을 위해 추가
import nest_asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

nest_asyncio.apply()

# API 키 불러오기
load_dotenv()

# --- 앱 기본 설정 ---
st.set_page_config(page_title="Perfacto AI", page_icon="🤖")
st.title("PerfactoAI")
st.markdown(
    """
Make your own vids automatically
"""
)            
# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "당신은 유능한 AI 어시스턴트입니다."
if "last_user_query" not in st.session_state:
    st.session_state.last_user_query = ""
if "video_topic" not in st.session_state:
    st.session_state.video_topic = ""
if "video_title" not in st.session_state: # 새롭게 추가된 부분: 영상 제목 세션 상태
    st.session_state.video_title = ""
if "edited_script_content" not in st.session_state:
    st.session_state.edited_script_content = ""
if "selected_tts_provider" not in st.session_state: # 새로운 TTS 공급자 세션 상태
    st.session_state.selected_tts_provider = "ElevenLabs" # 기본값 설정
if "selected_tts_template" not in st.session_state:
    st.session_state.selected_tts_template = "educational" # ElevenLabs 템플릿
if "selected_polly_voice_key" not in st.session_state: # Amazon Polly 음성 세션 상태
    st.session_state.selected_polly_voice_key = "korean_female" # 기본값 설정
if "selected_subtitle_template" not in st.session_state:
    st.session_state.selected_subtitle_template = "educational"
if "bgm_path" not in st.session_state:
    st.session_state.bgm_path = None
if "include_voice" not in st.session_state:
    st.session_state.include_voice = True
if "generated_topics" not in st.session_state:
    st.session_state.generated_topics = []
if "selected_generated_topic" not in st.session_state:
    st.session_state.selected_generated_topic = ""
if "audio_path" not in st.session_state: 
    st.session_state.audio_path = None
if "last_rag_sources" not in st.session_state:
    st.session_state.last_rag_sources = []
if "persona_rag_flags" not in st.session_state:
    st.session_state.persona_rag_flags = {}  # 각 페르소나가 RAG 사용할지 여부
if "persona_rag_retrievers" not in st.session_state:
    st.session_state.persona_rag_retrievers = {}  # 각 페르소나 전용 retriever


# --- 사이드바: AI 페르소나 설정 및 RAG 설정 ---
with st.sidebar:
    st.header("⚙️ AI 페르소나 및 RAG 설정")

    if "persona_blocks" not in st.session_state:
        st.session_state.persona_blocks = []

    delete_idx = None

    # --- 페르소나 문장 기반 생성기 (범용 응답 생성) ---
    if st.button("➕ 페르소나 추가"):
        st.session_state.persona_blocks.append({
            "name": "새 페르소나",
            "text": "",
            "use_prev_idx": [],
            "result": ""
        })

    for i, block in enumerate(st.session_state.persona_blocks):
        st.markdown(f"---\n### 페르소나 #{i+1} - `{block['name']}`")

        st.session_state.persona_blocks[i]["name"] = st.text_input(
            "페르소나 역할 이름", value=block["name"], key=f"name_{i}"
        )

        # 기본 옵션 (페르소나 블록 인덱스)
        persona_options = [("persona", idx) for idx in range(len(st.session_state.persona_blocks)) if idx != i]

        # 멀티 셀렉트 구성
        prev_idxs = st.multiselect(
            "이전 페르소나 응답 이어받기",
            options=persona_options,
            default=block.get("use_prev_idx", []),
            key=f"use_prev_idx_{i}"
        )
        st.session_state.persona_blocks[i]["use_prev_idx"] = prev_idxs

        st.session_state.persona_blocks[i]["text"] = st.text_area(
            "지시 문장", value=block["text"], key=f"text_{i}"
        )
        use_rag = st.checkbox("🔎 이 페르소나에 RAG 사용", value=st.session_state.persona_rag_flags.get(i, False), key=f"use_rag_{i}")
        st.session_state.persona_rag_flags[i] = use_rag

        if use_rag:
            with st.expander("RAG 설정", expanded=True):
                url_input = st.text_input("웹 키워드 입력", key=f"url_input_{i}")
                uploaded_files = st.file_uploader("파일 업로드 (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True, key=f"files_{i}")

                if st.button("📄 RAG 문서 분석", key=f"rag_analyze_{i}"):
                    all_documents = []

                    if uploaded_files:
                        file_docs = get_documents_from_files(uploaded_files)
                        all_documents.extend(file_docs)
                        st.success(f"{len(file_docs)}개의 파일 문서 로드 완료.")

                    if url_input:
                        web_docs, error = full_web_ingest(url_input)
                        if not error:
                            all_documents.extend(web_docs)
                            st.success(f"{len(web_docs)}개의 웹 문서 로드 완료.")
                        else:
                            st.error(f"웹페이지 수집 오류: {error}")

                    if all_documents:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        split_docs = text_splitter.split_documents(all_documents)

                        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                        vectorstore = FAISS.from_documents(split_docs, embedding)
                        st.session_state.persona_rag_retrievers[i] = vectorstore.as_retriever()
                        st.success("이 페르소나에 대한 문서 분석이 완료되었습니다.")

        if st.button(f"🧠 페르소나 실행", key=f"run_{i}"):
            prev_blocks = []
            for ptype, pidx in st.session_state.persona_blocks[i].get("use_prev_idx", []):
                if ptype == "persona" and pidx != i:
                    prev_blocks.append(f"[페르소나 #{pidx+1}]\n{st.session_state.persona_blocks[pidx]['result']}")

            joined_prev = "\n\n".join(prev_blocks)
            final_prompt = f"{joined_prev}\n\n지시:\n{block['text']}" if joined_prev else block["text"]

            if use_rag and i in st.session_state.persona_rag_retrievers:
                rag_result = rag_with_sources({
                    "input": final_prompt,
                    "chat_history": [],
                    "retriever": st.session_state.persona_rag_retrievers[i]
                })
                result_text = rag_result.get("answer", "")
                st.session_state.messages.append(
                    AIMessage(content=result_text, additional_kwargs={"sources": rag_result.get("sources", [])})
                )
                st.session_state.persona_blocks[i]["result"] = result_text
            else:
                result_text = generate_response_from_persona(final_prompt)
                st.session_state.messages.append(
                    AIMessage(content=result_text)
                )
                st.session_state.persona_blocks[i]["result"] = result_text

        if st.button(f"🗑️ 페르소나 삭제", key=f"delete_{i}"):
            delete_idx = i

    if delete_idx is not None:
        del st.session_state.persona_blocks[delete_idx]
        st.rerun()
    
    st.markdown("---")

    with st.expander("영상 제작 설정", expanded=True):
        st.subheader("📜 사용할 스크립트 선택 (페르소나 응답 중)")

        available_personas_with_results = [
            (i, block["name"]) for i, block in enumerate(st.session_state.persona_blocks)
            if block.get("result", "").strip()
        ]

        if available_personas_with_results:
            selected_script_persona_idx = st.selectbox(
                "스크립트로 사용할 페르소나 선택:",
                options=available_personas_with_results,
                format_func=lambda x: f"{x[0]+1} - {x[1]}",
                key="selected_script_persona_for_video",
                index=0
            )
            # 선택된 페르소나 응답을 영상 스크립트로 설정
            st.session_state.edited_script_content = st.session_state.persona_blocks[selected_script_persona_idx[0]]["result"]
        else:
            st.warning("사용 가능한 페르소나 결과가 없습니다. 먼저 페르소나 실행을 통해 결과를 생성해 주세요.")

        # 영상 주제 입력 필드
        st.session_state.video_topic = st.text_input(
            "이미지 생성에 사용될 키워드",
            value=st.session_state.video_topic,
            key="video_topic_input_final"
        )
        
        # 영상 주제 입력 필드 이름 변경 (Moved here)
        st.session_state.video_topic = st.text_input(
            "이미지 생성에 사용될 키워드", # 필드 이름 변경
            value=st.session_state.video_topic, # 세션 상태에서 가져옴
            key="video_topic_input_final" # Changed key to avoid conflict if any
        )

        # 새롭게 추가된 부분: 영상 제목 입력 필드
        st.session_state.video_title = st.text_input(
            "영상 제목 (영상 위에 표시될 제목)", # 필드 이름
            value=st.session_state.video_title, # 세션 상태에서 가져옴
            key="video_title_input_final" # 새로운 키
        )

        # 음성 포함 여부 선택
        st.session_state.include_voice = st.checkbox("영상에 AI 목소리 포함", value=st.session_state.include_voice)

        if st.session_state.include_voice:
            # TTS 서비스 공급자 선택 라디오 버튼 추가
            st.session_state.selected_tts_provider = st.radio(
                "음성 서비스 공급자 선택:",
                ("ElevenLabs", "Amazon Polly"),
                index=0 if st.session_state.selected_tts_provider == "ElevenLabs" else 1,
                key="tts_provider_select"
            )

            if st.session_state.selected_tts_provider == "ElevenLabs":
                # ElevenLabs 템플릿 선택
                elevenlabs_template_names = list(TTS_ELEVENLABS_TEMPLATES.keys())
                st.session_state.selected_tts_template = st.selectbox(
                    "ElevenLabs 음성 템플릿 선택:",
                    options=elevenlabs_template_names,
                    index=elevenlabs_template_names.index(st.session_state.selected_tts_template) if st.session_state.selected_tts_template in elevenlabs_template_names else 0,
                    key="elevenlabs_template_select"
                )
                # ElevenLabs는 voice_id를 따로 받을 수도 있지만, 여기서는 템플릿으로만 통일하여 간결하게 합니다.
                # 만약 특정 Voice ID를 직접 입력받고 싶다면 추가적인 text_input을 구성할 수 있습니다.

            elif st.session_state.selected_tts_provider == "Amazon Polly":
                # Amazon Polly 음성 선택
                polly_voice_keys = list(TTS_POLLY_VOICES.keys())
                st.session_state.selected_polly_voice_key = st.selectbox(
                    "Amazon Polly 음성 선택:",
                    options=polly_voice_keys,
                    index=polly_voice_keys.index(st.session_state.selected_polly_voice_key) if st.session_state.selected_polly_voice_key in polly_voice_keys else 0,
                    key="polly_voice_select"
                )

        # 자막 템플릿 선택
        st.session_state.selected_subtitle_template = st.selectbox(
            "자막 템플릿 선택",
            options=list(SUBTITLE_TEMPLATES.keys()),
            index=list(SUBTITLE_TEMPLATES.keys()).index(st.session_state.selected_subtitle_template)
        )

        # BGM 파일 업로드 (선택 사항)
        uploaded_bgm_file = st.file_uploader("BGM 파일 업로드 (선택 사항, .mp3, .wav)", type=["mp3", "wav"])
        if uploaded_bgm_file:
            temp_bgm_path = os.path.join("assets", uploaded_bgm_file.name) # Use original filename
            os.makedirs("assets", exist_ok=True)
            with open(temp_bgm_path, "wb") as f:
                f.write(uploaded_bgm_file.read())
            st.session_state.bgm_path = temp_bgm_path
            st.success(f"배경 음악 '{uploaded_bgm_file.name}' 업로드를 완료했어요!")
        else:
            # If no file is uploaded, and there was a previous BGM, keep it unless explicitly cleared.
            pass # Keep existing bgm_path if no new file is uploaded

        st.subheader("영상 제작")
        if st.button("영상 만들기"):
            # 사용자가 수정한 스크립트 내용과 주제를 사용
            final_script_for_video = st.session_state.edited_script_content
            final_topic_for_video = st.session_state.video_topic
            final_title_for_video = st.session_state.video_title # 새롭게 추가된 부분: 최종 영상 제목

            if not final_script_for_video.strip():
                st.error("스크립트 내용이 비어있습니다. 스크립트를 입력하거나 생성해주세요.")
                st.stop()
            if not final_topic_for_video.strip():
                st.error("영상 주제가 비어있습니다. 주제를 입력해주세요.")
                st.stop()
            if not final_title_for_video.strip(): # 새롭게 추가된 부분: 영상 제목 유효성 검사
                st.error("영상 제목이 비어있습니다. 제목을 입력하거나 생성해주세요.")
                st.stop()

            with st.spinner("✨ 영상 제작 중입니다..."):
                try:
                    # --- 0-1. 추출된 토픽을 영어로 번역 (GoogleTranslator 사용) ---
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

                    audio_path = None
                    segments = []

                    if st.session_state.include_voice:
                        # --- 1. Text-to-Speech (TTS) 생성 ---
                        audio_output_dir = "assets"
                        os.makedirs(audio_output_dir, exist_ok=True)
                        audio_path = os.path.join(audio_output_dir, "generated_audio.mp3")
                        
                        st.write("🗣️ 음성 파일 생성 중...")
                        
                        if st.session_state.selected_tts_provider == "ElevenLabs":
                            generated_audio_path = generate_tts(
                                text=final_script_for_video,
                                save_path=audio_path,
                                provider="elevenlabs", # 공급자 명시
                                template_name=st.session_state.selected_tts_template # ElevenLabs 템플릿
                            )
                        elif st.session_state.selected_tts_provider == "Amazon Polly":
                            generated_audio_path = generate_tts(
                                text=final_script_for_video,
                                save_path=audio_path,
                                provider="polly", # 공급자 명시
                                polly_voice_name_key=st.session_state.selected_polly_voice_key # Polly 음성 키
                            )

                        st.success(f"음성 파일 생성 완료: {generated_audio_path}")
                        st.session_state.audio_path = generated_audio_path # Store audio path in session state

                        # --- 2. Audio Transcription (ASR) 및 Subtitle (ASS) 파일 생성 ---
                        subtitle_output_dir = "assets"
                        os.makedirs(subtitle_output_dir, exist_ok=True)
                        ass_path = os.path.join(subtitle_output_dir, "generated_subtitle.ass")
                        
                        audio_save_path = "assets/generated_audio.mp3"
                        full_audio_path = generate_tts(
                        text=final_script_for_video,
                        save_path=audio_save_path,
                        provider="elevenlabs" if st.session_state.selected_tts_provider == "ElevenLabs" else "polly",
                        template_name=st.session_state.selected_tts_template if st.session_state.selected_tts_provider == "ElevenLabs"
                                else st.session_state.selected_polly_voice_key
                    )

                        st.write("📝 자막 생성을 위한 음성 분석 중...")
                        segments, audio_clips, ass_path = generate_subtitle_from_script(
                        script_text=final_script_for_video,
                        ass_path=ass_path,
                        full_audio_file_path=full_audio_path, 
                        provider="elevenlabs" if st.session_state.selected_tts_provider == "ElevenLabs" else "polly",
                        template=st.session_state.selected_tts_template if st.session_state.selected_tts_provider == "ElevenLabs"
                                else st.session_state.selected_polly_voice_key
                    )
                        st.success(f"자막 파일 생성 완료: {ass_path}")
                    else: # 음성이 없는 경우
                        st.write("음성 없이 자막과 이미지만으로 영상을 생성합니다.")

                        # 스크립트를 문장 단위로 분할
                        sentences = re.split(r'(?<=[.?!])\s*', final_script_for_video.strip())
                        sentences = [s.strip() for s in sentences if s.strip()]

                        if not sentences:
                            sentences = [final_script_for_video.strip()] # 전체 스크립트를 하나의 문장으로

                        words_per_minute = 150 # 분당 단어 수 (평균적인 읽기 속도)
                        total_script_words = len(final_script_for_video.split())
                        total_estimated_duration_seconds = (total_script_words / words_per_minute) * 60

                        if total_estimated_duration_seconds < 5: # 너무 짧은 영상 방지 (최소 5초)
                            total_estimated_duration_seconds = 5

                        current_time = 0.0 # 현재 시간 (누적)
                        segments = [] # 최종 segments 리스트

                        # total_chars 계산 (이전 코드에서 누락되어 있던 부분)
                        total_chars = sum(len(s) for s in sentences)

                        for sentence_text in sentences:
                            min_segment_duration = 1.5 # 초

                            if total_chars > 0: # 0으로 나누는 오류 방지
                                proportion = len(sentence_text) / total_chars
                                segment_duration = total_estimated_duration_seconds * proportion
                            else: # 스크립트가 비어있거나 특수한 경우 (이 경우는 거의 없겠지만 안전장치)
                                segment_duration = total_estimated_duration_seconds / len(sentences)

                            segment_duration = max(min_segment_duration, segment_duration)

                            segments.append({
                                "start": current_time,
                                "end": current_time + segment_duration,
                                "text": sentence_text
                            })
                            current_time += segment_duration

                        if segments:
                            segments[-1]["end"] = current_time 

                        subtitle_output_dir = "assets"
                        os.makedirs(subtitle_output_dir, exist_ok=True)
                        ass_path = os.path.join(subtitle_output_dir, "generated_subtitle.ass")

                        st.write("📝 자막 파일 생성 중...")
                        generate_ass_subtitle(
                            segments=segments,
                            ass_path=ass_path,
                            template_name=st.session_state.selected_subtitle_template
                        )
                        st.success(f"자막 파일 생성 완료: {ass_path}")

                    # --- 3. 이미지 생성 ---
                    num_images = max(3, len(segments)) if segments else 3 # 최소 3장 또는 세그먼트 수만큼
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
                        image_paths = [default_image_path] * num_images # Ensure enough default images
                        
                    st.success(f"이미지 {len(image_paths)}장 생성 완료.")

                    # --- 4. 비디오 생성 (자막 제외) ---
                    video_output_dir = "assets"
                    os.makedirs(video_output_dir, exist_ok=True)
                    temp_video_path = os.path.join(video_output_dir, "temp_video.mp4")
                    final_video_path = os.path.join(video_output_dir, "final_video_with_subs.mp4")

                    st.write("🎬 비디오 클립 조합 및 오디오 통합 중...")
                    created_video_path = create_video_with_segments(
                        image_paths=image_paths,
                        segments=segments, # segments를 사용하여 이미지 지속 시간 결정
                        audio_path=audio_path if st.session_state.include_voice else None, # 음성 미포함 시 None 전달
                        topic_title=final_title_for_video, # 새롭게 수정된 부분: 영상 제목을 전달
                        include_topic_title=True,
                        bgm_path=st.session_state.bgm_path,
                        save_path=temp_video_path,
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


# --- 메인 채팅 인터페이스 ---
for message in st.session_state.messages:
    # message 객체의 'type' 속성 (예: 'human', 'ai')을 사용합니다.
    with st.chat_message(message.type):
        # message 객체의 'content' 속성을 사용합니다.
        st.markdown(message.content)

        # AI 메시지이고, 추가적인 인자 (additional_kwargs)에 'sources'가 있다면 표시
        # AIMessage 객체에 'sources'를 직접 추가하는 대신 'additional_kwargs'에 저장됩니다.
        if message.type == "ai" and hasattr(message, "additional_kwargs") and "sources" in message.additional_kwargs and message.additional_kwargs["sources"]:
            st.subheader("📚 참고 문단 (RAG 기반)")
            for idx, source_item in enumerate(message.additional_kwargs["sources"], start=1):
                content_display = source_item["content"]
                source_url_display = source_item.get("source", "N/A")
                
                # 내용이 너무 길면 줄이기
                if len(content_display) > 200:
                    content_display = content_display[:200] + "..."
                
                # URL이 있으면 링크와 함께 표시
                if source_url_display != 'N/A':
                    st.markdown(f"**출처 {idx}:** [{source_url_display}]({source_url_display})\n> {content_display}")
                else:
                    st.markdown(f"**출처 {idx}:**\n> {content_display}")


# --- 챗봇 입력 및 응답 ---
if user_input := st.chat_input("메시지를 입력해 주세요 (예: 최근 AI 기술 트렌드 알려줘)"):
    st.session_state.messages.append(HumanMessage(content=user_input)) # Langchain HumanMessage 사용
    with st.chat_message("human"):
        st.markdown(user_input)
    st.session_state.last_user_query = user_input # 마지막 사용자 쿼리 저장

    with st.chat_message("ai"):
        container = st.empty()
        ai_answer = ""
        sources_list = []
        
        # RAG 사용 여부 결정 (URL 또는 파일이 처리된 경우)
        if st.session_state.retriever:
            inputs_for_rag = {"input": user_input, "chat_history": st.session_state.messages}
            rag_output = rag_with_sources(inputs_for_rag)
            ai_answer = rag_output["answer"]
            sources_list = rag_output["sources"]

            #현재 문서에서 핵심 키워드를 입력하지 못하면 답변을 잘 못해서 추가함
            if not sources_list:
                ai_answer = "⚠️ **문서에서 관련 내용을 찾지 못해 AI의 일반적인 지식으로 답변합니다.**\n\n" + ai_answer
            

            container.markdown(ai_answer)

        else:
            # 일반 챗봇 모드 (RAG 비활성화)
            chain = get_default_chain(st.session_state.system_prompt)
            
            # 챗봇 스트리밍 응답
            for token in chain.stream({"question": user_input, "chat_history": st.session_state.messages}):
                ai_answer += token
                container.markdown(ai_answer)
        
        st.session_state.messages.append(AIMessage(content=ai_answer, sources=sources_list)) # sources도 함께 저장
        
        # RAG 기반 출처 표시 (이전 코드를 통합)
        if sources_list: # sources_list에 값이 있을 때만 표시
            st.write("### 📚 참고 문단 (RAG 기반)")
            for idx, source_item in enumerate(sources_list, start=1):
                content_display = source_item["content"]
                source_url_display = source_item.get("source", "N/A")
                
                # 내용이 너무 길면 줄이기
                if len(content_display) > 200:
                    content_display = content_display[:200] + "..."
                
                # URL이 있으면 링크와 함께 표시
                if source_url_display != 'N/A':
                    st.markdown(f"**출처 {idx}:** [{source_url_display}]({source_url_display})\n> {content_display}")
                else:
                    st.markdown(f"**출처 {idx}:**\n> {content_display}")



    # 챗봇이 답변을 생성한 후, 사이드바의 스크립트와 주제 필드를 자동으로 채웁니다.
    # 일반 챗봇 답변을 스크립트로 활용
    st.session_state.edited_script_content = ai_answer
    with st.spinner("답변에서 영상 주제를 자동으로 추출하고 있습니다..."):
        topic_extraction_prompt = f"""다음 스크립트에서 이미지를 생성하기 위한 2-3개의 간결한 키워드 또는 아주 짧은 구문(최대 10단어)으로 메인 주제를 추출해주세요. 키워드/구문만 응답하세요.

        스크립트:
        {ai_answer}

        키워드/주제:"""
        topic_llm_chain = get_default_chain(system_prompt="당신은 주어진 텍스트에서 키워드를 추출하는 유용한 조수입니다.")
        extracted_topic_for_ui = topic_llm_chain.invoke({"question": topic_extraction_prompt, "chat_history": []}).strip()
        if extracted_topic_for_ui:
            st.session_state.video_topic = extracted_topic_for_ui

    # 새롭게 추가된 부분: 챗봇 답변에서 영상 제목 자동 추출
    with st.spinner("답변에서 영상 제목을 자동으로 추출하고 있습니다..."):
        title_extraction_prompt = f"""당신은 TikTok, YouTube Shorts, Instagram Reels과 같은 **매력적이고 바이럴성 있는 숏폼 비디오 제목**을 작성하는 전문 크리에이터입니다.
다음 스크립트에서 시청자의 스크롤을 멈추게 할 수 있는, **최대 5단어 이내의 간결하고 임팩트 있는 한국어 제목**을 생성해주세요.
이 제목은 호기심을 유발하고, 핵심 내용을 빠르게 전달하며, 클릭을 유도하는 강력한 후크 역할을 해야 합니다.
**예시: '체스 초고수 꿀팁!', '이거 알면 체스 끝!', '체스 천재되는 법?'**
**제목만 응답하세요.**

        스크립트:
        {ai_answer}

        영상 제목:"""
        title_llm_chain = get_default_chain(
    system_prompt="당신은 숏폼(Shorts) 비디오를 위한 매우 짧고 강렬한 한국어 제목을 생성하는 전문 AI입니다. 항상 5단어 이내로, 시청자의 호기심을 극대화하는 제목을 만드세요."
)
        extracted_title_for_ui = title_llm_chain.invoke({"question": title_extraction_prompt, "chat_history": []}).strip()
        if extracted_title_for_ui:
            st.session_state.video_title = extracted_title_for_ui
        else:
            st.session_state.video_title = "제목 없음" # 추출 실패 시 기본값
