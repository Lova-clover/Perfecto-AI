from deep_translator import GoogleTranslator
from ssml_converter import convert_line_to_ssml, breath_linebreaks, koreanize_if_english
from html import escape as _xml_escape
# generate_timed_segments.py
import os
import re, math
from elevenlabs_tts import generate_tts
from pydub import AudioSegment
import kss
import boto3, json
from elevenlabs_tts import TTS_POLLY_VOICES 
from botocore.exceptions import ClientError

def _visible_len(s: str) -> int:
    # 래핑 판단용 길이(개략). NBSP는 공백 취급.
    return len((s or "").replace(NBSP, " "))

def lock_oneliner_if_short(text: str, threshold: int = 12) -> str:
    if _visible_len(text) <= threshold:
        return (text or "").replace(" ", NBSP)
    return text

def _split_script_by_llm_breath(script_text: str) -> list[str]:
    """
    스크립트를 문단별로 쪼갠 뒤, 각 문단을 LLM 브레스 라인들로 확장.
    반환: 최종 TTS/세그먼트용 '한 줄씩' 리스트(빈 줄 제거).
    """
    blocks = [b.strip() for b in re.split(r"\n{2,}", script_text or "") if b.strip()]
    lines: list[str] = []
    for b in blocks:
        try:
            parts = [ln for ln in breath_linebreaks(b) if ln.strip()]
        except Exception:
            parts = [b]
        lines.extend(parts)
    return lines or [script_text.strip()]

def split_script_to_lines(script_text, mode="llm"):
    """항상 LLM 브레스 분절을 사용한다."""
    text = script_text or ""
    lines = breath_linebreaks(text)  # LLM 호출
    return [ln.strip() for ln in lines if ln.strip()]

def _drop_special_except_q(text: str) -> str:
    # 한글/영문/숫자/공백/물음표만 남김
    return re.sub(r"[^0-9A-Za-z\uac00-\ud7a3?\s]", "", text or "")

def _summarize_line_pitch(ssml: str) -> float | None:
    """라인 전체의 pitch를 요약. 빨강/파랑 임계에 의미 있게 반응하도록 설계."""
    pieces = _parse_ssml_pieces(ssml or "")
    if not pieces:
        return None
    vals = [float(p.get("pitch_pct", 0)) for p in pieces if "pitch_pct" in p]
    if not vals:
        return None
    # 규칙: 낮음 경고 우선, 그다음 높음, 아니면 평균
    low = min(vals)
    high = max(vals)
    if low <= -4:           # 빨강 임계 충족 시 그 값을 사용
        return low
    if high >= +8:          # 파랑 임계 충족 시 그 값을 사용
        return high
    return sum(vals)/len(vals)

def _pitch_to_hex(p):
    """
    ASS는 BGR 순서.
    - None/미미: 기본(하양)
    - 낮음(<= -6): 빨강  -> &H0000FF&
    - 높음(>= +6): 파랑 -> &HFF0000&
    """
    try:
        v = float(p)
    except Exception:
        return None
    if v <= -4:
        return "&H0000FF&"  # red
    if v >= +8:
        return "&HFF0000&"  # blue
    return None

def _strip_trailing_commas(s: str) -> str:
    return re.sub(r'[,\s]+$', '', (s or '').strip())

def harden_ko_sentence_boundaries(segments):
    """
    한국어 문장 경계를 더 강하게 보정:
    - '?', '…다/요/니다/습니다/입니다' 끝이 아니면 뒤 조각과 병합
    - 다음 조각이 '이다/것이다/수 있다/해야 한다/이 숫자면/하지만/그리고' 류 접속부면 병합
    - 매우 짧은 꼬리(<=6자)는 앞에 붙임
    """
    END_STRONG_RE = re.compile(r'(?:\?|!|\.|다|요|니다|습니다|입니다|예요|이에요|였다|겠[다습니다])$')
    NEXT_TAIL_RE  = re.compile(r'^(?:그리고|하지만|근데|그런데|그래서|그러니까|즉|특히|게다가|한편|반면에|다만|'
                               r'이다|것이다|것입니다|수 있다|수 없다|해야 한다|이 숫자면)\b')

    out = []
    i = 0
    while i < len(segments):
        cur = dict(segments[i]); i += 1
        cur_text = (cur.get("text") or "").strip()

        while i < len(segments):
            nxt = segments[i]
            nxt_text = (nxt.get("text") or "").strip()

            # 이미 강한 끝이면 멈춤
            if END_STRONG_RE.search(cur_text):
                break
            # 다음이 접속/말꼬리로 시작하거나 다음이 너무 짧은 꼬리면 병합
            if NEXT_TAIL_RE.match(nxt_text) or len(nxt_text) <= 6:
                cur["end"]  = float(nxt["end"])
                cur_text    = (cur_text.rstrip() + " " + nxt_text.lstrip()).strip()
                cur["text"] = cur_text
                i += 1
                continue
            break

        out.append(cur)
    return out

def _parse_ssml_pieces(ssml: str):
    """<prosody ...>text</prosody> (+ 선택적 <break>) 를 순서대로 추출"""
    if not ssml: return []
    body = ssml
    if body.strip().startswith("<speak"):
        body = re.sub(r"^<speak[^>]*>|</speak>\s*$", "", body.strip(), flags=re.I)

    pieces = []
    pos = 0
    tag_re = re.compile(r'<prosody\b([^>]*)>(.*?)</prosody\s*>', re.I|re.S)
    brk_re = re.compile(r'<break\b[^>]*time="(\d+)ms"[^>]*/\s*>', re.I)

    for m in tag_re.finditer(body):
        attrs = m.group(1) or ""
        text  = (m.group(2) or "").strip()
        rate  = re.search(r'rate="([+\-]?\d+)%"', attrs)
        pitch = re.search(r'pitch="([+\-]?\d+)%"', attrs)
        rate_pct  = int(rate.group(1)) if rate else 150  # 기본 150%
        pitch_pct = int(pitch.group(1)) if pitch else 0

        # 이 prosody 뒤에 즉시 오는 break 1개를 소비(있으면)
        tail = body[m.end():]
        brk = brk_re.match(tail)
        brk_ms = int(brk.group(1)) if brk else 0

        if brk:  # 소비된 break는 본문에서 제거된다고 가정
            pass

        pieces.append({
            "text": text,
            "rate_pct": rate_pct,
            "pitch_pct": pitch_pct,
            "break_ms": brk_ms,
        })
    return [p for p in pieces if p["text"]]

def _quantize_segments(segs, fps=24.0, clamp_start=None, clamp_end=None):
    """ASS/비디오 타임라인(24fps)에 맞춰 시작/끝을 프레임 단위로 스냅."""
    tick = 1.0 / float(fps)
    out, prev_end = [], None
    for s in segs:
        st = round(s["start"] / tick) * tick
        en = round(s["end"]   / tick) * tick
        if prev_end is not None and st < prev_end:
            st = prev_end
        if en <= st:  # 최소 1프레임
            en = st + tick
        out.append({**s, "start": st, "end": en})
        prev_end = en
    if clamp_start is not None:
        out[0]["start"] = max(clamp_start, out[0]["start"])
    if clamp_end is not None:
        out[-1]["end"]  = min(clamp_end,  out[-1]["end"])
    return out

def _pitch_level_from_attr(pitch_str: str) -> str:
    # "+20%" / "-15%" / "+3st" 등 → 대략 퍼센트/정수만 추출
    import re
    m = re.search(r"(-?\d+)\s*%?", pitch_str or "")
    v = int(m.group(1)) if m else 0
    if v >= +10: return "high"
    if v <= -10: return "low"
    return "mid"

def _build_dense_from_ssml(line_ssml: str, seg_start: float, seg_end: float, fps: float = 24.0):
    """
    한 줄(오디오 한 파일) SSML을 prosody 조각 단위로 시간 분배 → dense events 반환
    - 각 이벤트에 pitch(숫자 %), pitch_level(high/mid/low) 포함
    """
    pcs = _parse_ssml_pieces(line_ssml)  # ← 기존 함수 사용
    if not pcs:
        return []

    # 안전 디폴트
    for p in pcs:
        p.setdefault("text", "")
        p.setdefault("rate_pct", 150)   # 보통 100~200%, 내부 가중치 기준 150을 중심으로
        p.setdefault("pitch_pct", 0)
        p.setdefault("break_ms", 0)

    dur = max(0.01, seg_end - seg_start)
    total_break = sum(p["break_ms"] for p in pcs) / 1000.0
    speech_dur  = max(0.0, dur - total_break)

    # rate 반영 가중치 (rate 높을수록 같은 글자수라도 더 빨리 읽으니 시간 적게 배분)
    weights = []
    for p in pcs:
        char_len = max(1, len(p["text"]))
        rate_mul = max(0.1, float(p["rate_pct"]) / 150.0)  # 150%를 중심값으로
        w = char_len / rate_mul
        weights.append(w)
    W = sum(weights) or 1.0

    t = seg_start
    events = []
    for p, w in zip(pcs, weights):
        span = speech_dur * (w / W)
        t0 = t
        t1 = min(seg_end, t0 + span)

        pitch_pct = float(p.get("pitch_pct", 0))
        pitch_lvl = "high" if pitch_pct >= 10 else ("low" if pitch_pct <= -10 else "mid")

        events.append({
            "start": t0,
            "end":   t1,
            "text":  p["text"],
            "pitch": pitch_pct,        # 숫자 % (색상 매핑 시 사용)
            "pitch_level": pitch_lvl,  # 필요하면 문자열 레벨도 사용 가능
        })

        # prosody 사이의 break 반영
        t = t1 + (p["break_ms"] / 1000.0)

    # 프레임 격자 스냅 + 범위 클램프 (프로젝트에 이미 있는 헬퍼 사용)
    try:
        return _quantize_segments(events, fps=fps, clamp_start=seg_start, clamp_end=seg_end)
    except NameError:
        # fallback: 아주 얕은 스냅
        tick = 1.0 / float(fps)
        out = []
        for ev in events:
            s = max(seg_start, round(ev["start"] / tick) * tick)
            e = min(seg_end,   round(ev["end"]   / tick) * tick)
            if e <= s: e = s + tick
            out.append({**ev, "start": round(s, 3), "end": round(e, 3)})
        # 겹침 방지
        for i in range(len(out) - 1):
            if out[i]["end"] > out[i+1]["start"]:
                out[i]["end"] = max(out[i]["start"] + 0.02, out[i+1]["start"] - 0.001)
        return out

def _clean_for_align(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z\uac00-\ud7a3]+", "", s or "").strip()

def _align_breath_to_wordmarks(breath_lines, marks, line_offset, line_end, min_piece_dur=0.35):
    """
    breath_lines: 호흡 조각 리스트(원문 보존 줄)
    marks: Polly speechmarks(type='word') for the line
    반환: [{start, end, text, pitch}]
    """
    words = [(line_offset + (mk["time"]/1000.0), mk.get("value",""))
             for mk in marks if mk.get("type") == "word"]
    if not words:
        return []

    pieces, w_idx, w_n = [], 0, len(words)
    for breath in breath_lines:
        target_len = len(_clean_for_align(breath))
        if target_len == 0:  # 빈 호흡조각 방지
            continue

        st = words[w_idx][0] if w_idx < w_n else line_offset
        acc = ""
        while w_idx < w_n and len(_clean_for_align(acc)) < target_len:
            acc += words[w_idx][1]
            w_idx += 1

        en = words[w_idx][0] if w_idx < w_n else line_end

        st = max(line_offset, min(st, line_end))
        en = max(st, min(en, line_end))

        txt = re.sub(r"\s+", " ", breath).strip()
        if not txt:
            continue

        if (en - st) < min_piece_dur and pieces:
            pieces[-1]["end"]  = en
            pieces[-1]["text"] = _join_no_repeat(pieces[-1]["text"], txt)
        else:
            pieces.append({"start": st, "end": en, "text": txt, "pitch": _assign_pitch(txt)})

    if len(pieces) >= 2 and (pieces[-1]["end"] - pieces[-1]["start"]) < min_piece_dur:
        pieces[-2]["end"]  = pieces[-1]["end"]
        pieces[-2]["text"] = _join_no_repeat(pieces[-2]["text"], pieces[-1]["text"])
        pieces.pop()

    return pieces

def _join_no_repeat(a: str, b: str) -> str:
    import re
    A = re.sub(r"\s+", " ", (a or "")).strip()
    B = re.sub(r"\s+", " ", (b or "")).strip()
    if not A: return B
    if not B: return A
    if B in A: return A             # b가 a에 완전히 포함 → a만
    if A in B: return B             # a가 b에 완전히 포함 → b만
    A_toks, B_toks = A.split(), B.split()
    k = min(len(A_toks), len(B_toks))
    for n in range(k, 0, -1):
        # a의 접미 == b의 접두 → 겹친 부분 빼고 붙이기
        if A_toks[-n:] == B_toks[:n]:
            return " ".join(A_toks + B_toks[n:])
    return A + " " + B

def dedupe_adjacent_texts(segs):
    out = []
    prev_clean = None
    for s in segs:
        t = (s.get("text") or "").strip()
        t_clean = re.sub(r"\s+", " ", strip_ssml_tags(t)).strip()
        if out and t_clean == prev_clean:
            # 바로 이전과 동일하면 시간만 이어붙임
            out[-1]["end"] = max(out[-1]["end"], s["end"])
        else:
            out.append(dict(s))
            prev_clean = t_clean
    return out

TAG_RE = re.compile(r"<[^>]+>")
def strip_ssml_tags(s: str) -> str:
    return TAG_RE.sub(" ", s or "")

# --- SSML guard helpers (원문≠SSML 불일치/중복 방지) ---
import re as _re_guard

def _plain_text_from_ssml(ssml: str) -> str:
    t = _re_guard.sub(r"<[^>]+>", " ", ssml)  # 태그 제거
    return _re_guard.sub(r"\s+", " ", t).strip()

def _tokenize_ko_en(s: str):
    # 한글/영문/숫자만 토큰화(기호/공백 무시)
    return _re_guard.findall(r"[0-9A-Za-z\uac00-\ud7a3]+", s or "")

def _ssml_safe_or_fallback(orig_line: str, ssml_fragment: str):
    """원문과 LLM-SSML 불일치/중복이면 결정적 SSML로 폴백"""
    plain = _plain_text_from_ssml(ssml_fragment)
    tok_o = _tokenize_ko_en(orig_line)
    tok_s = _tokenize_ko_en(plain)

    # 새 단어 삽입 탐지
    inserted = [t for t in tok_s if t not in tok_o]
    # 연속 중복 탐지(“어떻게 어떻게” 등)
    repeated = any(tok_s[i] == tok_s[i-1] for i in range(1, len(tok_s)))

    if inserted or repeated or len(tok_s) < max(1, len(tok_o)//3):
        safe = f'<prosody rate="150%" volume="medium">{_xml_escape(orig_line)}</prosody>'
        return safe, True
    return ssml_fragment, False

def resolve_polly_voice_id(polly_voice_key: str, default="Seoyeon") -> str:
    return TTS_POLLY_VOICES.get(polly_voice_key, TTS_POLLY_VOICES.get("korean_female", default))

def _looks_ssml(s: str) -> bool:
    s = (s or "").strip()
    return s.startswith("<speak") or "<prosody" in s or "<break" in s

def _pick_engine_from_ssml(ssml: str) -> str:
    # 오디오 합성과 동일 규칙: pitch 있으면 standard, 없으면 neural
    return "standard" if ' pitch="' in (ssml or "") else "neural"

def get_polly_speechmarks(text_or_ssml: str, voice_id: str,
                          types=("word",), region="ap-northeast-2"):
    """오디오 합성에 사용한 SSML/엔진과 동일 조건으로 SpeechMarks를 받아온다."""
    payload = (text_or_ssml or "").strip()
    if not payload:
        return []
    text_type = "ssml" if _looks_ssml(payload) else "text"
    engine = _pick_engine_from_ssml(payload)

    polly = boto3.client("polly", region_name=region)
    resp = polly.synthesize_speech(
        Text=payload,
        TextType=text_type,
        VoiceId=voice_id,
        OutputFormat="json",
        SpeechMarkTypes=list(types),
        Engine=engine,              # ★ 중요: 오디오와 동일 엔진
    )
    body = resp["AudioStream"].read().decode("utf-8", errors="ignore")
    return [json.loads(line) for line in body.splitlines() if line.strip()]

def resolve_polly_voice_id(polly_voice_key: str, tts_lang: str | None = "ko") -> str:
    """
    polly_voice_key: 코드에서 쓰는 키("korean_female1" 등)
    tts_lang: 'ko' | 'en' 등, 키가 없을 때의 기본값 선택에만 사용
    """
    v = TTS_POLLY_VOICES.get(polly_voice_key)
    if v:
        return v
    # 키가 없을 때 안전 폴백
    if tts_lang == "ko":
        return TTS_POLLY_VOICES.get("korean_female2", "Seoyeon")  # ko-KR
    return TTS_POLLY_VOICES.get("default_female", "Joanna")       # en-US

SUBTITLE_TEMPLATES = {
    "educational": {
        "Fontname": "NanumGothic",
        "Fontsize": 12,
        "PrimaryColour": "&H00FFFFFF",       # 흰색 텍스트
        "OutlineColour": "&H00000000",       # 검정 외곽선
        "Outline": 2,
        "Alignment": 2,
        "MarginV": 40
    },
    "entertainer": {
        "Fontname": "NanumGothic",
        "Fontsize": 12,
        "PrimaryColour": "&H00FFFFFF",       # 흰색 텍스트
        "OutlineColour": "&H00000000",       # 검정 외곽선
        "Outline": 2,
        "Alignment": 2,
        "MarginV": 40
    },
    "slow": {
        "Fontname": "NanumGothic",
        "Fontsize": 12,
        "PrimaryColour": "&H00FFFFFF",       # 흰색 텍스트
        "OutlineColour": "&H00000000",       # 검정 외곽선
        "Outline": 2,
        "Alignment": 2,
        "MarginV": 40
    },
    "default": {
        "Fontname": "NanumGothic",
        "Fontsize": 12,
        "PrimaryColour": "&H00FFFFFF",       # 흰색 텍스트
        "OutlineColour": "&H00000000",       # 검정 외곽선
        "Outline": 2,
        "Alignment": 2,
        "MarginV": 40
    },
    "korean_male": {
        "Fontname": "NanumGothic",
        "Fontsize": 12,
        "PrimaryColour": "&H00FFFFFF",       # 흰색 텍스트
        "OutlineColour": "&H00000000",       # 검정 외곽선
        "Outline": 2,
        "Alignment": 2,
        "MarginV": 40
    },
    "korean_male2": {
        "Fontname": "NanumGothic",
        "Fontsize": 12,
        "PrimaryColour": "&H00FFFFFF",       # 흰색 텍스트
        "OutlineColour": "&H00000000",       # 검정 외곽선
        "Outline": 2,
        "Alignment": 2,
        "MarginV": 40
    },
    "korean_female": {
        "Fontname": "NanumGothic",
        "Fontsize": 12,
        "PrimaryColour": "&H00FFFFFF",       # 흰색 텍스트
        "OutlineColour": "&H00000000",       # 검정 외곽선
        "Outline": 2,
        "Alignment": 2,
        "MarginV": 40
    },
    "korean_female2": {
        "Fontname": "NanumGothic",
        "Fontsize": 12,
        "PrimaryColour": "&H00FFFFFF",       # 흰색 텍스트
        "OutlineColour": "&H00000000",       # 검정 외곽선
        "Outline": 2,
        "Alignment": 2,
        "MarginV": 40
    }
}

def _looks_english(text: str) -> bool:
    # 매우 단순한 휴리스틱: 알파벳이 한글보다 확실히 많으면 영어로 간주
    letters = len(re.findall(r'[A-Za-z]', text))
    hangul = len(re.findall(r'[\uac00-\ud7a3]', text))
    return letters >= max(3, hangul * 2)

def _detect_script_language(lines):
    eng = sum(_looks_english(x) for x in lines)
    kor = sum(bool(re.search(r'[\uac00-\ud7a3]', x)) for x in lines)
    return 'en' if eng > kor else 'ko'

def _maybe_translate_lines(lines, target='ko', only_if_src_is_english=True):
    if not lines:
        return lines
    try:
        src = _detect_script_language(lines)
        if only_if_src_is_english and src != 'en':
            # 원문이 영어가 아닐 때는 건드리지 않음
            return lines
        if target is None or target == src:
            return lines
        tr = GoogleTranslator(source='auto', target=target)
        return [tr.translate(l) if l.strip() else l for l in lines]
    except Exception:
        # 번역 실패 시 원문 유지 (크래시 방지)
        return lines

def _validate_ssml(text: str) -> str:
    """
    Polly 호출 전 SSML 안전성 검사 및 보정
    - 허용 태그: speak, prosody, break
    - Neural에서 문제되는 pitch 제거
    - 속성은 쌍따옴표로 표준화
    - 연속 break/과도한 time 보정
    - 빈 prosody 제거, 태그 불일치 보정
    """
    if not text:
        return ""

    t = text.strip().replace("\ufeff", "")

    # 0) 내부에 <speak> 조각이 있으면 걷어내기 (라인 단위)
    t = re.sub(r"</?speak\s*>", "", t, flags=re.I)

    # 1) 허용 외 태그는 제거 (speak/prosody/break만 남김)
    t = re.sub(r"</?(?!prosody\b|break\b)[a-zA-Z0-9:_-]+\b[^>]*>", "", t)

    # 2) 단일인용 속성 → 쌍따옴표
    t = re.sub(r'(\b[a-zA-Z:-]+)=\'([^\']*)\'', r'\1="\2"', t)

    ## 3) pitch 속성 제거 (Neural 호환)
    #t = re.sub(r'\s+pitch="[^"]*"', "", t)

    # 4) break time 보정: 숫자ms만 허용, 2000ms 초과시 2000ms로 clamp
    def _clamp_break(m):
        val = m.group(2)
        try:
            n = int(val)
        except Exception:
            n = 200  # 이상치면 200ms로
        n = max(0, min(n, 2000))
        return f'{m.group(1)}{n}{m.group(3)}'
    t = re.sub(r'(<break\b[^>]*\btime=")(\d+)(ms"[^>]*/?>)', _clamp_break, t, flags=re.I)

    # 5) 연속 break → 하나만 남김
    t = re.sub(r'(?:<break\b[^>]*/>\s*){2,}', lambda m: re.findall(r'<break\b[^>]*/>', m.group(0))[0], t, flags=re.I)

    # 6) 빈 prosody 제거
    t = re.sub(r"<prosody[^>]*>\s*</prosody>", "", t, flags=re.I)

    # 7) prosody 닫힘 보정
    open_count = len(re.findall(r"<prosody\b", t, flags=re.I))
    close_count = len(re.findall(r"</prosody>", t, flags=re.I))
    t += "</prosody>" * max(0, open_count - close_count)

    # 8) prosody가 아예 없으면 기본 래핑 (안전 기본값)
    if "<prosody" not in t:
        t = f'<prosody rate="155%" volume="medium">{_xml_escape(t)}</prosody>'

    return t.strip()

def generate_tts_per_line(script_lines, provider, template, polly_voice_key="korean_female1"):
    audio_paths = []
    temp_audio_dir = "temp_line_audios"
    os.makedirs(temp_audio_dir, exist_ok=True)

    print(f"디버그: 총 {len(script_lines)}개의 스크립트 라인에 대해 TTS 생성 시도.")
    # Prepare per-line payloads first (validate SSML / wrap speak for Polly)
    payloads = []
    for i, line in enumerate(script_lines):
        line_audio_path = os.path.join(temp_audio_dir, f"line_{i}.mp3")
        try:
            line_ssml = _validate_ssml(line)
            ls = line_ssml.strip()
            if provider == "polly" and not ls.startswith("<speak"):
                ls = f"<speak>{ls}</speak>"
        except Exception:
            ls = f"<speak>{_xml_escape(line or '')}</speak>"
        payloads.append((i, ls, line_audio_path, line))

    # If provider is Polly, parallelize using ThreadPoolExecutor (configurable workers)
    prov = (provider or "").strip().lower()
    if prov in ("polly", "amazon polly", "aws polly", "aws_polly"):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_workers = int(os.getenv("POLLY_TTS_MAX_WORKERS", str(min(4, max(1, len(payloads))))))
        max_workers = min(max_workers, max(1, len(payloads)))

        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {}
            for (i, ls, path, orig_line) in payloads:
                fut = ex.submit(
                    generate_tts,
                    ls,
                    path,
                    "polly",
                    template,
                    None,
                    polly_voice_key
                )
                future_to_idx[fut] = (i, orig_line, path)

            for fut in as_completed(future_to_idx):
                i, orig_line, path = future_to_idx[fut]
                try:
                    res = fut.result()
                    results[i] = res
                    print(f"디버그: 라인 {i+1} ('{(orig_line or '')[:30]}...') TTS 생성 성공. 파일: {res}")
                except Exception as e:
                    print(f"오류: 라인 {i+1} ('{(orig_line or '')[:30]}...') TTS 생성 실패: {e}")

        # preserve order of successful ones
        for i, _, path, _ in sorted(payloads, key=lambda x: x[0]):
            if i in results:
                audio_paths.append(results[i])

    else:
        # Non-Polly providers (ElevenLabs etc.) keep sequential behavior
        for i, ls, path, orig_line in payloads:
            try:
                generate_tts(
                    text=ls,
                    save_path=path,
                    provider=provider,
                    template_name=template,
                    polly_voice_name_key=polly_voice_key
                )
                audio_paths.append(path)
                print(f"디버그: 라인 {i+1} ('{(orig_line or '')[:30]}...') TTS 생성 성공. 파일: {path}")
            except Exception as e:
                print(f"오류: 라인 {i+1} ('{(orig_line or '')[:30]}...') TTS 생성 실패: {e}")
            
    print(f"디버그: 최종 생성된 오디오 파일 경로 수: {len(audio_paths)}")
    if not audio_paths:
        raise RuntimeError("라인별 TTS가 0건 생성됨 (각 라인의 실패 사유는 위 로그 참조)")
    
    return audio_paths

def merge_audio_files(audio_paths, output_path):
    merged = AudioSegment.empty()
    segments = []
    current_time = 0.0

    for path in audio_paths:
        a = AudioSegment.from_file(path)
        d = a.duration_seconds
        segments.append({"start": current_time, "end": current_time + d})
        merged += a
        current_time += d

    # ✅ 끊김 방지용 꼬리 무음
    tail = AudioSegment.silent(duration=120)
    merged += tail
    if segments:
        segments[-1]["end"] += 0.12

    merged.export(output_path, format="mp3")
    return segments

def get_segments_from_audio(audio_paths, script_lines):
    segments = []
    current_time = 0
    for i, audio_path in enumerate(audio_paths):
        try:
            audio = AudioSegment.from_file(audio_path)
            duration = audio.duration_seconds
            line = script_lines[i]
            segments.append({
                "start": current_time,
                "end": current_time + duration,
                "text": line
            })
            current_time += duration
        except Exception as e:
            print(f"오류: 오디오 파일 {audio_path} 처리 중 오류 발생: {e}")
            continue
    return segments

# --- pitch 할당 함수 ---
def _assign_pitch(text: str) -> int:
    if text.endswith("?"):
        return 20     # 질문/경고
    if text.endswith("습니다") or text.endswith("합니다"):
        return -5     # 일반 설명
    if text.endswith("이다") or text.endswith("없습니다"):
        return -18    # 결론/단정
    return 0

import os, re, math, unicodedata
from typing import List, Dict

NBSP = "\u00A0"

def _ass_time(t: float) -> str:
    """float 초 -> ASS 시간 H:MM:SS.cs (centi-second, 2자리)"""
    if t < 0: t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - math.floor(t)) * 100))
    if cs == 100:
        s += 1
        cs = 0
    if s == 60:
        m += 1
        s = 0
    if m == 60:
        h += 1
        m = 0
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

ASS_NL = r"\N"
NBSP   = "\u00A0"

def _sanitize_ass_text_for_dialog(text: str) -> str:
    """
    - 역슬래시(\)는 절대 손대지 않음 (줄바꿈 ASS_NL 보존)
    - { } 는 전각으로 바꿔 override 태그 주입 방지
    - 쓸데없는 탭/다중 공백만 정리
    """
    import re
    t = (text or "").replace("\r", "")
    t = t.replace("{", "｛").replace("}", "｝")
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip() or NBSP

def _best_two_line_break(text: str, max_len: int, min_each: int = 3) -> str:
    raw = text
    # 줄바꿈/제어문자 제거(라인 계산용)
    plain = raw.replace(r"\N", " ").replace("\n", " ")

    # 이미 1줄로 충분하면 그대로
    if len(plain) <= max_len:
        return raw

    # NBSP는 non-break로 취급하므로, 후보 탐색 시 제외
    # 이상적인 분기점: 대략 절반 근처
    tgt = max_len  # 2줄 목표라면 대략 첫 줄 max_len 부근이 자연스러움

    # 후보 수집(우선순위 그룹)
    def _cands(chars):
        idx = []
        for m in re.finditer(chars, plain):
            i = m.start()
            # NBSP 근처 금지: '단어 묶음 보호'
            if i > 0 and plain[i-1] == NBSP: 
                continue
            if i < len(plain)-1 and plain[i+1] == NBSP:
                continue
            idx.append(i)
        return idx

    spaces = _cands(r" ")
    mild   = _cands(r"[,·:\-\/]")
    strong = _cands(r"[?？]")

    groups = [spaces, mild, strong]
    best = None
    best_score = 10**9

    for cand_group, weight in zip(groups, [0, 1, 2]):
        for i in cand_group:
            left  = plain[:i].rstrip()
            right = plain[i+1:].lstrip()
            if len(left) < min_each or len(right) < min_each:
                continue
            if len(left) > max_len or len(right) > max_len:
                continue
            # 스코어: 목표점과 근접 + 그룹 가중치(공백이 가장 선호)
            score = abs(len(left) - tgt) * 2 + weight * 10
            if score < best_score:
                best_score, best = (score, i)

        if best is not None:
            break  # 더 좋은 그룹 탐색 전 종료(공백에서 성공 시 고정)

    if best is None:
        # 폴백: max_len에 가장 가까운 공백 또는 안전 위치
        cut = None
        for i in range(min_each, len(plain) - min_each):
            if plain[i] == " " and len(plain[:i]) <= max_len:
                cut = i
        if cut is None:
            cut = min(max_len, len(plain)-min_each)
        left, right = plain[:cut].rstrip(), plain[cut:].lstrip()
        return f"{left}\\N{right}"

    left, right = plain[:best].rstrip(), plain[best+1:].lstrip()
    return f"{left}\\N{right}"

def _prepare_text_for_lines(text: str, max_chars_per_line: int, max_lines: int) -> str:
    if not text:
        return NBSP

    # 이미 \N이 있으면(사용자/상위 로직에서 강제 분해) 그대로 둠
    if r"\N" in text:
        return text

    # 1줄로 충분하면 그대로
    if len(text) <= max_chars_per_line or max_lines <= 1:
        return text

    # 2줄 허용: 좋은 지점에서만 분리
    text2 = _best_two_line_break(text, max_chars_per_line, min_each=max(3, int(max_chars_per_line*0.3)))
    # 혹시라도 결과가 과도하게 길면 마지막 폴백(하드 컷)
    if any(len(x) > max_chars_per_line for x in text2.split(r"\N")) and max_lines >= 2:
        raw = text.replace(r"\N", " ")
        cut = max_chars_per_line
        text2 = raw[:cut].rstrip() + r"\N" + raw[cut:].lstrip()

    # 라인 개수 제한
    parts = text2.split(r"\N")
    if len(parts) > max_lines:
        text2 = r"\N".join(parts[:max_lines-1] + [" ".join(parts[max_lines-1:])])

    # 비어있는 라인이 생기지 않게 NBSP 보강
    fixed = []
    for p in text2.split(r"\N"):
        pp = p if p.strip().replace(NBSP, "") else NBSP
        fixed.append(pp)
    return r"\N".join(fixed)

def _strip_trailing_punct_last_line(text: str) -> str:
    """
    마지막 줄 끝의 '보이는 마침표/여분 공백'만 살짝 제거(렌더 안정성).
    물음표/느낌표/종결어미 등은 보존.
    """
    if not text:
        return text
    lines = text.split(r"\N")
    last = lines[-1]
    # 완전 공란 보호
    if not last.strip().replace(NBSP, ""):
        last = NBSP
    # 아주 약한 마침표/중복 공백만 정리
    last = re.sub(r"[ \t]+$", "", last)
    last = re.sub(r"[.]{2,}$", ".", last)   # "..." -> "."
    # 한 글자짜리 줄 방지용 NBSP
    if len(last.strip()) == 0:
        last = NBSP
    lines[-1] = last
    return r"\N".join(lines)

ASS_NL = r"\N"

# --- 템플릿 섹션 로더(여러분 파일에 이미 있으면 그걸 쓰세요) ---
def _resolve_template_blocks(template_name: str):
    script_info = [
        "[Script Info]","ScriptType: v4.00+","WrapStyle: 2",
        "ScaledBorderAndShadow: yes","PlayResX: 720","PlayResY: 1080",""
    ]
    styles = [
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default, Arial, 50, &H00FFFFFF, &H000000FF, &H00000000, &H64000000, "
        "-1, 0, 0, 0, 100, 100, 0, 0, 1, 2, 0, 2, 30, 30, 40, 1",
        ""
    ]
    events_header = ["[Events]","Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    try:
        if 'SUBTITLE_TEMPLATES' in globals() and template_name in SUBTITLE_TEMPLATES:
            tpl = SUBTITLE_TEMPLATES[template_name]
            if isinstance(tpl, dict):
                if "script_info" in tpl:
                    si = tpl["script_info"]
                    if isinstance(si, str):  script_info = ["[Script Info]"]+[ln for ln in si.splitlines() if ln.strip()]+[""]
                    elif isinstance(si, (list,tuple)): script_info = ["[Script Info]"]+[str(x) for x in si if str(x).strip()]+[""]
                if "styles" in tpl:
                    st = tpl["styles"]
                    if isinstance(st, str):  styles = ["[V4+ Styles]"]+[ln for ln in st.splitlines() if ln.strip()]+[""]
                    elif isinstance(st, (list,tuple)): styles = ["[V4+ Styles]"]+[str(x) for x in st if str(x).strip()]+[""]
                if "events_header" in tpl:
                    eh = tpl["events_header"]
                    if isinstance(eh, str):  events_header = [ln for ln in eh.splitlines() if ln.strip()]
                    elif isinstance(eh, (list,tuple)): events_header = [str(x) for x in eh if str(x).strip()]
    except Exception:
        pass
    return script_info, styles, events_header

# --- 여기서부터: BMJUA 고정 부분 ---
FORCE_FONT_FAMILY = "BMJUA_ttf"                     # 폰트 패밀리명(파일명과 맞춰 고정)
FORCE_FONT_DIR    = os.path.join("assets","fonts")  # 실제 파일 위치: assets/fonts/BMJUA_ttf.ttf

def _force_font_in_styles(styles_lines, family: str = FORCE_FONT_FAMILY):
    out = []
    fmt_fields = None
    for ln in styles_lines:
        s = ln.strip()
        if s.startswith("Format:"):
            # 필드 인덱스 파악(보통 Name, Fontname, Fontsize, ...)
            fmt_fields = [x.strip() for x in ln.split(":",1)[1].split(",")]
            out.append(ln); continue
        if s.startswith("Style:"):
            try:
                prefix, rest = ln.split(":", 1)
                vals = [v.strip() for v in rest.split(",")]
                idx = 1  # 기본: 두 번째가 Fontname
                if fmt_fields and "Fontname" in fmt_fields:
                    idx = fmt_fields.index("Fontname")
                if idx < len(vals):
                    vals[idx] = family
                ln = f"{prefix}: {', '.join(vals)}"
            except Exception:
                pass
        out.append(ln)
    return out

def _first_style_name(styles_lines, default="Default"):
    for ln in styles_lines:
        m = re.match(r"\s*Style:\s*([^,]+)\s*,", ln)
        if m:
            return m.group(1).strip()
    return default

# 파일 상단 어딘가(전역)에 고정 상수 추가
ASS_FONT_FILE = os.path.abspath(os.path.join("assets", "fonts", "BMJUA_ttf.ttf"))
ASS_FONT_FAMILY = "BM JUA"  # BMJUA_ttf.ttf의 내부 패밀리명(일반적으로 이 이름)

def _ensure_styles_with_bmjua(styles_block_lines: list[str]) -> list[str]:
    """
    템플릿의 [V4+ Styles] 블록과 무관하게, BM JUA 전용 스타일을 추가해둡니다.
    이벤트는 이 스타일 이름을 사용합니다.
    """
    out = []
    seen_header = False
    for ln in styles_block_lines:
        out.append(ln)
        if ln.strip().lower().startswith("format:"):
            seen_header = True
    if not seen_header:
        # 혹시 템플릿이 형식을 안썼으면 안전 기본 헤더 추가
        out = [
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ] + out

    # ★ BMJUA 전용 스타일을 'Style: BMJua'로 추가
    #  - Fontsize/Outline/MarginV는 원래 쓰시던 값 범위에 맞춰 적당히; 필요시 조정
    out.append(
        f"Style: BMJua, {ASS_FONT_FAMILY}, 58, &H00FFFFFF, &H000000FF, &H00000000, &H64000000, " #글자크기 58로 수정
        f"-1, 0, 0, 0, 100, 100, 0, 0, 1, 4, 0, 2, 30, 30, 60, 1"
    )
    out.append("")  # 마지막 공백 줄
    return out

def generate_ass_subtitle(
    segments,
    ass_path: str,
    template_name: str = "educational",
    strip_trailing_punct_last: bool = True,
    max_chars_per_line: int | None = None,
    max_lines: int | None = None,
    wrap_mode: str = "preserve"  # "preserve" | "smart"
) -> str:
    """
    wrap_mode:
      - "preserve": 텍스트를 절대 래핑하지 않고 한 줄 그대로 기록 (추천)
      - "smart": 길이 기준 2줄 분할 등 기존 동작 유지
    """
    if not segments:
        segments = [{"start": 0.00, "end": 0.02, "text": NBSP}]

    script_info, styles, events_header = _resolve_template_blocks(template_name)
    styles = _ensure_styles_with_bmjua(styles)

    lines = []
    for ev in segments:
        s = float(ev.get("start", 0.0))
        e = float(ev.get("end", s + 0.02))
        if e <= s:
            e = s + 0.02
        if ev is segments[-1]:
            e = max(e, s + 0.35)

        raw_text = (ev.get("text") or "")

        def _line_clean(one: str) -> str:
            t = one or ""
            if strip_trailing_punct_last:
                t = _strip_last_punct_preserve_closers(t)
            t = _drop_special_keep_units(t)  # '?', %, °, ℃, °C, °F, km/h, ㎦ 등 단위 보존
            t = _sanitize_ass_text_for_dialog(t)
            return t

        if wrap_mode == "preserve":
            # 🚫 래핑 없음: 원문 그대로(비어있으면 NBSP)
            plan_text = _line_clean(raw_text).strip() or NBSP
        else:
            # ✅ 14자 기준: 1줄이면 락, 넘으면 2줄 강제
            normalized = _line_clean(raw_text)
            maxc = max_chars_per_line or 14
            maxl = max_lines or 2

            if _visible_len(normalized) <= maxc:
                # 1줄 고정: 공백을 NBSP로 잠가 자동 줄바꿈 방지
                plan_text = lock_oneliner_if_short(normalized, threshold=maxc)
            else:
                # 2줄 강제 래핑
                plan_text = _prepare_text_for_lines(normalized, maxc, maxl)

        # ③ pitch → 색상
        col_hex = _pitch_to_hex(ev.get("pitch"))
        if col_hex:
            plan_text = "{\\c" + col_hex + "}" + plan_text

        dlg = f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},BMJua,,0,0,0,,{plan_text}"
        lines.append(dlg)

    os.makedirs(os.path.dirname(ass_path) or ".", exist_ok=True)
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(script_info) + "\n")
        f.write("\n".join(styles) + "\n")
        f.write("\n".join(events_header) + "\n")
        f.write("\n".join(lines) + "\n")

    return ass_path

def format_ass_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h:01}:{m:02}:{s:02}.{cs:02}"

def _drop_special_keep_units(s: str) -> str:
    """
    자막 텍스트에서 장식성 특수문자를 지우되,
    숫자/단위/질의부호는 보존:
    - 보존: 한글/영문/숫자/공백/NBSP, 물음표(?),
            %, °, ℃, ℉, '°C','°F', 슬래시(/),
            CJK 호환 단위(㎞, ㎦, ㎥ 등 U+3300~U+33FF)
    - 소수점은 '숫자.숫자' 안에서만 보존
    - 그 외 기호(따옴표, 마침표, 콜론, 세미콜론 등) 제거
    """
    NBSP = "\u00A0"
    if not s:
        return s

    # 소수점 보호
    s = re.sub(r'(?<=\d)\.(?=\d)', '§DECIMAL§', s)

    # '°C' / '°F'는 그대로 둠
    s = s.replace("°C", "°C").replace("°F", "°F")

    # 허용 문자 외 제거
    allowed = r"[A-Za-z\uAC00-\uD7A30-9 \t" + NBSP + r"\?%°/℃℉\u3300-\u33FF]"
    s = re.sub(rf"[^{allowed}]+", "", s)

    # 보호 소수점 복원
    s = s.replace('§DECIMAL§', '.')

    # 콤마는 숫자 사이가 아니면 제거(원하시면 전부 제거해도 무방)
    s = re.sub(r'(?<!\d),(?!\d)', '', s)

    # 앞뒤 공백 정리
    return re.sub(r"\s{2,}", " ", s).strip()

def split_script_to_lines(script_text: str, mode="llm") -> list[str]:
    import os
    text = (script_text or "").strip()
    if not text:
        return []

    # ✨ 개행이 있으면 우선 하드 경계로 쪼갬
    if "\n" in text:
        base_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    else:
        base_lines = [text]

    if mode == "newline":
        return base_lines

    # mode == "llm": 기본적으로 LLM(breath_linebreaks) 호출을 비활성화합니다.
    # 환경변수 USE_SSML_LLM=1 으로 원래 LLM 기반 분절을 복원할 수 있습니다.
    use_llm = os.getenv("USE_SSML_LLM", "0") == "1"
    out = []
    for line in base_lines:
        if use_llm:
            try:
                out.extend(breath_linebreaks(line, honor_newlines=True))
            except Exception:
                out.append(line)
        else:
            # 결정적 폴백: 라인을 그대로 사용 (순차 LLM 호출 회피)
            out.append(line)
    return out

# --- 변경 2: generate_subtitle_from_script 시그니처/로직 확장 ---
def generate_subtitle_from_script(
    script_text: str,
    ass_path: str,
    full_audio_file_path: str,
    provider: str = "polly",
    template: str = "default",
    polly_voice_key: str = "default_male",
    subtitle_lang: str = "ko",
    translate_only_if_english: bool = False,
    tts_lang: str | None = None,
    split_mode = "llm",
    strip_trailing_punct_last: bool = True,
):
    """
    목적: '라인 단위 세그먼트(base)'만 반환하고, 각 세그먼트에 SSML을 실어 메인에서 densify 하도록 한다.
    - 여기서는 ASS 생성/자막 쪼개기/병합을 하지 않는다.
    - 메인에서 auto_densify_for_subs(...)가 SSML( rate/pitch/break )을 읽어 SpeechMarks 기반으로 정확히 쪼갤 수 있게 함.
    반환: (segments_base, audio_clips, ass_path)
    """

    # --- 0) 보조
    def _strip_punct_and_quotes(s: str) -> str:
        if not s: return ""
        s = s.translate(str.maketrans({
            "“": "", "”": "", "„": "", "‟": "", '"': "",
            "‘": "", "’": "", "‚": "", "‛": "", "'": "",
            "！": "!", "？": "?"
        }))
        s = re.sub(r'\s{2,}', ' ', s)
        return s.strip()

    # --- 1) 스크립트 → 라인
    base_lines = split_script_to_lines(script_text or "", mode=split_mode)
    base_lines = [ln for ln in base_lines if ln.strip()]
    if not base_lines:
        return [], None, ass_path

    # SSML 태그 제거한 클린 텍스트(SSML 생성을 위해)
    clean_lines = split_script_to_lines(script_text, mode=split_mode)

    # --- 2) Polly 보이스/언어 정합
    if provider.lower() == "polly":
        if tts_lang == "en" and polly_voice_key.startswith("korean_"):
            polly_voice_key = "default_male"
        elif tts_lang == "ko" and polly_voice_key.startswith("default_"):
            polly_voice_key = "korean_female1"

    # --- 3) 라인별 SSML 생성(Polly) 또는 원문(타 공급자)
    prov = provider.lower()
    tts_lines = []
    ssml_meta_lines = []          # ★ 모든 공급자 공통: 메타 파싱용 SSML

    for ln in clean_lines:
        ln_for_ssml = koreanize_if_english(ln)   # ★ 영문이면 의미 동일 한국어로
        # 성능 최적화: 기본적으로 OpenAI/LLM 기반 SSML 변환을 비활성화합니다.
        # 필요 시 환경변수 USE_SSML_LLM=1 으로 복원하세요.
        import os
        use_llm = os.getenv("USE_SSML_LLM", "0") == "1"
        if use_llm:
            try:
                frag = convert_line_to_ssml(ln_for_ssml)  # <prosody>...</prosody> (+ <break/>)
            except Exception:
                frag = f'<prosody rate="150%" volume="medium">{_xml_escape(ln_for_ssml)}</prosody>'
        else:
            # 결정적 폴백: 간단한 prosody 래핑만으로 Polly/ElevenLabs에 텍스트 전송
            frag = f'<prosody rate="155%" volume="medium">{_xml_escape(ln_for_ssml)}</prosody>'

        safe = _validate_ssml(frag)
        if not safe.strip().startswith("<speak"):
            safe = f"<speak>{safe}</speak>"
        ssml_meta_lines.append(safe)

    # TTS에 넘길 라인: Polly면 SSML, 아니면 평문
    if prov == "polly":
        tts_lines = ssml_meta_lines[:]
    else:
        tts_lines = clean_lines[:]

    # --- 4) 라인별 TTS 생성 → 병합
    audio_paths = generate_tts_per_line(
        tts_lines, provider=provider, template=template, polly_voice_key=polly_voice_key
    )
    if not audio_paths:
        return [], None, ass_path

    segments_raw = merge_audio_files(audio_paths, full_audio_file_path)
    # segments_raw: [{"start":..., "end":...}, ...]

    # --- 5) ★★★ 라인 단위 'base 세그먼트' 구성: SSML을 심는다
    # text: 원문(또는 표시용 자막 기본값), ssml: Polly에 실제 보낸 SSML
    if len(clean_lines) != len(base_lines):
        try:
            import streamlit as st
            st.warning(f"라인 불일치: base={len(base_lines)} vs clean={len(clean_lines)} → base 기준으로 강제 정렬")
        except Exception:
            print(f"[warn] 라인 불일치: base={len(base_lines)} vs clean={len(clean_lines)}")
        clean_lines = base_lines[:]
        ssml_meta_lines = ssml_meta_lines[:len(base_lines)]

    # 오디오 병합 후, 세그먼트/텍스트/SSML 최소길이로 동기화
    segments_raw = merge_audio_files(audio_paths, full_audio_file_path)
    n = min(len(segments_raw), len(base_lines), len(ssml_meta_lines))
    if n != len(segments_raw):
        try:
            st.warning(f"TTS 세그먼트 수 불일치: audio={len(segments_raw)} vs base={len(base_lines)} → min={n}로 맞춤")
        except Exception:
            print(f"[warn] 세그먼트 수 불일치: audio={len(segments_raw)} base={len(base_lines)} → {n}")

    segments_base = []
    for i in range(n):
        s = segments_raw[i]
        line_text = base_lines[i]
        ssml_meta = ssml_meta_lines[i]
        pitch_sum = _summarize_line_pitch(ssml_meta)
        segments_base.append({
            "start": float(s["start"]),
            "end":   float(s["end"]),
            "text":  re.sub(r"\s+", " ", line_text).strip(),
            "ssml":  ssml_meta,
            "pitch": pitch_sum,
        })


    # --- 6) 여기서는 ASS/자막 분해를 하지 않는다(메인에서 처리)
    # generate_ass_subtitle(...) 호출 금지

    # audio_clips는 이 함수 내부에서 열었다가 닫지 않으니 None 반환(기존 관례 유지)
    return segments_base, None, ass_path

# === Auto-paced subtitle densifier (자연스러운 문맥 분할 우선) ===
def _auto_split_for_tempo(text: str, tempo: str = "fast"):
    """
    tempo: fast(짧게) | medium | slow
    분할 우선순위:
    1) 담화 표지(그리고/하지만/그래서/그런데/그러니까/즉/특히/반면에/게다가/또는/혹은 등) '뒤'에서 끊기
    2) 연결 어미(고/지만/는데/면서/라면/면/니까/다가/으며/며 등) '뒤'에서 끊기
    3) 쉼표/세미콜론/중점 등 구두점 뒤에서 끊기
    4) 여전히 길면 공백 근처로 길이 기반 분할
    이후 너무 짧은 조각은 이웃과 병합
    """
    import re

    # 길이 목표(상황 맞게 조절)
    max_len_map = {"fast": 12, "medium": 18, "slow": 24}
    max_len = max_len_map.get(tempo, 18)
    min_piece_chars = 2  # 너무 짧은 조각 병합 기준

    t = (text or "").strip()
    if not t:
        return []

    # 1) 담화 표지 후 분할(토큰은 앞 조각에 둠)
    discourse = r"(그리고|하지만|근데|그런데|그래서|그러니까|즉|특히|게다가|한편|반면에|또는|혹은|다만)"
    t = re.sub(rf"\b{discourse}\b\s*", r"\g<0>§", t)

    # 2) 연결 어미 후 분할(어미는 앞 조각에 둠)
    eomi = r"(고|지만|는데요?|면서|며|라면|면|니까|다가|으며|거나|든지)"
    t = re.sub(rf"({eomi})(?=\s|\Z)", r"\1§", t)

    # 3) 쉼표·세미콜론·중점 뒤 분할 (구두점은 앞 조각에)
    t = re.sub(r"(?<=[,，、;:·])\s*", "§", t)

    # 일차 분할
    parts = [p.strip() for p in t.split("§") if p.strip()]
    chunks = []

    # 4) 길이 보정: 너무 긴 건 공백 근처로 추가 분할
    for p in parts:
        if len(p) <= max_len:
            chunks.append(p)
            continue

        cur = p
        while len(cur) > max_len:
            window = cur[: max_len + 6]  # 여유
            spaces = [m.start() for m in re.finditer(r"\s", window)]
            split_pos = spaces[-1] if spaces else max_len
            chunks.append(cur[:split_pos].strip())
            cur = cur[split_pos:].strip()
        if cur:
            chunks.append(cur)

    # 5) 너무 짧은 조각 병합(양 옆과 자연스러운 공백 처리)
    def _wordish(ch: str) -> bool:
        return ch.isalnum() or ('\uAC00' <= ch <= '\uD7A3')  # 영문/숫자/한글
    i = 1
    while i < len(chunks):
        if len(chunks[i]) < min_piece_chars:
            prev = chunks[i-1].rstrip()
            cur  = chunks[i].lstrip()
            sep = " " if (prev and cur and _wordish(prev[-1]) and _wordish(cur[0])) else ""
            chunks[i-1] = (prev + sep + cur).strip()
            chunks.pop(i)
        else:
            i += 1

    return chunks

def auto_densify_for_subs(
    segments,
    tempo: str = "fast",
    words_per_piece: int = 3,
    min_tail_words: int = 2,
    chunk_strategy: str | None = None,
    marks_voice_key: str | None = None,
    max_chars_per_piece: int = 16,
    min_piece_dur: float = 0.42,
):
    """
    라인(=TTS 한 번) 기준 세그먼트를 '빠른 템포'로 잘게 쪼갭니다.
    - 각 입력 세그먼트는 {"start","end","text","ssml"} 구조라고 가정
    - 말꼬리/종결어미를 보호하고, 너무 짧은 꼬리는 앞 조각에 붙입니다.
    - pieces의 시간은 원 세그먼트 구간 내에서만 배분되며, 다음 세그먼트와 겹치지 않습니다.
    - (선택) max_chars_per_piece로 조각 길이를 하드캡합니다(한국어 12~18 추천).
    """
    out = []
    if not segments:
        return out

    # 템포별 기본 파라미터
    if tempo == "fast":
        base_words = max(1, min(5, words_per_piece))
        min_dur = float(min_piece_dur)
    elif tempo == "normal":
        base_words = max(2, min(6, words_per_piece + 1))
        min_dur = max(0.5, float(min_piece_dur))
    else:
        base_words = max(3, min(7, words_per_piece + 2))
        min_dur = max(0.6, float(min_piece_dur))

    # 종결/말꼬리 보호
    END_STRONG_RE = re.compile(r'(?:\?|…|이다|다|요|죠|니다|습니다|입니다|예요|이에요|였(?:다|습니다)|겠(?:다|죠)|맞(?:죠|다))$')

    def _tokenize_for_chunks(text: str):
        # 한국어/영어/숫자/부호 분리 (공백 포함 유지)
        toks = re.findall(r'[\uAC00-\uD7A3A-Za-z0-9]+|[^\s]', text or "")
        # 공백 복원
        merged, prev_is_word = [], False
        for t in toks:
            if re.match(r'^\s+$', t):
                merged.append(t)
                prev_is_word = False
            elif re.match(r'^[\uAC00-\uD7A3A-Za-z0-9]+$', t):
                # 단어
                if merged and not re.match(r'^\s+$', merged[-1]):
                    merged.append(' ')
                merged.append(t)
                prev_is_word = True
            else:
                # 구두점/기호는 붙여쓰기
                merged.append(t)
                prev_is_word = False
        s = ''.join(merged).strip()
        parts = re.findall(r'\S+|\s+', s)
        return parts

    def _join_parts(parts):
        return ''.join(parts).strip()

    for seg in segments:
        s0 = float(seg["start"])
        e0 = float(seg["end"])
        t  = (seg.get("text") or "").strip()
        if not t:
            continue
        parts = _tokenize_for_chunks(t)

        # 길이 기준으로 조각내기
        pieces = []
        cur, cur_chars, cur_words = [], 0, 0
        def _flush():
            nonlocal cur, cur_chars, cur_words
            if not cur:
                return
            txt = _join_parts(cur)
            pieces.append(txt)
            cur, cur_chars, cur_words = [], 0, 0

        for p in parts:
            if p.isspace():
                cur.append(p)
                cur_chars += len(p)
                continue
            cur.append(p)
            cur_chars += len(p)
            if re.match(r'^[\uAC00-\uD7A3A-Za-z0-9]+$', p):
                cur_words += 1
            # 하드캡: 글자수 초과 또는 단어수 초과일 때 끊기
            if cur_chars >= max_chars_per_piece or cur_words >= base_words:
                # 종결 꼬리 보호는 harden 단계에서 추가로 처리
                _flush()

        _flush()
        # 병합 규칙: 아주 짧은 꼬리(<=4자) 단독이면 앞에 합침
        merged = []
        for s in pieces:
            if merged and len(s) <= 4 and not END_STRONG_RE.search(merged[-1]):
                merged[-1] = (merged[-1] + ' ' + s).strip()
            else:
                merged.append(s)

        # 시간 배분 (글자 비율)
        total_chars = sum(len(x) for x in merged) or 1
        dur_total   = max(0.01, e0 - s0)
        t_cursor    = s0
        for i, txt in enumerate(merged):
            if i == len(merged) - 1:
                t1 = e0
            else:
                frac = max(1, len(txt)) / total_chars
                t1   = t_cursor + dur_total * frac
            # 최소 표시 시간 확보 (다음 세그먼트 침범 금지)
            if (t1 - t_cursor) < min_dur:
                t1 = min(e0, t_cursor + min_dur)
            out.append({"start": round(t_cursor, 3), "end": round(t1, 3), "text": txt})
            t_cursor = t1

    return out

def _strip_last_punct_preserve_closers(s: str) -> str:
    # 끝이 [. , ! ? …] 이고 그 뒤에는 공백/닫는 따옴표/괄호만 오는 경우 그 구두점만 제거
    return re.sub(r'([.,!?…])(?=\s*(?:["\'”’)\]\}]|$))', '', s.strip())

# --- helper: 단어 단위 마이크로 분할 ---
def _micro_split_by_words(piece: str, target_words: int = 3, min_tail_words: int = 2):
    # 공백 기준 토큰화(토큰에 붙은 쉼표 등은 그대로 유지 → 중간의 구두점은 살림)
    tokens = re.findall(r'\S+', piece.strip())
    if not tokens:
        return []
    chunks = [" ".join(tokens[i:i+target_words]).strip()
              for i in range(0, len(tokens), target_words)]
    # 마지막 덩어리가 너무 짧으면 앞 덩어리로 합치기
    if len(chunks) >= 2 and len(chunks[-1].split()) < min_tail_words:
        chunks[-2] = (chunks[-2] + " " + chunks[-1]).strip()
        chunks.pop()
    return chunks

def _sentence_split_by_dot(text: str):
    """'.' 뒤에서 문장 분리(공백 무시). 마침표가 없다면 전체를 한 문장으로."""
    if not text or not text.strip():
        return []
    parts = re.split(r'(?<=\.)\s*', text.strip())
    # 빈 조각 제거 + 원문 유지
    return [p.strip() for p in parts if p and p.strip()]

def _split_tokens_into_n(tokens, n, prefer_punct=True):
    """
    토큰 리스트를 n개로 균형 있게 자름.
    prefer_punct=True면 경계 근처에서 쉼표/세미콜론 등 뒤를 우선 경계로 선택.
    """
    if n <= 1 or len(tokens) <= n:
        return [" ".join(tokens).strip()]

    desired = [round(len(tokens) * i / n) for i in range(1, n)]
    boundaries = []
    for idx in desired:
        # 자르기 좋은 근처 후보 인덱스(토큰 사이 경계)
        cand = list(range(max(1, idx - 2), min(len(tokens) - 1, idx + 2) + 1))
        pick = None
        if prefer_punct:
            for j in cand:
                if re.search(r'[,:;·…]$', tokens[j - 1]):
                    pick = j
                    break
        if pick is None:
            pick = min(cand, key=lambda j: abs(j - idx)) if cand else idx
        # 단조 증가 보장
        if boundaries and pick <= boundaries[-1]:
            pick = boundaries[-1] + 1
        pick = min(max(1, pick), len(tokens) - 1)
        boundaries.append(pick)

    # 경계로 분할
    out = []
    start = 0
    for b in boundaries + [len(tokens)]:
        out.append(" ".join(tokens[start:b]).strip())
        start = b
    return [x for x in out if x]


def _smooth_chunks_by_flow(pieces, target_words=3, min_words=2, max_words=5):
    """
    pieces: 문자열 리스트(이미 조각난 자막)
    - 1단어/너무짧은 조각은 앞/뒤와 합침
    - 담화표지(그리고/하지만/근데/그런데/그러니까/또/또한/게다가/한편/반면에/즉)는 뒤 조각에 붙이는 걸 우선
    - 너무 길어진 조각은 단어 경계 + 구두점 뒤를 선호해 다시 나눔
    """
    discourse_heads = {"그리고","하지만","근데","그런데","그러니까","또","또한","게다가","한편","반면에","즉"}

    # 1) 토큰화
    toks = [re.findall(r"\S+", p.strip()) for p in pieces if p.strip()]
    if not toks:
        return []

    # 2) 1단어/짧은 조각 병합
    i = 0
    while i < len(toks):
        wc = len(toks[i])
        if wc >= min_words or len(toks) == 1:
            i += 1
            continue

        first_tok = toks[i][0] if toks[i] else ""
        # 담화표지만 단독이면 다음과 합치기 우선
        if first_tok in discourse_heads and i + 1 < len(toks):
            toks[i + 1] = toks[i] + toks[i + 1]
            toks.pop(i)
            continue

        # 그 외: 앞 조각이 목표보다 짧으면 앞과 합치기, 아니면 뒤와 합치기
        prev_ok = (i > 0 and len(toks[i - 1]) < target_words)
        if prev_ok:
            toks[i - 1] = toks[i - 1] + toks[i]
            toks.pop(i)
        elif i + 1 < len(toks):
            toks[i] = toks[i] + toks[i + 1]
            toks.pop(i + 1)
        else:
            i += 1  # 마지막 하나 남은 예외
    # 3) 너무 긴 조각은 자연스럽게 재분할(구두점 선호)
    refined = []
    for tt in toks:
        wc = len(tt)
        if wc > max_words:
            n = max(2, round(wc / target_words))
            refined.extend(_split_tokens_into_n(tt, n, prefer_punct=True))
        else:
            refined.append(" ".join(tt).strip())

    # 4) 각 조각 끝의 꼬리 구두점 정리(따옴표/괄호 보존)
    refined = [_strip_last_punct_preserve_closers(x) for x in refined if x.strip()]
    return refined