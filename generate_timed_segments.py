# generate_timed_segments.py
from __future__ import annotations
import os
import re
import math
from typing import List, Dict, Tuple, Optional

from moviepy.editor import AudioFileClip

# LLM 호출은 ssml_converter에만 존재
from ssml_converter import breath_linebreaks_batch, convert_lines_to_ssml_batch

# Polly 전용 합성/믹스
from polly_tts import synthesize_lines_polly, mix_polly

NBSP = "\u00A0"

SUBTITLE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "educational": {
        "Style": (
            "Style: Default,Pretendard-Bold,56,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            "-1,0,0,0,100,100,0,0,1,2,2,10,10,40,1"
        ),
        "Pos": r"{\an2}",
    },
    "center": {
        "Style": (
            "Style: Default,Pretendard-Bold,64,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            "-1,0,0,0,100,100,0,0,5,2,2,10,10,40,1"
        ),
        "Pos": r"{\an5}",
    },
}

# -------- 호환 유틸 (외부 참조 가능성) --------
def dedupe_adjacent_texts(lines: List[str]) -> List[str]:
    out, prev = [], None
    for s in lines:
        if s and s != prev:
            out.append(s)
        prev = s
    return out

def _auto_split_for_tempo(lines: List[str], max_chars: int = 9999) -> List[str]:
    return list(lines or [])

# -------- ASS 유틸 --------
def _drop_special_except_q(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9\uAC00-\uD7A3\s?]", "", s or "")

def sanitize_ass_text(s: str) -> str:
    s = (s or "").replace("\n", " ").replace("\r", " ").strip()
    s = _drop_special_except_q(s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s or NBSP

def prepare_text_for_ass(text: str, one_line_threshold: int = 14, biline_target: int = 16) -> str:
    t = sanitize_ass_text(text)
    if len(t) <= one_line_threshold:
        return t
    words = t.split()
    left = []
    while words and len(" ".join(left + [words[0]])) <= biline_target:
        left.append(words.pop(0))
    right = " ".join(words).strip()
    if not left or not right:
        mid = max(1, len(t)//2)
        return t[:mid] + r"\N" + t[mid:]
    return " ".join(left) + r"\N" + right

def _fmt_time(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600); t -= h * 3600
    m = int(t // 60);   t -= m * 60
    s = int(t)
    cs = int(round((t - s) * 100))
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

def _ass_header(style_line: str) -> str:
    return f"""[Script Info]
ScriptType: v4.00+
Collisions: Normal
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{style_line}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

# -------- SSML pitch → 색상 --------
def _extract_pitch_value(ssml_line: str) -> Optional[float]:
    if not ssml_line:
        return None
    m = re.search(r'pitch="\s*([+-]?\d+)\s*st"', ssml_line, flags=re.I)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m = re.search(r'pitch="\s*([+-]?\d+)\s*%[^"]*"', ssml_line, flags=re.I)
    if m:
        try:
            p = float(m.group(1))
            return 12.0 * math.log2(1.0 + (p / 100.0))
        except Exception:
            pass
    return None

def _pitch_to_hex(pitch_st: Optional[float]) -> Optional[str]:
    if pitch_st is None:
        return None
    if pitch_st >= 8:   return "33CCFF"
    if pitch_st >= 6:   return "55E0FF"
    if pitch_st >= 4:   return "77F5FF"
    if pitch_st <= -8:  return "7777FF"
    if pitch_st <= -6:  return "8890FF"
    if pitch_st <= -4:  return "99AAFF"
    return None

# -------- ASS 생성 --------
def generate_ass_subtitle(
    segments: List[Dict],
    ass_path: str,
    template_name: str = "educational",
    strip_trailing_punct_last: bool = True,
    max_chars_per_line: int = 14,
    max_lines: int = 2,
) -> str:
    tmpl = SUBTITLE_TEMPLATES.get(template_name, SUBTITLE_TEMPLATES["educational"])
    style_line = tmpl["Style"]
    anchor = tmpl["Pos"]
    os.makedirs(os.path.dirname(ass_path) or ".", exist_ok=True)

    out_lines = [_ass_header(style_line)]

    for i, seg in enumerate(segments):
        s = _fmt_time(seg.get("start", 0.0))
        e = _fmt_time(seg.get("end", 0.0))
        raw = (seg.get("text") or "").strip()

        if strip_trailing_punct_last and i == len(segments) - 1:
            raw = re.sub(r"[.!…]+$", "", raw).strip()

        txt = sanitize_ass_text(raw)
        if max_lines == 2 and len(txt) > max_chars_per_line:
            txt = prepare_text_for_ass(txt, one_line_threshold=max_chars_per_line, biline_target=max_chars_per_line + 2)

        ssml_line = seg.get("ssml") or ""
        pitch_st = _extract_pitch_value(ssml_line)
        hexcol = _pitch_to_hex(pitch_st)
        color_tag = r"{\c&H" + hexcol + "&}" if hexcol else ""

        content = anchor + color_tag + txt
        out_lines.append(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{content}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    return ass_path

# -------- 메인 파이프라인 (LLM 호출 없음) --------
def generate_subtitle_from_script(
    script_text: str,
    ass_path: str,
    provider: str = "polly",
    template: str = "default",
    polly_voice_key: str = "korean_female1",
    strip_trailing_punct_last: bool = True,
    pre_split_lines: Optional[List[str]] = None,  # ✅ 추가
):
    """
    Returns:
        segments: [{"start":..,"end":..,"text":..,"ssml":..,"pitch":..}, ...]
        chunk_paths: 라인별 오디오 파일 경로 리스트
        ass_path: 생성된 ASS 파일 경로
    """
    # 1) 절(B)
    if pre_split_lines:
        clause_lines = [str(s).strip() for s in pre_split_lines if str(s).strip()]
    else:
        clause_lines = breath_linebreaks_batch(script_text)

    if not clause_lines:
        return [], [], ass_path

    # 2) SSML(C)
    ssml_lines = convert_lines_to_ssml_batch(clause_lines)

    # 3) Polly 합성 → 병합
    out_dir = os.path.join("assets", "_tts_chunks")
    os.makedirs(out_dir, exist_ok=True)
    chunk_paths = synthesize_lines_polly(ssml_lines, voice_key=polly_voice_key, out_dir=out_dir)

    final_mix = os.path.join("assets", "auto", "_mix_audio.mp3")
    os.makedirs(os.path.dirname(final_mix) or ".", exist_ok=True)
    mix_polly(chunk_paths, final_mix)

    # 4) 세그먼트 타이밍
    segments: List[Dict] = []
    cur = 0.0
    for p, text, ssml in zip(chunk_paths, clause_lines, ssml_lines):
        try:
            with AudioFileClip(p) as a:
                dur = float(a.duration or 0.0)
        except Exception:
            dur = max(0.6, min(8.0, len(text) / 7.0))
        start, end = cur, cur + dur
        pitch_st = _extract_pitch_value(ssml)
        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "ssml": ssml,
                "pitch": pitch_st,
            }
        )
        cur = end

    # 5) ASS
    generate_ass_subtitle(
        segments=segments,
        ass_path=ass_path,
        template_name="educational",
        strip_trailing_punct_last=strip_trailing_punct_last,
        max_chars_per_line=14,
        max_lines=2,
    )

    return segments, chunk_paths, ass_path
