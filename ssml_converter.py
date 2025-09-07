from RAG.chain_builder import get_default_chain
import re
from html import escape as _xml_escape

try:
    from llm_utils import complete_text  # 존재하면 활용
except Exception:
    complete_text = None

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

# ssml_converter.py
import json
from typing import List

# ⬇️ 신규: 전체 대본 → 분절 라인 배열(JSON) 1회
def breath_linebreaks_batch(script_text: str) -> List[str]:
    """
    전체 대본 1개를 입력하고, LLM에서 분절된 라인 배열을 JSON으로 1회 반환.
    - BREATH_PROMPT(기존 프롬프트 본문)는 그대로 사용하고,
      '입/출력은 JSON 배열'만 강제한다.
    """
    text = (script_text or "").strip()
    if not text:
        return []

    prompt = f"""
{BREATH_PROMPT}

[입력/출력 형식(중요)]
- 입력은 전체 대본 1개(개행 포함 가능).
- 출력은 JSON 배열 lines: ["라인1","라인2",...]
- 마크다운/설명/주석 금지. 오직 JSON만.
- 빈 문자열 요소 금지.

입력:
<<<SCRIPT
{text}
SCRIPT>>>
"""
    raw = _complete_with_any_llm(prompt)
    try:
        lines = json.loads(raw)
    except Exception:
        # 혹시 모델이 JSON이 아닌 텍스트를 내보내면 줄 단위로 보정
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    # 최종 정리
    lines = [ln.strip() for ln in lines if isinstance(ln, str) and ln.strip()]
    return lines

# ⬇️ 신규: 분절 라인 배열 → SSML 배열(JSON) 1회
def convert_lines_to_ssml_batch(lines: List[str]) -> List[str]:
    """
    라인 배열을 통째로 입력하고, 동일 길이의 SSML 배열(JSON)을 1회 반환.
    - SSML_PROMPT(기존 프롬프트 본문)는 그대로 사용.
    - 배열 길이/순서 동일성 강제.
    """
    if not lines:
        return []

    prompt = f"""
{SSML_PROMPT}

[입력/출력 형식(중요)]
- 입력은 JSON 배열 lines: ["라인1","라인2",...]
- 출력은 JSON 배열 ssml_list: ["<speak>..</speak>", ...]
- 배열 길이와 순서는 반드시 동일.
- 마크다운/설명/주석 금지. 오직 JSON만.

입력(JSON):
{json.dumps({"lines": lines}, ensure_ascii=False)}
"""
    raw = _complete_with_any_llm(prompt)
    try:
        ssml_list = json.loads(raw)
    except Exception:
        # 비정형 출력 대비: 줄 나눠서 받기(권장X)
        ssml_list = [x.strip() for x in raw.split("\n") if x.strip()]

    if not isinstance(ssml_list, list):
        raise ValueError("SSML 배치 결과가 JSON 배열이 아닙니다.")
    if len(ssml_list) != len(lines):
        raise AssertionError(f"SSML 배열 길이 불일치: {len(ssml_list)} != {len(lines)}")

    return ssml_list

def _looks_english(text: str) -> bool:
    if not text: return False
    en = len(re.findall(r"[A-Za-z]", text))
    ko = len(re.findall(r"[\uac00-\ud7a3]", text))
    return en >= 3 and en > ko * 1.2

def koreanize_if_english(text: str) -> str:
    """문장이 사실상 영어면, 의미 동일 한국어 한 문장으로 변환."""
    t = (text or "").strip()
    if not t or not _looks_english(t):
        return t
        
    # LLM 호출 없이 Google 번역만 사용하여 한국어로 변환
    if GoogleTranslator is not None:
        try:
            return GoogleTranslator(source="auto", target="ko").translate(t)
        except Exception:
            pass

    # 번역 실패 시 원문 유지
    return t
    
def _heuristic_breath_lines(text: str, strict: bool = True) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    if strict:
        # ✨ LLM 실패 시엔 '추가 분절/병합' 절대 하지 않고 원문 라인 그대로 사용
        return [t]

import streamlit as st

# 전역 캐시
_BREATH_CACHE: dict[str, list[str]] = {}


BREATH_PROMPT = """역할: 너는 한국어 대본의 호흡(브레스) 라인브레이크 편집기다.
출력은 텍스트만, 줄바꿈으로만 호흡을 표현한다. 다른 기호·주석·설명·마크다운·태그를 절대 쓰지 않는다.

[불변 규칙]

원문 완전 보존: 글자·공백·숫자·단위·어미·어순을 그대로 유지한다. 줄바꿈만 추가한다.

빈 줄 금지: 연속 빈 줄을 만들지 않는다(모든 줄은 실제 텍스트여야 함).

한 줄 길이 가이드: 기본 3–6단어(또는 8–18글자) 권장. 지나치게 짧은 1–2단어 줄은 피한다.

수치·부호 결합 유지: -173도, 1만 2천 km 같은 숫자+단위/부호는 한 줄에 붙여 둔다.

문장 어미 보존: ~습니다/~합니다/~다/~이다/~것입니다/~수 없습니다 등은 앞말과 한 줄로 유지한다.

질문부호: ?에서는 줄을 바꿔도 좋다(질문 뒤 새 리듬 시작).

담화표지 처리 — 핵심

담화표지 단독 줄 허용: 물론/따라서/즉/그러니까/그리고/그러나/하지만/한쪽으로는/다른 쪽으로는 등은 강조 목적일 때 단독 줄 가능.

단, 담화표지 뒤에 아주 짧은 주어·지시어가 오면 같은 줄로 묶는다:
예) 하지만 우리는, 그리고 우리는, 한쪽으로는 태양의, 다른 쪽으로는 태양이.

보조 용언·문말 구문 유지 — 매우 중요

… 수 있다/없다, … 것이다/것입니다, … 해야 한다, … 할 수 없다 등은 중간에서 끊지 말고 한 줄에 둔다.

예) 지구에서 생명체는 살아남을수 없습니다 ← 한 줄 유지.

명사구/조사 단위: 명사구 내부나 조사 바로 앞·뒤에서 어색하게 끊지 않는다.
"""

SSML_PROMPT = """역할: 너는 한국어 대본을 숏폼용 Amazon Polly SSML로 변환하는 변환기다.
출력은 SSML만, <speak>…</speak> 구조로만 낸다. 마크다운/주석/설명 금지.

[불변 규칙 — 반드시 지켜]
1) 원문 보존: 단어·어순·어미(경어체/평서체) 절대 변경 금지.
   - 각 문장의 끝 어미는 입력 그대로 유지한다. (예: "~습니다/~합니다/~이다/~다" 등)
   - 문장 끝 어미를 다른 형태로 바꾸지 말 것.
   - 단, 비한글 문자는 모두 자연스러운 한글로 교정한다. 수치·단위는 한국어 발음으로 표기(예: 섭씨 칠십 도, 초속 십 킬로미터, 산소 이십 퍼센트).
   
2) 숫자·단위 표기, 고유명사 그대로 유지.
3) 허용 태그: <speak>, <prosody>, <break>만.
4) 허용 문장부호: 물음표(?)와 쉼표(,)만. 마침표(.)/느낌표(!)/줄임표(…)는 출력 금지.
5) 일시정지 규칙:
   - 구(절) 사이: <break time="20ms"/>
   - 문장 사이: <break time="50ms"/>
   - 90ms 초과 금지, 20ms+50ms 연속 사용 금지(중복 브레이크 금지).
6) 변환은 ‘분할’만 한다. 재작성·의역·어휘 치환 금지.
   - 쉼표는 추가해도 되지만, 단어/어미는 그대로여야 한다.

[억양/속도 설계]
- 훅/질문/경고: rate 160~165%, pitch +15~+25%
- 일반 설명/정보: rate 140~155%, pitch -10%~+5%
- 결론/단정/무거운 문장: rate 130~140%, pitch -15%~-20%
- 같은 문장 내 2~3개의 구(절)로 분할하고, 의미가 고조되면 뒤 구절의 rate/pitch를 최대 +5%p 상향,
  침잠이면 최대 -5%p 하향.

[끝맺음 ‘말꼬리’ 짧게 (대본 수정 없이)]
- 문장 마지막 구절(원문 어미 그대로)에만 미세 조정:
  그 구절을 <prosody rate="원래값+5%" pitch="+3%">…</prosody>로 감싼 뒤,
  바로 <break time="50ms"/> 또는 다음 문장으로 넘어간다.
- 이 조정은 어미 텍스트를 바꾸지 않고 발화만 또렷하게 만든다.

[출력 형식]
- 최상위: <speak><prosody volume="+2dB"> … </prosody></speak>
- 각 구(절): <prosody rate="…" pitch="…">원문 일부(어미 포함, 원형 유지)</prosody>
- 구(절) 사이는 30ms, 문장 사이는 50ms. 중복 브레이크 금지.

[검증 체크리스트(내부 적용 후 통과된 경우에만 출력)]
- <speak> 루트, 허용 태그 외 사용 없음?
- break는 20ms/50ms만, 90ms 이하, 연속 중복 없음?
- 각 문장의 마지막 prosody 텍스트가 원문 마지막 어미와 ‘완전히 동일’한가?
- 재작성/치환/어미 변경 없이 원문 부분문자열로만 구성했는가?
- 물음표/쉼표 외의 마침표/느낌표/줄임표를 쓰지 않았는가?
"""

def _complete_with_any_llm(prompt: str) -> str | None:
    if complete_text is not None:
        try:
            return complete_text(prompt)
        except Exception:
            pass
    try:
        # system_prompt를 명시적으로 줌
        chain = get_default_chain("너는 한국어 대본의 호흡(브레스) 라인브레이크 편집기다.")
        out = chain.invoke({"question": prompt})
        if isinstance(out, str):
            return out
        if isinstance(out, dict):
            texts = [str(v) for v in out.values() if isinstance(v, (str, bytes))]
            return max(texts, key=len) if texts else None
    except Exception as e:
        st.error(f"⚠️ LLM 호출 실패: {e}")
    return None

def _unwrap_speak(ssml: str) -> str:
    m = re.search(r"<speak[^>]*>(.*)</speak>", ssml or "", flags=re.S|re.I)
    return (m.group(1) if m else (ssml or "")).strip()
