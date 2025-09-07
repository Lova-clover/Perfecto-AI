# ssml_converter.py
import json
from typing import List
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ===== 공통 LLM 호출 유틸 =====
def _complete_with_any_llm(prompt: str, temperature: float = 1, model: str = "gpt-5-mini") -> str:
    llm = ChatOpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        model=model,
        temperature=temperature,
    )
    output_parser = StrOutputParser()
    chain = ChatPromptTemplate.from_messages([
        ("system", "너는 한국어 숏폼 스크립트의 편집을 돕는 보조 AI다."),
        ("human", "{question}")
    ]) | llm | output_parser

    return chain.invoke({"question": prompt}).strip()

# ===== 2차 분절(절/호흡) 배치 프롬프트 =====
BREATH_LINEBREAK_PROMPT = """역할: 너는 한국어 대본의 호흡(브레스) 라인브레이크 편집기다.
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

# ===== SSML 변환 배치 프롬프트 =====
# * 여기에서만 "비한국어 → 자연스러운 한국어" 교정을 허용
SSML_PROMPT =  """역할: 너는 한국어 대본을 숏폼용 Amazon Polly SSML로 변환하는 변환기다.
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

def breath_linebreaks_batch(text: str) -> List[str]:
    """전체 대본을 LLM 한 번 호출하여 '절/호흡' 단위 리스트로 분절."""
    prompt = BREATH_LINEBREAK_PROMPT.format(script=text)
    raw = _complete_with_any_llm(prompt, temperature=1)
    try:
        arr = json.loads(raw)
        lines = [ (x or "").strip() for x in arr if isinstance(x, str) and x.strip() ]
        return lines
    except Exception:
        # JSON 실패 시 줄단위 폴백
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        return lines

def convert_lines_to_ssml_batch(lines: List[str]) -> List[str]:
    """절 리스트(B)를 LLM 한 번 호출하여 SSML 라인 배열(C)로 변환."""
    if not lines:
        return []
    payload = {"lines": lines}
    prompt = SSML_PROMPT.format(lines_json=json.dumps(payload, ensure_ascii=False, indent=2))
    raw = _complete_with_any_llm(prompt, temperature=1)
    try:
        arr = json.loads(raw)
        out = []
        for i, it in enumerate(arr):
            s = (it or "").strip() if isinstance(it, str) else ""
            # <speak>로 감싸져 있지 않으면 보정
            if s and "<speak" not in s.lower():
                s = f"<speak>{s}</speak>"
            out.append(s)
        if len(out) != len(lines):
            # 길이 불일치 시 보수적으로 맞춰줌
            out = (out + ["<speak></speak>"] * len(lines))[:len(lines)]
        return out
    except Exception:
        # JSON 실패 시 라인 분해 폴백
        arr = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(arr) != len(lines):
            arr = (arr + ["<speak></speak>"] * len(lines))[:len(lines)]
        return arr
