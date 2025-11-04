# image_generator.py — Pexels API rate-limit safe version
import logging # <-- 추가
import os, time, random, subprocess, shlex, requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Iterable, List, Tuple, Optional, Set, Dict, Any

load_dotenv()
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# === Add: English-enforcer ===
import re
from functools import lru_cache
try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

@lru_cache(maxsize=1024)
def _ensure_english(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    # 이미 영문/숫자/기본 기호만 있으면 그대로
    if re.fullmatch(r"[A-Za-z0-9 ,./°%-]+", t):
        return t
    # deep_translator 사용 가능 시 자동 번역
    if GoogleTranslator is not None:
        try:
            return GoogleTranslator(source="auto", target="en").translate(t) or t
        except Exception:
            pass
    return t  # 실패 시 원문 유지

# ===== Global throttle & cache =====
# 연속 호출 사이 최소 간격(초). 300~500ms 권장.
_MIN_INTERVAL_SEC = float(os.getenv("PEXELS_MIN_INTERVAL_SEC", "0.35"))
# 헤더의 reset/Retry-After가 너무 길 때 대기 상한(초)
_MAX_WAIT_ON_429 = int(os.getenv("PEXELS_MAX_WAIT_SEC", "60"))
# 간단 메모리 캐시 (프로세스 생존 동안)
_CACHE: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Any] = {}
_last_call_ts = 0.0

def _sleep_for_throttle():
    global _last_call_ts
    now = time.time()
    elapsed = now - _last_call_ts
    gap = _MIN_INTERVAL_SEC - elapsed
    if gap > 0:
        # 지터를 약간 주어 동시 다발 호출 완화
        time.sleep(gap + random.uniform(0, 0.05))

def _mark_called():
    global _last_call_ts
    _last_call_ts = time.time()

def _pexels_session(total=3, backoff=0.5) -> requests.Session:
    """
    Pexels 전용 requests 세션:
    - 429는 status_forcelist에서 제외 (우리가 직접 처리)
    - 5xx/네트워크 오류만 지수백오프로 재시도
    - 커넥션풀 설정
    """
    if not PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY가 설정되지 않았습니다(.env).")

    s = requests.Session()
    s.headers.update({
        "User-Agent": "PerfectoAI/1.0 (+rate-limit-aware)",
        "Authorization": PEXELS_API_KEY
    })
    retry = Retry(
        total=total, read=total, connect=total, status=total,
        backoff_factor=backoff,
        status_forcelist=(500, 502, 503, 504),  # 429 제외
        allowed_methods={"GET", "HEAD"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def _headers_wait_seconds(h: requests.structures.CaseInsensitiveDict) -> int:
    """
    429 응답 헤더에서 대기시간 계산:
    - Retry-After(초) 우선
    - X-Ratelimit-Reset(UNIX epoch)까지 남은 시간
    """
    ra = h.get("Retry-After")
    if ra:
        try:
            v = int(float(ra))
            return max(1, min(v, _MAX_WAIT_ON_429))
        except:
            pass
    reset = h.get("X-Ratelimit-Reset")
    if reset:
        try:
            remain = int(float(reset)) - int(time.time())
            return max(1, min(remain, _MAX_WAIT_ON_429))
        except:
            pass
    return 5  # 정보 없으면 짧게 대기

def _log_quota(h: requests.structures.CaseInsensitiveDict):
    lim = h.get("X-Ratelimit-Limit")
    rem = h.get("X-Ratelimit-Remaining")
    reset = h.get("X-Ratelimit-Reset")
    if rem is not None:
        try:
            irem = int(rem)
        except:
            irem = -1
        if irem >= 0:
            msg = f"[PEXELS] remaining={irem}"
            if lim: msg += f"/{lim}"
            if reset:
                try:
                    eta = max(0, int(float(reset)) - int(time.time()))
                    msg += f" reset~{eta}s"
                except:
                    pass
            if irem <= 5:
                print("⚠️", msg)
            else:
                print("ℹ️", msg)

def _cache_key(url: str, params: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
    # 파라미터 정렬하여 키 생성
    return (url, tuple(sorted(params.items())))

def _pexels_get_json(sess: requests.Session, url: str, params: Dict[str, Any]) -> Any:
    """
    - 전역 슬로틀링 적용
    - 429면 헤더 기반으로 대기 후 재시도
    - 간단 캐시 사용(동일 url+params)
    """
    key = _cache_key(url, params)
    if key in _CACHE:
        return _CACHE[key]

    while True:
        _sleep_for_throttle()
        # <<<--- [API 호출 로깅: 시작] ---
        logging.info(f"[PEXELS API CALL] URL: {url}, Query: {params.get('query')}, Page: {params.get('page')}")
        api_start_time = time.time()
        # --- [API 호출 로깅: 끝] --->>>
        r = sess.get(url, params=params, timeout=(5, 20))
        # <<<--- [API 응답 로깅: 시작] ---
        api_end_time = time.time()
        logging.info(f"[PEXELS API RESP] Status: {r.status_code}, Elapsed: {api_end_time - api_start_time:.2f}s")
        # --- [API 응답 로깅: 끝] --->>>
        
        if r.status_code == 429:
            wait_s = _headers_wait_seconds(r.headers)
            print(f"⏳ 429 Too Many Requests. {wait_s}s 대기 후 재시도.")
            time.sleep(wait_s)
            continue
        # 5xx 등은 세션의 Retry가 알아서 처리, 여기서는 상태 확인만
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            # 비정상 상태 로깅 후 즉시 raise
            print(f"❌ HTTP {r.status_code} on {url} params={params}")
            raise e

        _mark_called()
        _log_quota(r.headers)
        data = r.json()
        _CACHE[key] = data
        return data

# ===== 영상 리사이즈 유틸 =====
def _shrink_to_720_inplace(path: str):
    """
    원본을 720p 세로 기준으로 즉시 재인코딩하여 덮어쓴다.
    - 9:16 캔버스 기준으로 scale+crop
    - CRF 30, ultrafast → 클라우드에서도 가볍게
    """
    tmp = path + ".shrink.mp4"
    vf = 'scale=720:-2:force_original_aspect_ratio=increase,crop=720:1080,format=yuv420p'
    cmd = f'ffmpeg -y -i {shlex.quote(path)} -vf {shlex.quote(vf)} -r 30 -c:v libx264 -preset ultrafast -crf 30 -c:a aac -b:a 96k {shlex.quote(tmp)}'
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    os.replace(tmp, path)

# ===== 이미지 단건 =====
def generate_image_pexels(query: str, save_path: str, per_page: int = 1) -> str:
    sess = _pexels_session()
    per_page = max(1, min(per_page, 80))
    q_en = _ensure_english(query)
    params = {"query": q_en, "per_page": per_page, "page": 1}
    data = _pexels_get_json(sess, "https://api.pexels.com/v1/search", params)
    photos = data.get("photos", [])
    if not photos:
        raise RuntimeError(f"Pexels 이미지 검색 결과 없음: {query}")

    image_url = photos[0]["src"].get("large2x") or photos[0]["src"].get("large") or photos[0]["src"].get("original")
    if not image_url:
        raise RuntimeError("다운로드 가능한 이미지 URL을 찾지 못했습니다.")
    _sleep_for_throttle()
    r2 = sess.get(image_url, timeout=(5, 30))
    r2.raise_for_status()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(r2.content)
    print(f"✅ 이미지 저장 완료: {save_path}")
    _mark_called()
    return save_path

# ===== 이미지 배치 =====
def generate_images_for_topic(
    query: str,
    num_images: int,
    start_index: int = 0,
    page: int = 1,                               # 시작 페이지
    exclude_ids: Optional[Iterable[int]] = None, # 이미 사용한 사진 ID 집합
    return_ids: bool = False                     # 사진 ID 반환 여부
):
    sess = _pexels_session()
    per_page = min(max(num_images, 1), 80)
    image_paths: List[str] = []
    kept_ids: List[int] = []
    exclude: Set[int] = set(int(x) for x in (exclude_ids or []))
    current_page = max(1, page)

    while len(image_paths) < num_images:
        q_en = _ensure_english(query)
        params = {"query": q_en, "per_page": per_page, "page": current_page}
        data = _pexels_get_json(sess, "https://api.pexels.com/v1/search", params)
        photos = data.get("photos", [])
        if not photos:
            break

        for photo in photos:
            if len(image_paths) >= num_images:
                break
            pid = int(photo.get("id") or 0)
            if pid and pid in exclude:
                continue

            src = photo["src"].get("large2x") or photo["src"].get("large") or photo["src"].get("original")
            if not src:
                continue

            save_path = f"assets/image_{start_index + len(image_paths)}.jpg"
            try:
                _sleep_for_throttle()
                rimg = sess.get(src, timeout=(5, 30))
                rimg.raise_for_status()
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(rimg.content)
                _mark_called()
                image_paths.append(save_path)
                if pid:
                    kept_ids.append(pid)
            except Exception as e:
                print(f"⚠️ 이미지 다운로드 실패 ({src}): {e}")

        current_page += 1

    return (image_paths, kept_ids) if return_ids else image_paths

# ===== 영상 배치 =====
def generate_videos_for_topic(
    query: str,
    num_videos: int,
    start_index: int = 0,
    min_duration: float = 3.0,
    orientation: str = "portrait",
    page: int = 1,                               # 시작 페이지
    exclude_ids: Optional[Iterable[int]] = None, # 제외할 Pexels video id
    return_ids: bool = False,                    # 선택된 id도 반환
) -> List[str] | Tuple[List[str], List[int]]:
    sess = _pexels_session()
    saved: List[str] = []
    chosen_ids: List[int] = []
    exclude: Set[int] = set(int(x) for x in (exclude_ids or []))
    per_page = min(max(num_videos, 1), 80)
    current_page = max(1, page)

    while len(saved) < num_videos:
        q_en = _ensure_english(query)
        params = {"query": q_en, "per_page": per_page, "page": current_page}
        data = _pexels_get_json(sess, "https://api.pexels.com/videos/search", params)
        videos = data.get("videos", [])
        if not videos:
            break

        for v in videos:
            if len(saved) >= num_videos:
                break

            vid = int(v.get("id") or 0)
            if vid and vid in exclude:
                continue

            w, h = v.get("width"), v.get("height")
            dur = float(v.get("duration", 0) or 0)
            if orientation == "portrait" and not (w and h and h > w):
                continue
            if dur < min_duration:
                continue

            candidates = [f for f in v.get("video_files", [])
                          if f.get("link", "").endswith(".mp4")
                          and 720 <= (f.get("width") or 0) <= 1080]
            if not candidates:
                candidates = [f for f in v.get("video_files", [])
                              if f.get("link", "").endswith(".mp4")]
            if not candidates:
                continue

            picked = sorted(candidates, key=lambda f: f.get("width") or 10**9)[0]
            url = picked["link"]
            save_path = f"assets/video_{start_index + len(saved)}.mp4"
            tmp_path = save_path + ".part"

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            try:
                _sleep_for_throttle()
                with sess.get(url, stream=True, timeout=(5, 30)) as r:
                    r.raise_for_status()
                    with open(tmp_path, "wb") as f:
                        for chunk in r.iter_content(1024 * 64):
                            if chunk:
                                f.write(chunk)
                _mark_called()

                # sanity check: 너무 작은 파일 제거
                if os.path.getsize(tmp_path) < 10 * 1024:  # 10KB 미만이면 실패로 간주
                    raise IOError(f"Downloaded file too small: {tmp_path}")

                os.replace(tmp_path, save_path)

                # 9:16 캔버스 720x1080으로 가볍게 정리(선택)
                try:
                    _shrink_to_720_inplace(save_path)
                except Exception:
                    pass

                saved.append(save_path)
                if vid:
                    chosen_ids.append(vid)

            except Exception as e:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except:
                    pass
                print(f"⚠️ 영상 다운로드 실패: {e} | {url}")

        current_page += 1

    return (saved, chosen_ids) if return_ids else saved
