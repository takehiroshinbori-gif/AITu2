from __future__ import annotations

import os
import sys
import re
import io
import json
import base64
import tempfile
import ctypes
import threading
import queue
import time
import random
import uuid
import wave
import math
import subprocess
import shutil
from typing import List, Dict, Tuple, Optional, Any

from dotenv import load_dotenv
from openai import OpenAI
import requests
from janome.tokenizer import Tokenizer
import websocket


# Load env early
load_dotenv(override=True)


# ======================
# Behavior / Text config
# ======================
SYSTEM_PROMPT = (
    "あなたは可愛い女の子のAIアシスタント『もも』。親しみやすいタメ口で、絵文字は1つだけ。"
    "会話の状況に合わせて言葉づかいで感情表現する。感情タグは出力しない。"
    "1〜2文・最大80文字・テンポ良く短く返す。"
)

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TTS_ENABLED = os.getenv("TTS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "FISHAUDIO").upper()  # FISHAUDIO or OPENAI
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "verse")

FISH_API_KEY = os.getenv("FISHAUDIO_API_KEY", "")
FISH_VOICE_ID = os.getenv("FISHAUDIO_VOICE_ID", "")
FISH_TTS_URL = os.getenv("FISHAUDIO_TTS_URL", "https://api.fish.audio/v1/tts")
FISH_FORMAT = os.getenv("FISHAUDIO_FORMAT", "mp3")
FISH_SPEED = os.getenv("FISHAUDIO_SPEED", "")
FISH_PITCH = os.getenv("FISHAUDIO_PITCH", "")
FISHAUDIO_USE_CONTROL_TAGS = os.getenv("FISHAUDIO_USE_CONTROL_TAGS", "false").lower() in {"1", "true", "yes", "on"}

LIPSYNC_MP3_VIA_FFMPEG = os.getenv("LIPSYNC_MP3_VIA_FFMPEG", "true").lower() in {"1", "true", "yes", "on"}
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
FFPLAY_PATH = os.getenv("FFPLAY_PATH", "ffplay")
LIPSYNC_PRE_DELAY_MS = int(os.getenv("LIPSYNC_PRE_DELAY_MS", "60"))

MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
MEMORY_FILE = os.getenv("MEMORY_FILE", "memory.json")
MEMORY_MAX_CHARS = int(os.getenv("MEMORY_MAX_CHARS", "360"))
IDLE_AUTO_TALK_SEC = int(os.getenv("IDLE_AUTO_TALK_SEC", "30"))

MENTION_ENABLED = os.getenv("MENTION_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
MENTION_MIN_SEC = int(os.getenv("MENTION_MIN_SEC", "25"))
MENTION_JITTER_SEC = int(os.getenv("MENTION_JITTER_SEC", "20"))

AUTO_SELF_TALE_PCT_BASE = float(os.getenv("AUTO_SELF_TALE_PCT", "0.45"))
ADAPTIVE_AUTO_SELF = os.getenv("ADAPTIVE_AUTO_SELF", "true").lower() in {"1", "true", "yes", "on"}
ASK_SUPPRESS_MIN = int(os.getenv("ASK_SUPPRESS_MIN", "20"))
ASK_SUPPRESS_KEYWORDS = [w.strip() for w in os.getenv("ASK_SUPPRESS_KEYWORDS", "質問しないで,聞かないで,クエスチョン禁止").split(",") if w.strip()]
ASK_ALLOW_KEYWORDS = [w.strip() for w in os.getenv("ASK_ALLOW_KEYWORDS", "質問していいよ,聞いていいよ").split(",") if w.strip()]

LAUGH_SEQUENCE_ENABLED = os.getenv("LAUGH_SEQUENCE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
LAUGH_BURST_MAX = int(os.getenv("LAUGH_BURST_MAX", "3"))

SANITIZE_STRICT = os.getenv("TEXT_SANITIZE_STRICT", "true").lower() in {"1", "true", "yes", "on"}
DECAY_HALF_LIFE_MIN = int(os.getenv("DECAY_HALF_LIFE_MIN", "30"))
SELF_TALE_BLACKLIST = {w.strip() for w in os.getenv("SELF_TALE_BLACKLIST", "夢,夢見,夢の中,空想,妄想,ファンタジー,異世界").split(",") if w.strip()}
SELF_TALE_MAX_RATIO_5 = float(os.getenv("SELF_TALE_MAX_RATIO_5", "0.4"))
SELF_TALE_MAX_PER_MIN = int(os.getenv("SELF_TALE_MAX_PER_MIN", "2"))

DEBUG_TTS = os.getenv("DEBUG_TTS", "false").lower() in {"1", "true", "yes", "on"}

# ======================
# VTS / Motion / Limits
# ======================
VTS_ENABLED = os.getenv("VTS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
VTS_URL = os.getenv("VTS_URL", "ws://127.0.0.1:8001")
VTS_PLUGIN_NAME = os.getenv("VTS_PLUGIN_NAME", "MomoLive")
VTS_PLUGIN_DEV = os.getenv("VTS_PLUGIN_DEV", "Takeh")

VTS_PARAM_MOUTH = os.getenv("VTS_PARAM_MOUTH", "MouthOpen")
VTS_PARAM_EYE_L_OPEN = os.getenv("VTS_PARAM_EYE_L_OPEN", "EyeOpenLeft")
VTS_PARAM_EYE_R_OPEN = os.getenv("VTS_PARAM_EYE_R_OPEN", "EyeOpenRight")
VTS_PARAM_EYE_X = os.getenv("VTS_PARAM_EYE_X", "EyeRightX")
VTS_PARAM_EYE_Y = os.getenv("VTS_PARAM_EYE_Y", "EyeRightY")
VTS_PARAM_HEAD_X = os.getenv("VTS_PARAM_HEAD_X", "FaceAngleX")
VTS_PARAM_HEAD_Y = os.getenv("VTS_PARAM_HEAD_Y", "FaceAngleY")
VTS_PARAM_HEAD_Z = os.getenv("VTS_PARAM_HEAD_Z", "FaceAngleZ")
VTS_PARAM_MOUTH_SMILE = os.getenv("VTS_PARAM_MOUTH_SMILE", "MouthSmile")
VTS_PARAM_BROWS = os.getenv("VTS_PARAM_BROWS", "Brows")

VTS_PARAM_EYE_SMILE_L = os.getenv("VTS_PARAM_EYE_SMILE_L", "")
VTS_PARAM_EYE_SMILE_R = os.getenv("VTS_PARAM_EYE_SMILE_R", "")
VTS_PARAM_CHEEK = os.getenv("VTS_PARAM_CHEEK", "CheekPuff")
VTS_PARAM_MOUTH_X = os.getenv("VTS_PARAM_MOUTH_X", "")
VTS_PARAM_BODY_ROT_X = os.getenv("VTS_PARAM_BODY_ROT_X", "")
VTS_PARAM_BODY_ROT_Y = os.getenv("VTS_PARAM_BODY_ROT_Y", "")
VTS_PARAM_BODY_ROT_Z = os.getenv("VTS_PARAM_BODY_ROT_Z", "")
VTS_PARAM_AUTO_BREATH = os.getenv("VTS_PARAM_AUTO_BREATH", "")
VTS_PARAM_TONGUE_OUT = os.getenv("VTS_PARAM_TONGUE_OUT", "")
VTS_L2D_EYE_SHOCK_VISIBLE = os.getenv("VTS_L2D_EYE_SHOCK_VISIBLE", "")
VTS_L2D_EYE_NORMAL_VISIBLE = os.getenv("VTS_L2D_EYE_NORMAL_VISIBLE", "")

# Global VTS rate limiting
def _f_env(name: str, default_value: float) -> float:
    try:
        return float(os.getenv(name, str(default_value)))
    except Exception:
        return default_value

VTS_MAX_UPDATES_PER_SEC = _f_env("VTS_MAX_UPDATES_PER_SEC", 45.0)
VTS_MIN_DELTA = _f_env("VTS_MIN_DELTA", 0.02)

HOTKEY_DELAY_MS = int(os.getenv("VTS_HOTKEY_DELAY_MS", "200"))
VTS_HOTKEY_LIST = [s.strip() for s in (os.getenv("VTS_HOTKEY_LIST", "")).split(",") if s.strip()]

VTS_HK_EYES_CRY = os.getenv("VTS_HK_EYES_CRY", "")
VTS_HK_EYES_LOVE = os.getenv("VTS_HK_EYES_LOVE", "")
VTS_HK_SIGN_ANGRY = os.getenv("VTS_HK_SIGN_ANGRY", "")
VTS_HK_SIGN_SHOCK = os.getenv("VTS_HK_SIGN_SHOCK", "")
VTS_HK_HELLO = os.getenv("VTS_HK_HELLO", "")
VTS_HK_BYE = os.getenv("VTS_HK_BYE", "")


def _safe_float_env(name: str, dflt: float) -> float:
    try:
        return float(os.getenv(name, str(dflt)))
    except Exception:
        return dflt

VTS_MOUTH_SCALE = _safe_float_env("VTS_MOUTH_SCALE", 4.0)
VTS_HEAD_AMP = _safe_float_env("VTS_HEAD_AMP", 0.75)
VTS_EYE_AMP = _safe_float_env("VTS_EYE_AMP", 0.55)
VTS_HEAD_TILT_SCALE = _safe_float_env("VTS_HEAD_TILT_SCALE", 0.32)
VTS_SMILE_BASE = _safe_float_env("VTS_SMILE_BASE", 0.08)
VTS_SMILE_HAPPY = _safe_float_env("VTS_SMILE_HAPPY", 0.72)
VTS_SMILE_LAUGH = _safe_float_env("VTS_SMILE_LAUGH", 0.92)
VTS_SMILE_SAD = _safe_float_env("VTS_SMILE_SAD", 0.06)

ACTIONS_ENABLED = os.getenv("ACTIONS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
PLANNER_ENABLED = os.getenv("PLANNER_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
PLANNER_USE_LLM = os.getenv("PLANNER_USE_LLM", "false").lower() in {"1", "true", "yes", "on"}

BLINK_ENABLED = os.getenv("BLINK_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
BLINK_MIN_SEC = _safe_float_env("BLINK_MIN_SEC", 3.0)
BLINK_MAX_SEC = _safe_float_env("BLINK_MAX_SEC", 7.0)
BLINK_DOUBLE_PROB = _safe_float_env("BLINK_DOUBLE_PROB", 0.20)
BLINK_ASYM_OFFSET_MS = int(os.getenv("BLINK_ASYM_OFFSET_MS", "30"))

EYE_MICRO_NOISE = _safe_float_env("EYE_MICRO_NOISE", 0.025)
GAZE_HEAD_COUPLE = _safe_float_env("GAZE_HEAD_COUPLE", 0.18)
GAZE_AVERSION_SEC = _safe_float_env("GAZE_AVERSION_SEC", 1.4)
EYE_HAPPY_SQUINT = _safe_float_env("EYE_HAPPY_SQUINT", 0.85)
EYE_SURPRISE_OPEN = _safe_float_env("EYE_SURPRISE_OPEN", 1.0)
EYE_TIRED_LEVEL = _safe_float_env("EYE_TIRED_LEVEL", 0.70)

BREATH_ENABLED = os.getenv("BREATH_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
BREATH_PERIOD_SEC = _safe_float_env("BREATH_PERIOD_SEC", 5.5)
BREATH_AMP = _safe_float_env("BREATH_AMP", 0.06)

IDLE_MOTION_ENABLED = os.getenv("IDLE_MOTION_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
IDLE_MICRO_MOVE_MIN_SEC = _safe_float_env("IDLE_MICRO_MOVE_MIN_SEC", 4.0)
IDLE_MICRO_MOVE_MAX_SEC = _safe_float_env("IDLE_MICRO_MOVE_MAX_SEC", 9.0)
IDLE_LOOK_AROUND = os.getenv("IDLE_LOOK_AROUND", "true").lower() in {"1", "true", "yes", "on"}
IDLE_STRETCH_PROB = _safe_float_env("IDLE_STRETCH_PROB", 0.12)

SIGH_SOUND = os.getenv("SIGH_SOUND", "")
HUM_SOUNDS = [s.strip() for s in os.getenv("HUM_SOUNDS", "").split(",") if s.strip()]

BASE_TEMPO = _safe_float_env("BASE_TEMPO", 1.0)
NIGHT_SLOW = _safe_float_env("NIGHT_SLOW", 0.85)

# Lipsync tuning (upward-compatible)
LIPSYNC_NOISE_FLOOR = _safe_float_env("LIPSYNC_NOISE_FLOOR", 0.015)
LIPSYNC_RELEASE_MS = int(os.getenv("LIPSYNC_RELEASE_MS", "90"))  # how quickly to close after silence
LIPSYNC_DECAY = _safe_float_env("LIPSYNC_DECAY", 0.55)  # smoothing alpha in envelope builder
LIPSYNC_ZERO_AFTER_MS = int(os.getenv("LIPSYNC_ZERO_AFTER_MS", "250"))  # force zero if silence exceeds this

# Multi-brain pipeline (dialogue / tts / motion) for speed
TRIPLE_BRAIN = os.getenv("TRIPLE_BRAIN", "true").lower() in {"1", "true", "yes", "on"}


# =========
# Globals
# =========
SESSION = requests.Session()
T = Tokenizer()
JP_STOP = {"これ", "それ", "あれ", "ここ", "そこ", "あそこ", "の", "こと", "もの", "よう", "ため", "ところ", "人", "私", "あなた", "今日", "昨日", "明日", "今", "時", "子", "さん", "ちゃん", "くん", "です", "ます", "する", "いる", "ある", "なる", "できる", "笑", "www"}


# =========
# Utilities
# =========
def ensure_env_present() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not found", file=sys.stderr)
        sys.exit(1)
    if TTS_ENABLED and TTS_PROVIDER == "FISHAUDIO":
        if not FISH_API_KEY or not FISH_VOICE_ID:
            print("[ERROR] FishAudio keys missing", file=sys.stderr)
            sys.exit(1)


def openai_client() -> OpenAI:
    return OpenAI()


def safe_float(x: Any, d: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return d


def load_memory(path: str) -> dict:
    p = os.path.abspath(path)
    mem: Dict[str, Any] = {"profile": ""}
    if os.path.exists(p):
        try:
            mem = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            mem = {"profile": ""}
    mem.setdefault("topic_counts", {})
    mem.setdefault("recent_auto_texts", [])
    mem.setdefault("user_samples", [])
    mem.setdefault("mentioned_ids", [])
    mem.setdefault("last_auto_ts", 0.0)
    mem.setdefault("last_auto_kind", "")
    mem.setdefault("ask_suppress_until", 0.0)
    mem.setdefault("ask_suppress_hits", 0.0)
    mem.setdefault("last_decay_ts", time.time())
    mem.setdefault("topic_ring", [])
    mem.setdefault("motion_history", [])
    mem.setdefault("mood", "neutral")
    mem.setdefault("intimacy", 0.2)
    ra = mem.get("recent_auto_texts", [])
    mem["recent_auto_texts"] = [r for r in ra if isinstance(r, dict) and {"kind", "ts", "text"}.issubset(r.keys())][-10:]
    us = mem.get("user_samples", [])
    mem["user_samples"] = [str(x) for x in us][-50:]
    return mem


def save_memory(path: str, memory: dict) -> None:
    json.dump(memory, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def memory_system_message(memory: dict) -> str:
    profile = (memory or {}).get("profile", "").strip()
    return f"ユーザーの長期プロフィール: {profile}" if profile else ""


def update_memory_summary(client: OpenAI, memory: dict, user_text: str, assistant_text: str, max_chars: int) -> dict:
    prior = (memory or {}).get("profile", "")
    prompt = (
        f"長期的に使えるプロフィールのみを要約・更新してください（最大{max_chars}文字・推測除外・固有名を短縮）。\n"
        f"現在:{prior}\nユーザー:{user_text}\nアシスタント:{assistant_text}\n出力は本文のみ。"
    )
    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=240,
    )
    newp = (res.choices[0].message.content or "").strip()
    memory["profile"] = newp[:max_chars]
    return memory


def build_messages(base_prompt: str, memory: dict, chat_history: List[Dict[str, str]], keep_last: int = 16) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": base_prompt}]
    mem_msg = memory_system_message(memory) if MEMORY_ENABLED else ""
    if mem_msg:
        msgs.append({"role": "system", "content": mem_msg})
    if chat_history:
        msgs.extend(chat_history[-keep_last:])
    return msgs


def sanitize_for_tts(text: str) -> str:
    _EMO_WORDS = {
        "happy",
        "sad",
        "angry",
        "surprised",
        "calm",
        "laugh",
        "cry",
        "whisper",
        "shout",
        "excited",
        "tender",
        "serious",
        "playful",
        "fear",
        "disgust",
    }
    _EMOJI_RE = re.compile(
        "[" "\U0001F300-\U0001F5FF" "\U0001F600-\U0001F64F" "\U0001F680-\U0001F6FF" "\U0001F700-\U0001F77F" "\U0001F780-\U0001F7FF" "\U0001F800-\U0001F8FF" "\U0001F900-\U0001F9FF" "\U0001FA00-\U0001FA6F" "\U0001FA70-\U0001FAFF" "\U00002700-\U000027BF" "\U00002600-\U000026FF" "\U00002B00-\U00002BFF" + "]",
        flags=re.UNICODE,
    )

    def strip_any_tags(s: str) -> str:
        s = re.sub(r"\\[(?:happy|sad|angry|surprised|calm|laugh|cry|whisper|shout|excited|tender|serious|playful|fear|disgust)\\]", "", s, flags=re.I)
        s = re.sub(r"<[^>]+>", "", s)
        return s.strip()

    if SANITIZE_STRICT:
        text = _EMOJI_RE.sub("", text)
        text = strip_any_tags(text)
        text = re.sub(r"[^0-9A-Za-z\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303F。、．，！!？\?\sー〜…（）()\-\:・／/]+", "", text)
        text = re.sub(r"[。]{3,}", "。", text)
        text = re.sub(r"[！!]{3,}", "！！", text)
        text = re.sub(r"[？\?]{3,}", "？？", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        text = re.sub(r"^[、。・／/!！\?？\-ー]+", "", text)
        text = re.sub(r"[、。・／/!！\?？\-ー]+$", "", text)
    else:
        text = strip_any_tags(text)
    return text or "えっと…うん。"


def generate_reply(client: OpenAI, messages: List[Dict[str, str]]) -> str:
    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.8,
        max_tokens=160,
    )
    content = (res.choices[0].message.content or "").strip()
    if not content:
        return "うん。"
    if random.random() < 0.10:
        pre = random.choice(["えっと…", "あ、", "うーん…", "そのね、"])
        content = f"{pre}{content}"
    return content


def get_sentiment(user_text: str) -> str:
    laugh = {"爆笑", "草", "www", "ﾜﾗ", "ワロタ", "🤣", "😂", "lol", "lmao"}
    pos = {"楽しい", "嬉しい", "好き", "最高", "すごい", "面白い", "可愛い", "笑", "ナイス", "ありがと", "助かる"}
    neg = {"嫌い", "悲しい", "辛い", "最悪", "怖い", "怒り", "怒", "むかつく", "ごめん", "謝"}
    for t in T.tokenize(user_text):
        w = t.surface
        if w in laugh:
            return "laugh"
        if w in pos:
            return "happy"
        if w in neg:
            return "sad"
    if any(x in user_text for x in laugh):
        return "laugh"
    if any(x in user_text for x in pos):
        return "happy"
    if any(x in user_text for x in neg):
        return "sad"
    if any(x in user_text for x in ["驚", "びっくり", "!?", "！?"]):
        return "surprised"
    if any(x in user_text for x in ["落ち着", "穏やか", "大丈夫", "平気"]):
        return "calm"
    return "neutral"


def _extract_audio_bytes(resp: requests.Response) -> bytes:
    ctype = (resp.headers.get("content-type") or "").lower()
    if "json" in ctype:
        data = resp.json()
        cands = [
            data.get("audio"),
            data.get("audio_base64"),
            data.get("audioContent"),
            (data.get("data") or {}).get("audio") if isinstance(data.get("data"), dict) else None,
            (data.get("result") or {}).get("audio") if isinstance(data.get("result"), dict) else None,
            data.get("url") or data.get("audio_url"),
        ]
        b64 = next((x for x in cands if isinstance(x, str) and not x.startswith("http")), None)
        if b64:
            return base64.b64decode(b64)
        url = next((x for x in cands if isinstance(x, str) and x.startswith("http")), None)
        if url:
            r = SESSION.get(url, timeout=20)
            r.raise_for_status()
            return r.content
        raise RuntimeError("FishAudio JSON missing audio")
    return resp.content


def _emotion_speed_pitch(emotion: str) -> Tuple[float, float]:
    def f(x: str, df: float) -> float:
        try:
            return float(x) if x else df
        except Exception:
            return df

    base_speed = f(FISH_SPEED, 1.0)
    base_pitch = f(FISH_PITCH, 1.0)
    m = (emotion or "neutral").lower()
    return {
        "laugh": (base_speed * 1.15, base_pitch * 1.10),
        "happy": (base_speed * 1.08, base_pitch * 1.05),
        "sad": (base_speed * 0.90, base_pitch * 0.96),
        "angry": (base_speed * 1.05, base_pitch * 1.02),
        "surprised": (base_speed * 1.12, base_pitch * 1.08),
        "calm": (base_speed * 0.97, base_pitch * 0.98),
        "tender": (base_speed * 0.98, base_pitch * 1.02),
        "serious": (base_speed * 0.98, base_pitch * 0.98),
        "playful": (base_speed * 1.06, base_pitch * 1.05),
    }.get(m, (base_speed, base_pitch))


def request_fish_audio_bytes(text: str, emotion: str = "neutral") -> bytes:
    headers = {"Authorization": f"Bearer {FISH_API_KEY}", "Content-Type": "application/json"}
    speed, pitch = _emotion_speed_pitch(emotion)
    payload: Dict[str, Any] = {"format": FISH_FORMAT}
    clean_text = text
    if FISHAUDIO_USE_CONTROL_TAGS:
        clean_text = f"[{emotion}] {text}"
        try:
            payload["emotion"] = emotion
            payload["speed"] = float(f"{speed:.3f}")
            payload["pitch"] = float(f"{pitch:.3f}")
        except Exception:
            pass

    # Try both reference_id and voice_id field names for broad compatibility
    for base in ({"reference_id": FISH_VOICE_ID, "text": clean_text}, {"voice_id": FISH_VOICE_ID, "text": clean_text}):
        p = dict(base)
        p.update(payload)
        resp = SESSION.post(FISH_TTS_URL, headers=headers, json=p, timeout=45)
        if resp.status_code == 200:
            return _extract_audio_bytes(resp)

    # Fallback without speed/pitch
    for base in ({"reference_id": FISH_VOICE_ID, "text": clean_text}, {"voice_id": FISH_VOICE_ID, "text": clean_text}):
        p = dict(base)
        p["format"] = FISH_FORMAT
        resp = SESSION.post(FISH_TTS_URL, headers=headers, json=p, timeout=45)
        if resp.status_code == 200:
            return _extract_audio_bytes(resp)

    raise RuntimeError("FishAudio TTS failed")


def request_openai_tts_bytes(client: OpenAI, text: str) -> bytes:
    with client.audio.speech.with_streaming_response.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=text,
        format="mp3",
    ) as response:
        bio = io.BytesIO()
        for chunk in response.iter_bytes(chunk_size=4096):
            bio.write(chunk)
        return bio.getvalue()


def request_tts_audio_bytes(client: Optional[OpenAI], text: str, emotion: str) -> bytes:
    if TTS_PROVIDER == "FISHAUDIO":
        return request_fish_audio_bytes(text, emotion)
    if TTS_PROVIDER == "OPENAI":
        if client is None:
            raise RuntimeError("OpenAI client not initialized for TTS")
        return request_openai_tts_bytes(client, text)
    raise RuntimeError(f"Unknown TTS_PROVIDER: {TTS_PROVIDER}")


def is_mp3(b: bytes) -> bool:
    return (len(b) >= 3 and b[:3] == b"ID3") or (len(b) >= 2 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0)


def is_wav(b: bytes) -> bool:
    return len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WAVE"


def _mci_play_blocking(path: str, type_hint: str) -> bool:
    alias = f"ttsa_{int(time.time()*1000)}"
    open_cmd = f'open "{path}" type {type_hint} alias {alias}'
    if ctypes.windll.winmm.mciSendStringW(open_cmd, None, 0, None) != 0:
        return False
    try:
        return ctypes.windll.winmm.mciSendStringW(f"play {alias} wait", None, 0, None) == 0
    finally:
        ctypes.windll.winmm.mciSendStringW(f"close {alias}", None, 0, None)


def _play_file_blocking(path: str) -> None:
    if sys.platform.startswith("win"):
        ext = os.path.splitext(path)[1].lower()
        type_hint = "mpegvideo" if ext == ".mp3" else "waveaudio"
        if not _mci_play_blocking(path, type_hint):
            os.startfile(path)  # type: ignore[attr-defined]
            time.sleep(0.1)
        return
    if sys.platform == "darwin":
        try:
            subprocess.run(["afplay", path], check=True)
            return
        except Exception:
            subprocess.run(["open", path])
            return
    player = shutil.which(FFPLAY_PATH) or shutil.which("ffplay")
    if player:
        subprocess.run([player, "-nodisp", "-autoexit", "-hide_banner", "-loglevel", "error", path], check=False)
        return
    if shutil.which("mpg123"):
        subprocess.run(["mpg123", "-q", path], check=False)
        return
    if shutil.which("aplay"):
        subprocess.run(["aplay", path], check=False)
        return
    subprocess.run(["xdg-open", path], check=False)


def _play_bytes_blocking(audio: bytes) -> None:
    suffix, type_hint = (".mp3", "mpegvideo") if is_mp3(audio) else (".wav", "waveaudio") if is_wav(audio) else (".mp3", "mpegvideo")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(audio)
        tmp.flush()
    finally:
        tmp.close()
    try:
        _play_file_blocking(tmp.name)
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def _build_wav_envelope(audio_bytes: bytes, frame_ms: int = 12) -> List[float]:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as w:
            ch = w.getnchannels()
            sw = w.getsampwidth()
            sr = w.getframerate()
            nf = w.getnframes()
            data = w.readframes(nf)
        if ch < 1 or sw < 1 or sr < 8000:
            return []
        frame_size = ch * sw
        win_samples = max(1, int(sr * frame_ms / 1000))
        win_bytes = win_samples * frame_size
        max_val = float((1 << (8 * sw - 1)) - 1) if sw > 1 else 127.0

        def read_le(buf: bytes, off: int, width: int) -> int:
            if width == 1:
                return buf[off] - 128
            if width == 2:
                return int.from_bytes(buf[off : off + 2], "little", signed=True)
            if width == 3:
                b = buf[off : off + 3]
                return int.from_bytes(b + (b[2:3] if b[2] < 0x80 else b"\x00"), "little", signed=True)
            return int.from_bytes(buf[off : off + width], "little", signed=True)

        env: List[float] = []
        prev = 0.0
        i = 0
        n = len(data)
        silence_ms_accum = 0
        while i < n:
            end = min(i + win_bytes, n)
            sums = 0.0
            cnt = 0
            j = i
            while j + frame_size <= end:
                total = 0
                o = j
                for _ in range(ch):
                    total += read_le(data, o, sw)
                    o += sw
                mono = total / max(1, ch)
                sums += mono * mono
                cnt += 1
                j += frame_size
            r = math.sqrt(sums / cnt) / max_val if cnt else 0.0
            r = max(0.0, min(1.0, r))
            # noise gate
            if r < LIPSYNC_NOISE_FLOOR:
                r = 0.0
                silence_ms_accum += frame_ms
            else:
                silence_ms_accum = 0
            # exponential smoothing towards r
            alpha = max(0.1, min(0.95, LIPSYNC_DECAY))
            val = prev * alpha + r * (1.0 - alpha)
            # if prolonged silence, force fast release to zero
            if silence_ms_accum >= LIPSYNC_ZERO_AFTER_MS:
                val = 0.0
                prev = 0.0
                env.append(0.0)
                i = end
                continue
            env.append(val)
            prev = val
            i = end
        return env
    except Exception:
        return []


def _decode_mp3_to_wav_bytes_ffmpeg(mp3_bytes: bytes) -> bytes:
    path = os.getenv("FFMPEG_PATH") or "ffmpeg"
    exe = shutil.which(path) or shutil.which("ffmpeg")
    if not exe:
        return b""
    try:
        proc = subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", "pipe:0", "-f", "wav", "pipe:1"],
            input=mp3_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return proc.stdout or b""
    except Exception:
        return b""


class VTSClient(threading.Thread):
    def __init__(self, url: str, plugin_name: str, plugin_dev: str, param_mouth: str) -> None:
        super().__init__(daemon=True)
        self.url = url
        self.plugin_name = plugin_name
        self.plugin_dev = plugin_dev
        self.param_mouth = param_mouth
        self.ws: Optional[Any] = None
        self._stop = False
        self._token_path = os.path.join(os.path.abspath(os.getcwd()), "vts_token.json")
        self._lock = threading.Lock()
        self.last_error = ""
        # Rate control
        self.max_updates_per_sec = max(1.0, VTS_MAX_UPDATES_PER_SEC)
        self.min_delta = max(0.0, VTS_MIN_DELTA)
        self._last_inject_ts = 0.0
        self._last_sent_values: Dict[str, float] = {}
        self._last_hotkey_ts: Dict[str, float] = {}
        self.start()

    def run(self) -> None:
        while not self._stop:
            try:
                if self.ws is None:
                    self.ws = websocket.create_connection(self.url, timeout=10)
                    self._handshake()
                    self.last_error = ""
                time.sleep(0.1)
            except Exception as e:
                self.last_error = str(e)
                try:
                    if self.ws:
                        self.ws.close()
                except Exception:
                    pass
                self.ws = None
                time.sleep(1.0)

    def stop(self) -> None:
        self._stop = True
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass
        self.ws = None

    def _send(self, msg_type: str, data: dict) -> bool:
        if not self.ws:
            self.last_error = "WebSocket not connected"
            return False
        msg = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": str(uuid.uuid4()),
            "messageType": msg_type,
            "data": data,
        }
        try:
            with self._lock:
                self.ws.send(json.dumps(msg))
            return True
        except Exception as e:
            try:
                if self.ws:
                    self.ws.close()
            except Exception:
                pass
            self.last_error = str(e)
            self.ws = None
            return False

    def _recv(self) -> dict:
        if not self.ws:
            return {}
        try:
            with self._lock:
                raw = self.ws.recv()
            return json.loads(raw) if raw else {}
        except Exception as e:
            self.last_error = str(e)
            return {}

    def _load_token(self) -> str:
        try:
            return (json.load(open(self._token_path, "r", encoding="utf-8")) or {}).get("token", "")
        except Exception:
            return ""

    def _save_token(self, token: str) -> None:
        try:
            json.dump({"token": token}, open(self._token_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _handshake(self) -> None:
        token = self._load_token()
        if not token:
            self._send("AuthenticationTokenRequest", {"pluginName": self.plugin_name, "pluginDeveloper": self.plugin_dev})
            resp = self._recv()
            token = ((resp or {}).get("data") or {}).get("authenticationToken", "")
            if token:
                self._save_token(token)
        self._send(
            "AuthenticationRequest",
            {"pluginName": self.plugin_name, "pluginDeveloper": self.plugin_dev, "authenticationToken": token},
        )
        _ = self._recv()

    def _handshake_with_ws(self, ws) -> None:
        try:
            token = self._load_token()
            if not token:
                msg = {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": str(uuid.uuid4()),
                    "messageType": "AuthenticationTokenRequest",
                    "data": {"pluginName": self.plugin_name, "pluginDeveloper": self.plugin_dev},
                }
                ws.send(json.dumps(msg))
                raw = ws.recv() or "{}"
                resp = json.loads(raw)
                token = ((resp or {}).get("data") or {}).get("authenticationToken", "")
                if token:
                    self._save_token(token)
            msg = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": str(uuid.uuid4()),
                "messageType": "AuthenticationRequest",
                "data": {"pluginName": self.plugin_name, "pluginDeveloper": self.plugin_dev, "authenticationToken": token},
            }
            ws.send(json.dumps(msg))
            _ = ws.recv()
        except Exception as e:
            self.last_error = str(e)
            raise

    def _open_temp_ws(self):
        ws = websocket.create_connection(self.url, timeout=12)
        self._handshake_with_ws(ws)
        return ws

    def status(self) -> Dict[str, str]:
        return {
            "connected": str(self.ws is not None),
            "url": self.url,
            "token_file": self._token_path,
            "last_error": self.last_error or "",
        }

    def list_hotkeys(self) -> List[Dict[str, str]]:
        try:
            ws = self._open_temp_ws()
            msg = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": str(uuid.uuid4()),
                "messageType": "HotkeysInCurrentModelRequest",
                "data": {},
            }
            ws.send(json.dumps(msg))
            result: List[Dict[str, str]] = []
            for _ in range(20):
                raw = ws.recv() or "{}"
                resp = json.loads(raw)
                if (resp.get("messageType") or "") == "HotkeysInCurrentModelResponse":
                    arr = ((resp.get("data") or {}).get("hotkeys")) or []
                    for h in arr:
                        if isinstance(h, dict):
                            result.append({
                                "name": str(h.get("name") or ""),
                                "type": str(h.get("type") or ""),
                                "id": str(h.get("id") or ""),
                            })
                    break
            try:
                ws.close()
            except Exception:
                pass
            return result
        except Exception as e:
            self.last_error = str(e)
            return []

    def trigger_hotkey(self, name_or_id: str) -> bool:
        """Trigger a VTS hotkey by its name or id (best-effort)."""
        if not name_or_id:
            return False
        try:
            ws = self._open_temp_ws()
            ok = False
            # Try by id first
            for payload in (
                {"hotkeyID": name_or_id},
                {"hotkeyName": name_or_id},
            ):
                msg = {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": str(uuid.uuid4()),
                    "messageType": "HotkeyTriggerRequest",
                    "data": payload,
                }
                ws.send(json.dumps(msg))
                for _ in range(10):
                    raw = ws.recv() or "{}"
                    resp = json.loads(raw)
                    mt = resp.get("messageType") or ""
                    if mt == "HotkeyTriggerResponse":
                        ok = True
                        break
                    if mt == "RequestFail":
                        break
                if ok:
                    break
            try:
                ws.close()
            except Exception:
                pass
            return ok
        except Exception as e:
            self.last_error = str(e)
            return False

    def trigger_hotkey_once(self, name_or_id: str, min_interval_ms: int = 1500) -> bool:
        """Trigger a hotkey with a simple debounce to avoid spam during TTS."""
        now = time.time()
        last = self._last_hotkey_ts.get(name_or_id, 0.0)
        if (now - last) * 1000.0 < max(0, min_interval_ms):
            return True
        ok = self.trigger_hotkey(name_or_id)
        if ok:
            self._last_hotkey_ts[name_or_id] = now
        return ok

    def trigger_hotkeys_sequence(self, names: List[str], delay_ms: int = HOTKEY_DELAY_MS) -> bool:
        """Trigger a sequence of hotkeys with a delay between them."""
        overall_ok = True
        for nm in names:
            if not nm:
                continue
            if not self.trigger_hotkey(nm):
                overall_ok = False
            time.sleep(max(0, int(delay_ms)) / 1000.0)
        return overall_ok

    def _rate_limit_wait(self) -> None:
        min_interval = 1.0 / max(1.0, self.max_updates_per_sec)
        now = time.time()
        wait = min_interval - (now - self._last_inject_ts)
        if wait > 0:
            time.sleep(min(wait, min_interval))
        self._last_inject_ts = time.time()

    def inject(self, values: Dict[str, float], mode: str = "set") -> bool:
        # Filter by delta to avoid tiny spam
        filtered: Dict[str, float] = {}
        for pid, v in values.items():
            if pid is None or pid == "":
                continue
            try:
                val = float(max(0.0, min(1.0, v)))
            except Exception:
                continue
            last = self._last_sent_values.get(pid)
            if last is None or abs(val - last) >= self.min_delta:
                filtered[pid] = val
        if not filtered:
            return True

        # Basic rate limiting (shared across callers)
        self._rate_limit_wait()

        pairs = [{"id": pid, "value": val} for pid, val in filtered.items()]
        ok = self._send("InjectParameterDataRequest", {"parameterValues": pairs, "faceFound": True, "mode": mode})
        if ok:
            self._last_sent_values.update(filtered)
        return ok

    def set_param(self, param_id: str, value: float) -> bool:
        try:
            v = float(value)
        except Exception:
            return False
        return self.inject({param_id: v}, "set")

    def animate_param(self, param_id: str, start: Optional[float], end: float, duration_ms: int, fps: int = 60, ease: str = "sine") -> bool:
        # Note: animate uses inject under the hood; global rate limit applies
        try:
            s = float(start) if start is not None else 0.5
            e = float(end)
            dur = max(1, int(duration_ms))
            frames = max(1, int(fps))
        except Exception:
            return False
        steps = max(1, int(dur / max(1, int(1000 / frames))))

        def ease_sine_in_out(t: float) -> float:
            return 0.5 - 0.5 * math.cos(math.pi * t)

        def f(t: float) -> float:
            if ease == "linear":
                return t
            return ease_sine_in_out(t)

        def _run() -> None:
            for i in range(steps + 1):
                t = i / steps
                val = s + (e - s) * f(t)
                if not self.inject({param_id: max(0.0, min(1.0, val))}, "set"):
                    break
                time.sleep(max(0.005, 1.0 / max(1, frames)))

        threading.Thread(target=_run, daemon=True).start()
        return True

    def drive_mouth(self, envelope: List[float], frame_ms: int = 16, scale: float = 1.0, close_tail: bool = True, speed_factor: float = 1.0) -> None:
        if not envelope or not VTS_PARAM_MOUTH:
            return
        alpha = 0.55 if speed_factor >= 1.0 else 0.70
        smooth: List[float] = []
        prev = 0.0
        for v in envelope:
            val = prev * alpha + v * (1.0 - alpha)
            smooth.append(val)
            prev = val

        def _r() -> None:
            if LIPSYNC_PRE_DELAY_MS > 0:
                time.sleep(LIPSYNC_PRE_DELAY_MS / 1000.0)
            # slow slightly to avoid spam
            ms = max(10, int(frame_ms / max(0.5, min(2.0, speed_factor))))
            last_nonzero_ts = time.time()
            for v in smooth:
                send_val = max(0.0, min(1.0, v * scale))
                if send_val > 0.001:
                    last_nonzero_ts = time.time()
                # early close if we've been silent for a while
                if LIPSYNC_RELEASE_MS > 0 and (time.time() - last_nonzero_ts) * 1000.0 >= LIPSYNC_RELEASE_MS:
                    send_val = 0.0
                if not self.inject({VTS_PARAM_MOUTH: send_val}, "set"):
                    break
                time.sleep(ms / 1000.0)
            if close_tail:
                # stronger close tail to guarantee shut mouth
                for z in (0.18, 0.08, 0.0):
                    self.inject({VTS_PARAM_MOUTH: z}, "set")
                    time.sleep(0.04)

        threading.Thread(target=_r, daemon=True).start()

    def drive_from_env(self, param_id: str, envelope: List[float], amp: float = 0.1, frame_ms: int = 40) -> None:
        if not envelope or not param_id:
            return

        def _r() -> None:
            if LIPSYNC_PRE_DELAY_MS > 0:
                time.sleep(LIPSYNC_PRE_DELAY_MS / 1000.0)
            for v in envelope:
                val = max(0.0, min(1.0, 0.5 + (v - 0.3) * amp))
                if not self.inject({param_id: val}, "set"):
                    break
                time.sleep(frame_ms / 1000.0)

        threading.Thread(target=_r, daemon=True).start()

    def list_params(self) -> List[str]:
        self.last_error = ""
        names: List[str] = []
        ws = None
        try:
            ws = self._open_temp_ws()
            msg = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": str(uuid.uuid4()),
                "messageType": "InputParameterListRequest",
                "data": {},
            }
            ws.send(json.dumps(msg))
            for _ in range(20):
                raw = ws.recv() or "{}"
                resp = json.loads(raw)
                mt = resp.get("messageType") or ""
                if mt in ("InputParameterListResponse", "ParameterListResponse"):
                    data = (resp.get("data") or {})
                    merged: List[dict] = []
                    for key in ("defaultParameters", "customParameters", "parameters", "availableParameters"):
                        arr = data.get(key)
                        if isinstance(arr, list):
                            merged.extend(arr)
                    for p in merged:
                        if isinstance(p, dict):
                            n = p.get("name") or p.get("parameterName") or p.get("id") or ""
                            if n:
                                names.append(n)
                    break
                if mt == "RequestFail":
                    reason = ((resp.get("data") or {}).get("message")) or ((resp.get("data") or {}).get("reason")) or "request failed"
                    self.last_error = f"InputParameterListRequest -> {reason}"
                    break
            if not names:
                msg = {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": str(uuid.uuid4()),
                    "messageType": "ParameterListRequest",
                    "data": {},
                }
                ws.send(json.dumps(msg))
                for _ in range(20):
                    raw = ws.recv() or "{}"
                    resp = json.loads(raw)
                    mt = resp.get("messageType") or ""
                    if mt == "ParameterListResponse":
                        data = (resp.get("data") or {})
                        arr = data.get("availableParameters", [])
                        if isinstance(arr, list):
                            for p in arr:
                                if isinstance(p, dict):
                                    n = p.get("name") or p.get("parameterName") or p.get("id") or ""
                                    if n:
                                        names.append(n)
                        break
                    if mt == "RequestFail":
                        reason = ((resp.get("data") or {}).get("message")) or ((resp.get("data") or {}).get("reason")) or "request failed"
                        self.last_error = f"ParameterListRequest -> {reason}"
                        break
        except Exception as e:
            self.last_error = str(e)
        finally:
            try:
                if ws:
                    ws.close()
            except Exception:
                pass
        names = sorted(set(names), key=lambda s: s.lower())
        return names

    def list_in_params(self) -> List[str]:
        """Return only Input Parameters (no fallback)."""
        self.last_error = ""
        names: List[str] = []
        ws = None
        try:
            ws = self._open_temp_ws()
            msg = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": str(uuid.uuid4()),
                "messageType": "InputParameterListRequest",
                "data": {},
            }
            ws.send(json.dumps(msg))
            for _ in range(20):
                raw = ws.recv() or "{}"
                resp = json.loads(raw)
                mt = resp.get("messageType") or ""
                if mt == "InputParameterListResponse":
                    data = (resp.get("data") or {})
                    merged: List[dict] = []
                    for key in ("defaultParameters", "customParameters", "parameters", "availableParameters"):
                        arr = data.get(key)
                        if isinstance(arr, list):
                            merged.extend(arr)
                    for p in merged:
                        if isinstance(p, dict):
                            n = p.get("name") or p.get("parameterName") or p.get("id") or ""
                            if n:
                                names.append(n)
                    break
                if mt == "RequestFail":
                    reason = ((resp.get("data") or {}).get("message")) or ((resp.get("data") or {}).get("reason")) or "request failed"
                    self.last_error = f"InputParameterListRequest -> {reason}"
                    break
        except Exception as e:
            self.last_error = str(e)
        finally:
            try:
                if ws:
                    ws.close()
            except Exception:
                pass
        names = sorted(set(names), key=lambda s: s.lower())
        return names

    def write_env_auto(self) -> str:
        names = self.list_params()

        def norm(s: str) -> str:
            return "".join(ch for ch in s.lower() if ch.isalnum())

        norm_map = [(n, norm(n)) for n in names]

        def pick(patterns: List[str], fallback: str) -> str:
            pats = [norm(p) for p in patterns]
            for orig, nrm in norm_map:
                for p in pats:
                    if p and p in nrm:
                        return orig
            return fallback

        mapping = {
            "VTS_PARAM_MOUTH": pick(["mouth open", "mouthopen", "parammouthopeny", "parammouthopen"], VTS_PARAM_MOUTH),
            "VTS_PARAM_MOUTH_SMILE": pick(["mouth smile", "mouthsmile", "parammouthsmile"], VTS_PARAM_MOUTH_SMILE),
            "VTS_PARAM_HEAD_X": pick(["face left/right rotation", "faceleftrightrotation", "faceanglex", "headx", "body rotation x"], VTS_PARAM_HEAD_X),
            "VTS_PARAM_HEAD_Y": pick(["face up/down rotation", "faceupdownrotation", "faceangley", "heady", "body rotation y"], VTS_PARAM_HEAD_Y),
            "VTS_PARAM_HEAD_Z": pick(["face lean rotation", "faceleanrotation", "faceanglez", "headz", "body rotation z", "tilt"], VTS_PARAM_HEAD_Z),
            "VTS_PARAM_EYE_L_OPEN": pick(["eye open left", "eyeopenleft", "lefteyeopen", "parameyelopen"], VTS_PARAM_EYE_L_OPEN),
            "VTS_PARAM_EYE_R_OPEN": pick(["eye open right", "eyeopenright", "righteyeopen", "parameyeropen"], VTS_PARAM_EYE_R_OPEN),
            "VTS_PARAM_EYE_X": pick(["eye x", "eyerightx", "eyex", "lookx", "gazex"], VTS_PARAM_EYE_X),
            "VTS_PARAM_EYE_Y": pick(["eye y", "eyerighty", "eyey", "looky", "gazey"], VTS_PARAM_EYE_Y),
            "VTS_PARAM_BROWS": pick(["brow form left", "brow form right", "brow height left", "brow height right", "brow", "brows"], VTS_PARAM_BROWS),
        }
        optional = {
            "VTS_PARAM_EYE_SMILE_L": pick(["eye l smile", "eye left smile", "eyesmileleft"], VTS_PARAM_EYE_SMILE_L),
            "VTS_PARAM_EYE_SMILE_R": pick(["eye smile right", "eyesmileright"], VTS_PARAM_EYE_SMILE_R),
            "VTS_PARAM_CHEEK": pick(["cheek", "cheekpuff", "blush"], VTS_PARAM_CHEEK),
            "VTS_PARAM_AUTO_BREATH": pick(["auto breath", "autobreath", "breath"], VTS_PARAM_AUTO_BREATH),
            "VTS_PARAM_MOUTH_X": pick(["mouthx"], VTS_PARAM_MOUTH_X),
            "VTS_PARAM_BODY_ROT_X": pick(["body rotation x", "bodyanglex"], VTS_PARAM_BODY_ROT_X),
            "VTS_PARAM_BODY_ROT_Y": pick(["body rotation y", "bodyangley"], VTS_PARAM_BODY_ROT_Y),
            "VTS_PARAM_BODY_ROT_Z": pick(["body rotation z", "bodyanglez"], VTS_PARAM_BODY_ROT_Z),
        }
        lines = [f"{k}={v}"] if (k := None) else []  # dummy to keep type hints happy
        lines = [f"{k}={v}" for k, v in mapping.items()]
        lines += [f"{k}={v}" for k, v in optional.items() if v]
        path = os.path.join(os.getcwd(), ".env.auto")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path


class MotionEngine:
    def __init__(self, vts: Optional["VTSClient"]) -> None:
        self.vts = vts
        self.cap: Dict[str, str] = {}
        self._build_capabilities()
        self._stop = False
        if IDLE_MOTION_ENABLED and vts is not None:
            threading.Thread(target=self._idle_loop, daemon=True).start()

    def _build_capabilities(self) -> None:
        preferred = {
            "jump_y": ["FacePositionY", "MocopiBodyPositionY", VTS_PARAM_HEAD_Y],
            "pos_x": ["FacePositionX", "MocopiBodyPositionX"],
            "pos_z": ["FacePositionZ", "MocopiBodyPositionZ"],
            "head_x": [VTS_PARAM_HEAD_X, "FaceAngleX"],
            "head_y": [VTS_PARAM_HEAD_Y, "FaceAngleY"],
            "head_z": [VTS_PARAM_HEAD_Z, "FaceAngleZ"],
            "eye_x": [VTS_PARAM_EYE_X, "EyeRightX", "EyeLeftX"],
            "eye_y": [VTS_PARAM_EYE_Y, "EyeRightY", "EyeLeftY"],
            "eye_l_open": [VTS_PARAM_EYE_L_OPEN, "EyeOpenLeft"],
            "eye_r_open": [VTS_PARAM_EYE_R_OPEN, "EyeOpenRight"],
            "smile": [VTS_PARAM_MOUTH_SMILE, "MouthSmile"],
            "mouth": [VTS_PARAM_MOUTH, "MouthOpen"],
            "brows": [VTS_PARAM_BROWS, "Brows"],
            "cheek": [VTS_PARAM_CHEEK, "CheekPuff"],
            "body_y": [VTS_PARAM_BODY_ROT_Y, "BodyAngleY", "Body Rotation Y", "MocopiBodyAngleY"],
        }
        names: List[str] = []
        try:
            if self.vts is not None:
                names = self.vts.list_params()
        except Exception:
            names = []
        name_set = set(names)

        def pick(cands: List[str]) -> str:
            for c in cands:
                if c and c in name_set:
                    return c
            return cands[0] if cands else ""

        for k, arr in preferred.items():
            self.cap[k] = pick(arr)

    def stop(self) -> None:
        self._stop = True

    def _idle_loop(self) -> None:
        rng = random.Random()
        while not self._stop:
            wait = rng.uniform(max(0.5, IDLE_MICRO_MOVE_MIN_SEC), max(IDLE_MICRO_MOVE_MIN_SEC, IDLE_MICRO_MOVE_MAX_SEC))
            time.sleep(wait)
            if self.vts is None:
                continue
            try:
                ex = self.cap.get("eye_x")
                ey = self.cap.get("eye_y")
                if ex:
                    self.vts.animate_param(ex, 0.5, min(1.0, max(0.0, 0.35 + random.random() * 0.3)), 280, 45)
                if ey:
                    self.vts.animate_param(ey, 0.5, min(1.0, max(0.0, 0.35 + random.random() * 0.3)), 280, 45)
                hy = self.cap.get("head_y")
                if hy:
                    self.vts.animate_param(hy, 0.5, 0.42, 300, 45)
                sm = self.cap.get("smile")
                if sm:
                    self.vts.animate_param(sm, 0.25, 0.45, 340, 45)
                if IDLE_LOOK_AROUND and random.random() < IDLE_STRETCH_PROB:
                    hx = self.cap.get("head_x")
                    if hx:
                        self.vts.animate_param(hx, 0.5, 0.2, 240, 45)
                        self.vts.animate_param(hx, 0.2, 0.8, 320, 45)
                        self.vts.animate_param(hx, 0.8, 0.5, 260, 45)
            except Exception:
                continue

    def action_jump(self, strength: float = 0.9) -> None:
        if self.vts is None:
            return
        y = self.cap.get("jump_y") or self.cap.get("head_y")
        if not y:
            return
        mid = 0.5
        up = min(1.0, mid + 0.25 + 0.35 * strength)
        self.vts.animate_param(y, mid, up, 200, 60)
        self.vts.animate_param(y, up, mid, 260, 60)

    def action_wave(self, hand: str = "right", cycles: int = 3) -> None:
        if self.vts is None:
            return
        hx = self.cap.get("head_x")
        if not hx:
            return
        for _ in range(cycles):
            self.vts.animate_param(hx, 0.5, 0.2, 180, 50)
            self.vts.animate_param(hx, 0.2, 0.8, 240, 50)
        self.vts.animate_param(hx, 0.8, 0.5, 220, 50)

    def action_nod(self, times: int = 2) -> None:
        if self.vts is None:
            return
        hy = self.cap.get("head_y")
        if not hy:
            return
        for _ in range(max(1, times)):
            self.vts.animate_param(hy, 0.5, 0.25, 180, 60)
            self.vts.animate_param(hy, 0.25, 0.5, 220, 60)

    def action_shake(self, times: int = 2, small: bool = False) -> None:
        if self.vts is None:
            return
        hx = self.cap.get("head_x")
        if not hx:
            return
        a = 0.12 if small else 0.3
        for _ in range(max(1, times)):
            self.vts.animate_param(hx, 0.5, 0.5 - a, 150, 60)
            self.vts.animate_param(hx, 0.5 - a, 0.5 + a, 200, 60)
            self.vts.animate_param(hx, 0.5 + a, 0.5, 170, 60)

    def action_wink(self, side: str = "right") -> None:
        if self.vts is None:
            return
        pid = self.cap.get("eye_r_open") if side == "right" else self.cap.get("eye_l_open")
        if not pid:
            return
        self.vts.animate_param(pid, 0.85, 0.05, 120, 60)
        self.vts.animate_param(pid, 0.05, 0.85, 160, 60)

    def action_blush(self, strong: bool = False) -> None:
        if self.vts is None:
            return
        pid = self.cap.get("cheek") or self.cap.get("smile")
        if not pid:
            return
        a = 0.85 if strong else 0.6
        self.vts.animate_param(pid, 0.25, a, 280, 45)
        self.vts.animate_param(pid, a, 0.35, 280, 45)

    def anticipation(self) -> None:
        if self.vts is None:
            return
        hy = self.cap.get("head_y")
        if hy:
            self.vts.animate_param(hy, 0.5, 0.62, 140, 60)

    def follow_through(self) -> None:
        if self.vts is None:
            return
        hy = self.cap.get("head_y")
        if hy:
            self.vts.animate_param(hy, 0.5, 0.38, 180, 60)
        if self.cap.get("smile"):
            self.vts.animate_param(self.cap["smile"], 0.45, 0.3, 220, 50)

    def laugh_shake(self) -> None:
        self.action_shake(times=3, small=False)


class StateManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.state = "idle"
        self.emotion = "neutral"
        self.tempo = BASE_TEMPO
        self.night = False

    def set(self, state: Optional[str] = None, emotion: Optional[str] = None, tempo: Optional[float] = None, night: Optional[bool] = None):
        with self._lock:
            if state is not None:
                self.state = state
            if emotion is not None:
                self.emotion = emotion
            if tempo is not None:
                self.tempo = tempo
            if night is not None:
                self.night = night

    def get(self) -> Tuple[str, str, float, bool]:
        with self._lock:
            return self.state, self.emotion, self.tempo, self.night


class EyeController(threading.Thread):
    def __init__(self, vts: Optional["VTSClient"], state: StateManager, engine: Optional[MotionEngine]):
        super().__init__(daemon=True)
        self.vts = vts
        self.state = state
        self.engine = engine
        self.stop_flag = False
        self._caps: Dict[str, str] = {}
        self._last_blink = time.time() + random.uniform(BLINK_MIN_SEC, BLINK_MAX_SEC)
        self._rng = random.Random()
        self._ensure_caps()

    def _ensure_caps(self) -> None:
        if self.vts is None:
            return
        try:
            names = self.vts.list_params()
        except Exception:
            names = []
        s = set(names)

        def pick(*cands: str) -> str:
            for c in cands:
                if c and c in s:
                    return c
            return cands[0] if cands else ""

        self._caps = {
            "lx": VTS_PARAM_EYE_X,
            "ly": VTS_PARAM_EYE_Y,
            "lo": VTS_PARAM_EYE_L_OPEN,
            "ro": VTS_PARAM_EYE_R_OPEN,
            "hx": VTS_PARAM_HEAD_X,
            "hy": VTS_PARAM_HEAD_Y,
            "hz": VTS_PARAM_HEAD_Z,
            "esl": VTS_PARAM_EYE_SMILE_L,
            "esr": VTS_PARAM_EYE_SMILE_R,
        }

    def run(self) -> None:
        if self.vts is None or not BLINK_ENABLED:
            return
        while not self.stop_flag:
            try:
                state, emo, tempo, night = self.state.get()
                now = time.time()
                target_x = 0.5 + (self._rng.random() - 0.5) * EYE_MICRO_NOISE * 2.0
                target_y = 0.5 + (self._rng.random() - 0.5) * EYE_MICRO_NOISE * 2.0
                if state == "think":
                    target_x += 0.08 * (1 if self._rng.random() > 0.5 else -1)
                    target_y -= 0.06
                elif state == "listen":
                    target_x = 0.5 + (self._rng.random() - 0.5) * EYE_MICRO_NOISE
                    target_y = 0.5 + (self._rng.random() - 0.5) * EYE_MICRO_NOISE
                elif state == "speak":
                    if self._rng.random() < 0.04:
                        target_x += 0.18 if self._rng.random() > 0.5 else -0.18
                        target_y += 0.06 if self._rng.random() > 0.5 else -0.06

                # Use set_param to avoid animation spam
                self._set(self._caps.get("lx"), target_x)
                self._set(self._caps.get("ly"), target_y)

                if self._rng.random() < 0.25 and GAZE_HEAD_COUPLE > 0 and self._caps.get("hx") and self._caps.get("hy"):
                    self.vts.animate_param(self._caps["hx"], 0.5, max(0, min(1, 0.5 + (target_x - 0.5) * GAZE_HEAD_COUPLE)), int(220 / max(0.5, tempo)), 45)
                    self.vts.animate_param(self._caps["hy"], 0.5, max(0, min(1, 0.5 + (target_y - 0.5) * GAZE_HEAD_COUPLE)), int(260 / max(0.5, tempo)), 45)

                base_eye_open = 0.9
                if emo in ("happy", "laugh"):
                    base_eye_open = EYE_HAPPY_SQUINT
                if emo == "surprised":
                    base_eye_open = EYE_SURPRISE_OPEN
                if night:
                    base_eye_open = min(base_eye_open, EYE_TIRED_LEVEL)
                lo = max(0.0, min(1.0, base_eye_open + (self._rng.random() - 0.5) * 0.03))
                ro = max(0.0, min(1.0, base_eye_open + (self._rng.random() - 0.5) * 0.03))
                if emo != "surprised" and now >= self._last_blink:
                    self._blink(lo, ro, double=(self._rng.random() < BLINK_DOUBLE_PROB), tempo=tempo)
                    self._last_blink = now + self._rng.uniform(BLINK_MIN_SEC, BLINK_MAX_SEC)
                else:
                    self._set(self._caps.get("lo"), lo)
                    self._set(self._caps.get("ro"), ro)
            except Exception:
                pass
            time.sleep(0.12)

    def stop(self) -> None:
        self.stop_flag = True

    def _set(self, pid: Optional[str], value: float) -> None:
        if not pid:
            return
        self.vts.set_param(pid, value)

    def _blink(self, lo_base: float, ro_base: float, double: bool, tempo: float) -> None:
        lo = self._caps.get("lo")
        ro = self._caps.get("ro")
        if not lo or not ro:
            return
        d = max(60, int(180 / max(0.5, tempo)))
        self.vts.animate_param(lo, lo_base, 0.0, d, 60)
        time.sleep(BLINK_ASYM_OFFSET_MS / 1000.0)
        self.vts.animate_param(ro, ro_base, 0.0, d, 60)
        time.sleep(d / 1000.0)
        self.vts.animate_param(lo, 0.0, lo_base, d + 40, 60)
        self.vts.animate_param(ro, 0.0, ro_base, d + 40, 60)
        if double:
            time.sleep(0.06)
            self.vts.animate_param(lo, lo_base, 0.0, d, 60)
            self.vts.animate_param(ro, ro_base, 0.0, d, 60)
            time.sleep(d / 1000.0)
            self.vts.animate_param(lo, 0.0, lo_base, d + 40, 60)
            self.vts.animate_param(ro, 0.0, ro_base, d + 40, 60)


class ActionPlanner:
    def __init__(self, vts: Optional["VTSClient"]) -> None:
        self.vts = vts

    def list_hotkeys(self) -> List[Dict[str, str]]:
        if self.vts is None:
            return []
        return self.vts.list_hotkeys()

    def _hk_auto(self, emotion: str) -> Optional[str]:
        hks = self.list_hotkeys()

        def find_one(keys: List[str]) -> Optional[str]:
            for h in hks:
                name = (h.get("name") or "").lower()
                if any(k in name for k in keys):
                    return h.get("name")
            return None

        if emotion == "sad":
            return VTS_HK_EYES_CRY or find_one(["cry", "涙", "泣"])
        if emotion in ("happy", "laugh"):
            return VTS_HK_EYES_LOVE or find_one(["love", "heart", "ハート"])
        if emotion == "angry":
            return VTS_HK_SIGN_ANGRY or find_one(["angry", "怒"])
        if emotion == "surprised":
            return VTS_HK_SIGN_SHOCK or find_one(["shock", "驚", "びっくり"])
        return None

    def plan(self, comment: str, use_llm: bool = False) -> Dict[str, Any]:
        c = comment.lower()
        acts: List[Dict[str, Any]] = []
        if any(k in c for k in ["はじめまして", "こんにちは", "こんちゃ", "おは", "やあ", "hi", "hello"]):
            if VTS_HK_HELLO:
                acts.append({"type": "hotkey", "name": VTS_HK_HELLO})
        if any(k in c for k in ["バイバイ", "またね", "おやすみ", "じゃあね", "bye"]):
            if VTS_HK_BYE:
                acts.append({"type": "hotkey", "name": VTS_HK_BYE})
        if any(k in c for k in ["手振", "バイバイ", "wave", "振って"]):
            acts.append({"type": "wave"})
        if any(k in c for k in ["ジャンプ", "跳", "飛べ", "ぴょん"]):
            acts.append({"type": "jump"})
        if any(k in c for k in ["うなず", "頷", "はい"]):
            acts.append({"type": "nod"})
        if any(k in c for k in ["首振", "ぶんぶん", "いいえ", "no"]):
            acts.append({"type": "shake"})
        if any(k in c for k in ["ウィンク", "wink"]):
            acts.append({"type": "wink", "side": "right"})
        if any(k in c for k in ["照", "赤面", "てれる", "恥ず", "ドキドキ", "ハート目"]):
            acts.append({"type": "blush", "strong": True})
        if any(k in c for k in ["可愛い", "かわいい", "天才", "すごい", "最高"]):
            acts += [{"type": "blush"}, {"type": "wink", "side": "right"}]
        return {"actions": acts[:8]}

    def execute(self, plan: Dict[str, Any], engine: Optional[MotionEngine]) -> None:
        if engine is None:
            return
        for a in (plan.get("actions") or []):
            t = a.get("type")
            if t == "jump":
                engine.action_jump()
            elif t == "wave":
                engine.action_wave()
            elif t == "nod":
                engine.action_nod()
            elif t == "shake":
                engine.action_shake(small=True)
            elif t == "wink":
                engine.action_wink(a.get("side", "right"))
            elif t == "blush":
                engine.action_blush(bool(a.get("strong", False)))
            elif t == "hotkey" and self.vts is not None:
                self.vts.trigger_hotkey(str(a.get("name", "")))


class TTSWorker:
    def __init__(self, client: Optional[OpenAI], vts_client: Optional["VTSClient"] = None, engine: Optional[MotionEngine] = None, state: Optional[StateManager] = None) -> None:
        self.queue: "queue.Queue[tuple[str, str]]" = queue.Queue(maxsize=8)
        self._stop = False
        self.vts = vts_client
        self.client = client
        self.engine = engine
        self.state = state or StateManager()
        # Triple-brain: separate TTS generation and playback
        self._use_pipeline = bool(TRIPLE_BRAIN)
        if self._use_pipeline:
            self._audio_queue: "queue.Queue[tuple[bytes, list[float], str]]" = queue.Queue(maxsize=4)
            self._tts_thread = threading.Thread(target=self._run_tts, daemon=True)
            self._play_thread = threading.Thread(target=self._run_playback, daemon=True)
            self._tts_thread.start()
            self._play_thread.start()
        else:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def enqueue(self, text: str, emotion: str) -> None:
        while self.queue.qsize() > 6:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break
        self.queue.put((text, emotion))

    def _apply_emotion_pose(self, emotion: str) -> None:
        if not self.vts:
            return
        emo = emotion.lower()
        smile = VTS_SMILE_BASE
        if emo == "happy":
            smile = VTS_SMILE_HAPPY
        elif emo == "laugh":
            smile = VTS_SMILE_LAUGH
        elif emo == "sad":
            smile = VTS_SMILE_SAD
        tilt = 0.5 + (0.18 if emo in ("tender", "happy") else 0.12 if emo in ("surprised", "playful") else 0.0) * (1 if random.random() > 0.5 else -1)
        tilt = max(0.0, min(1.0, 0.5 + (tilt - 0.5) * (VTS_HEAD_TILT_SCALE / 0.18)))
        vals: Dict[str, float] = {}
        if VTS_PARAM_MOUTH_SMILE:
            vals[VTS_PARAM_MOUTH_SMILE] = smile
        if VTS_PARAM_HEAD_Z:
            vals[VTS_PARAM_HEAD_Z] = tilt
        # Optional visibility/tongue controls
        if VTS_PARAM_TONGUE_OUT and emo in ("playful", "laugh"):
            vals[VTS_PARAM_TONGUE_OUT] = 0.85
        if DEBUG_TTS:
            try:
                log_parts = []
                if VTS_PARAM_MOUTH_SMILE:
                    log_parts.append(f"smile={VTS_PARAM_MOUTH_SMILE}:{smile:.2f}")
                if VTS_PARAM_HEAD_Z:
                    log_parts.append(f"tiltZ={VTS_PARAM_HEAD_Z}:{tilt:.2f}")
                if log_parts:
                    print("[VTS] pose " + " | ".join(log_parts))
            except Exception:
                pass
        if vals:
            self.vts.inject(vals, "set")
        # Live2D eye visible toggles (best-effort)
        try:
            if VTS_L2D_EYE_SHOCK_VISIBLE or VTS_L2D_EYE_NORMAL_VISIBLE:
                toggles: Dict[str, float] = {}
                if emo == "surprised" and VTS_L2D_EYE_SHOCK_VISIBLE:
                    toggles[VTS_L2D_EYE_SHOCK_VISIBLE] = 1.0
                    if VTS_L2D_EYE_NORMAL_VISIBLE:
                        toggles[VTS_L2D_EYE_NORMAL_VISIBLE] = 0.0
                elif VTS_L2D_EYE_NORMAL_VISIBLE:
                    toggles[VTS_L2D_EYE_NORMAL_VISIBLE] = 1.0
                    if VTS_L2D_EYE_SHOCK_VISIBLE:
                        toggles[VTS_L2D_EYE_SHOCK_VISIBLE] = 0.0
                if toggles:
                    self.vts.inject(toggles, "set")
        except Exception:
            pass
        # Optional: trigger emotion hotkeys once per utterance
        try:
            hk: Optional[str] = None
            if emo in ("happy", "laugh") and VTS_HK_EYES_LOVE:
                hk = VTS_HK_EYES_LOVE
            elif emo == "sad" and VTS_HK_EYES_CRY:
                hk = VTS_HK_EYES_CRY
            elif emo == "angry" and VTS_HK_SIGN_ANGRY:
                hk = VTS_HK_SIGN_ANGRY
            elif emo == "surprised" and VTS_HK_SIGN_SHOCK:
                hk = VTS_HK_SIGN_SHOCK
            if hk:
                self.vts.trigger_hotkey_once(hk)
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop:
            try:
                text, emotion = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self.engine is not None:
                    self.engine.anticipation()
                if self.state:
                    self.state.set(state="speak", emotion=emotion)
                audio = request_tts_audio_bytes(self.client, text, emotion)
                if self.vts and self.engine:
                    self._apply_emotion_pose(emotion)
                    env_wav = b""
                    if is_wav(audio):
                        env_wav = audio
                    elif is_mp3(audio) and LIPSYNC_MP3_VIA_FFMPEG:
                        env_wav = _decode_mp3_to_wav_bytes_ffmpeg(audio)
                    if env_wav:
                        env = _build_wav_envelope(env_wav, frame_ms=12)
                        if DEBUG_TTS:
                            try:
                                drivers = []
                                if VTS_PARAM_HEAD_Y:
                                    drivers.append(f"headY={VTS_PARAM_HEAD_Y} amp={VTS_HEAD_AMP}")
                                if VTS_PARAM_EYE_X:
                                    drivers.append(f"eyeX={VTS_PARAM_EYE_X} amp={VTS_EYE_AMP}")
                                if VTS_PARAM_EYE_Y:
                                    drivers.append(f"eyeY={VTS_PARAM_EYE_Y} amp={VTS_EYE_AMP*0.8:.2f}")
                                extra = (" | " + ", ".join(drivers)) if drivers else ""
                                print(f"[VTS] env_len={len(env)} mouth={VTS_PARAM_MOUTH} scale={VTS_MOUTH_SCALE}{extra}")
                            except Exception:
                                pass
                        if env:
                            speed_factor = 1.2 if emotion in ("angry", "laugh", "excited", "surprised") else 0.95 if emotion in ("sad", "tender") else 1.0
                            self.vts.drive_mouth(env, frame_ms=16, scale=VTS_MOUTH_SCALE, close_tail=True, speed_factor=speed_factor)
                            if VTS_PARAM_HEAD_Y:
                                self.vts.drive_from_env(VTS_PARAM_HEAD_Y, env, amp=VTS_HEAD_AMP, frame_ms=36)
                            if VTS_PARAM_EYE_X:
                                self.vts.drive_from_env(VTS_PARAM_EYE_X, env, amp=VTS_EYE_AMP, frame_ms=42)
                            if VTS_PARAM_EYE_Y:
                                self.vts.drive_from_env(VTS_PARAM_EYE_Y, env, amp=VTS_EYE_AMP * 0.8, frame_ms=46)
                            if emotion == "laugh":
                                self.engine.laugh_shake()
                _play_bytes_blocking(audio)
            except Exception as e:
                if DEBUG_TTS:
                    print(f"[TTS] error: {e}")
            finally:
                if self.engine is not None:
                    self.engine.follow_through()
                if self.state:
                    self.state.set(state="listen")
                self.queue.task_done()

    # ============
    # Triple-brain
    # ============
    def _run_tts(self) -> None:
        while not self._stop:
            try:
                text, emotion = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self.engine is not None:
                    self.engine.anticipation()
                if self.state:
                    self.state.set(state="speak", emotion=emotion)
                audio = request_tts_audio_bytes(self.client, text, emotion)
                env: list[float] = []
                if self.vts and self.engine:
                    try:
                        self._apply_emotion_pose(emotion)
                    except Exception:
                        pass
                    env_wav = b""
                    if is_wav(audio):
                        env_wav = audio
                    elif is_mp3(audio) and LIPSYNC_MP3_VIA_FFMPEG:
                        env_wav = _decode_mp3_to_wav_bytes_ffmpeg(audio)
                    if env_wav:
                        env = _build_wav_envelope(env_wav, frame_ms=12)
                # enqueue for playback
                try:
                    self._audio_queue.put((audio, env, emotion), timeout=0.1)
                except queue.Full:
                    # drop oldest to keep latency low
                    try:
                        _ = self._audio_queue.get_nowait()
                        self._audio_queue.task_done()
                        self._audio_queue.put((audio, env, emotion), timeout=0.1)
                    except Exception:
                        pass
            except Exception as e:
                if DEBUG_TTS:
                    print(f"[TTS] gen error: {e}")
            finally:
                self.queue.task_done()

    def _run_playback(self) -> None:
        while not self._stop:
            try:
                audio, env, emotion = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self.vts and self.engine and env:
                    speed_factor = 1.2 if emotion in ("angry", "laugh", "excited", "surprised") else 0.95 if emotion in ("sad", "tender") else 1.0
                    self.vts.drive_mouth(env, frame_ms=16, scale=VTS_MOUTH_SCALE, close_tail=True, speed_factor=speed_factor)
                    if VTS_PARAM_HEAD_Y:
                        self.vts.drive_from_env(VTS_PARAM_HEAD_Y, env, amp=VTS_HEAD_AMP, frame_ms=36)
                    if VTS_PARAM_EYE_X:
                        self.vts.drive_from_env(VTS_PARAM_EYE_X, env, amp=VTS_EYE_AMP, frame_ms=42)
                    if VTS_PARAM_EYE_Y:
                        self.vts.drive_from_env(VTS_PARAM_EYE_Y, env, amp=VTS_EYE_AMP * 0.8, frame_ms=46)
                    if emotion == "laugh":
                        self.engine.laugh_shake()
                _play_bytes_blocking(audio)
            except Exception as e:
                if DEBUG_TTS:
                    print(f"[TTS] play error: {e}")
            finally:
                if self.engine is not None:
                    self.engine.follow_through()
                if self.state:
                    self.state.set(state="listen")
                try:
                    self._audio_queue.task_done()
                except Exception:
                    pass

    def stop(self) -> None:
        self._stop = True
        try:
            if self._use_pipeline:
                try:
                    self._tts_thread.join(timeout=1)
                except Exception:
                    pass
                try:
                    self._play_thread.join(timeout=1)
                except Exception:
                    pass
            else:
                self.thread.join(timeout=1)
        except Exception:
            pass


class InputThread(threading.Thread):
    def __init__(self, out_queue: "queue.Queue[str]") -> None:
        super().__init__(daemon=True)
        self.out_queue = out_queue
        self._stop_flag = False

    def run(self) -> None:
        while not self._stop_flag:
            try:
                line = input("あなた: ").strip()
                self.out_queue.put(line)
            except EOFError:
                break

    def stop(self) -> None:
        self._stop_flag = True


def _parse_name_and_tail(s: str) -> Tuple[str, str]:
    s = s.strip()
    if not s:
        return "", ""
    if s[0] in ('"', "'"):
        q = s[0]
        end = s.find(q, 1)
        if end != -1:
            name = s[1:end]
            tail = s[end + 1 :].strip()
            return name, tail
    parts = s.split(" ", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1].strip()


def extract_topics(text: str) -> List[str]:
    topics: List[str] = []
    for t in T.tokenize(text):
        if t.part_of_speech.split(",")[0] == "名詞":
            w = t.surface.strip()
            if w and w not in JP_STOP and len(w) <= 16:
                topics.append(w)
    return topics[:8]


def decay_topics(memory: dict) -> None:
    now = time.time()
    last = float(memory.get("last_decay_ts", now))
    dt = max(0.0, now - last)
    half = max(1.0, DECAY_HALF_LIFE_MIN) * 60.0
    f = 2 ** (-dt / half)
    tc = memory.get("topic_counts", {})
    for k in list(tc.keys()):
        tc[k] = float(tc[k]) * f
        if tc[k] < 0.05:
            del tc[k]
    memory["topic_counts"] = tc
    memory["last_decay_ts"] = now


def update_topics(memory: dict, topics: List[str]) -> None:
    decay_topics(memory)
    tc = memory.get("topic_counts", {})
    for w in topics:
        tc[w] = float(tc.get(w, 0.0)) + 1.0
    memory["topic_counts"] = tc


def push_recent_auto(memory: dict, kind: str, text: str, limit: int = 10) -> None:
    ra = memory.get("recent_auto_texts", [])
    ra.append({"kind": kind, "ts": time.time(), "text": text})
    memory["recent_auto_texts"] = ra[-limit:]


def too_many_tales(memory: dict) -> bool:
    ra = memory.get("recent_auto_texts", [])
    items = [r for r in ra if isinstance(r, dict) and r.get("kind") in ("tale", "question")]
    last5 = items[-5:]
    ratio = (sum(1 for r in last5 if r.get("kind") == "tale") / max(1, len(last5))) if last5 else 0.0
    last60 = [r for r in items if (time.time() - float(r.get("ts", 0))) <= 60 and r.get("kind") == "tale"]
    return ratio > SELF_TALE_MAX_RATIO_5 or len(last60) > SELF_TALE_MAX_PER_MIN


AUTO_TEMPLATES = [
    "そういえば{topic}の最近のハイライト教えてほしいな！",
    "{topic}で嬉しかったこと、ひとつ聞きたい〜！",
    "{topic}の小さな失敗談もOK！学びがあったら共有して！",
    "今の気分の{topic}に10点満点で点数つけるなら？理由もぜひ！",
    "{topic}以外でも、最近ハマってることある？気になる〜！",
    "少し休憩しよ。今日の{topic}で一番印象に残った瞬間は？",
]


def top_topics(memory: dict, k: int = 3) -> List[str]:
    tc = memory.get("topic_counts", {})
    if not tc:
        return []
    return [t for t, _ in sorted(tc.items(), key=lambda x: x[1], reverse=True)[:k]]


def choose_next_topic(memory: dict) -> str:
    cands = top_topics(memory, 5)
    if not cands:
        return "今日"
    ring = memory.get("topic_ring", [])
    for t in cands:
        if t not in ring:
            ring.append(t)
            memory["topic_ring"] = ring[-8:]
            return t
    t = random.choice(cands)
    ring.append(t)
    memory["topic_ring"] = ring[-8:]
    return t


def build_context_snippet(memory: dict) -> str:
    tops = top_topics(memory, 3)
    samples = memory.get("user_samples", [])[-5:]
    body = " | ".join(samples) if samples else ""
    return (f"上位話題: {', '.join(tops) if tops else 'なし'}\n直近コメント例: {body}")[:320]


def generate_auto_question(client: OpenAI, memory: dict) -> str:
    topic = choose_next_topic(memory)
    ra = set(r["text"] for r in memory.get("recent_auto_texts", []))
    cands = [tmpl.format(topic=topic) for tmpl in AUTO_TEMPLATES if tmpl.format(topic=topic) not in ra] or [
        "最近あったちょっと嬉しいこと、教えてほしいな！",
        "今の気分で話したいテーマある？雑談でもOKだよ！",
        "今日の一番のハイライトは何だった？共有してほしい〜！",
    ]
    context = build_context_snippet(memory)
    ask_suppress = time.time() < float(memory.get("ask_suppress_until", 0.0))
    system = (
        "あなたは生配信中のAI。1文・最大80文字・絵文字1つ以内で、視聴者に自然に『話題を提示』する短い独り言を作る。"
        + ("自己言及・タグは禁止。疑問符は使わない。" if ask_suppress else "自己言及・タグは禁止。疑問符は最小限。")
    )
    user = f"コンテキスト:\n{context}\n候補:\n- " + "\n- ".join(cands) + "\n条件: 1文/<=80文字/絵文字1つ以内/自然/タグなし"
    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.6,
        max_tokens=120,
    )
    return (res.choices[0].message.content or "").strip()


def generate_self_tale(client: OpenAI, memory: dict) -> str:
    topic = choose_next_topic(memory)
    profile = (memory or {}).get("profile", "")
    user_samples = " / ".join(memory.get("user_samples", [])[-3:])
    system = (
        "あなたは生配信中のAI。作り話を1〜2文・80文字以内・絵文字1つ以内で、現実味を持って可愛く語る。夢/空想/妄想は禁止。自己言及も禁止。"
    )
    user = f"キーワード:{topic}\nユーザーの雰囲気:{user_samples}\nプロフィール:{profile}\n条件:1〜2文/<=80文字/絵文字1つ以内/自然/タグなし"
    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.9,
        max_tokens=120,
    )
    return (res.choices[0].message.content or "").strip()


def is_blacklisted_tale(txt: str) -> bool:
    lw = "".join(txt.split())
    return any(kw in lw for kw in SELF_TALE_BLACKLIST)


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.last_event_ts = time.time()
        self.stop = False
        self.auto_enabled = True

    def bump(self) -> None:
        with self._lock:
            self.last_event_ts = time.time()

    def should_mention(self, min_sec: float) -> bool:
        with self._lock:
            return (time.time() - self.last_event_ts) >= min_sec

    def set_auto(self, enabled: bool) -> None:
        with self._lock:
            self.auto_enabled = enabled
            self.last_event_ts = time.time()


class MentionWorker(threading.Thread):
    def __init__(self, client: OpenAI, memory: dict, chat_history: List[Dict[str, str]], tts_worker: "TTSWorker|None", shared: SharedState) -> None:
        super().__init__(daemon=True)
        self.client = client
        self.memory = memory
        self.chat_history = chat_history
        self.tts_worker = tts_worker
        self.shared = shared

    def run(self) -> None:
        if not MENTION_ENABLED:
            return
        self.memory.setdefault("user_samples", [])
        self.memory.setdefault("mentioned_ids", [])
        while not self.shared.stop:
            wait_s = MENTION_MIN_SEC + random.uniform(0, max(1, MENTION_JITTER_SEC))
            for _ in range(int(wait_s * 2)):
                if self.shared.stop:
                    return
                time.sleep(0.5)
            if not self.shared.auto_enabled:
                continue
            if not self.shared.should_mention(MENTION_MIN_SEC):
                continue
            cands = [c for c in self.memory.get("user_samples", [])[-12:] if c and c not in self.memory["mentioned_ids"]]
            if not cands:
                continue
            target = random.choice(cands)
            try:
                text = generate_auto_question(self.client, self.memory)
                if not text:
                    continue
                print(f"もも: {text}")
                self.chat_history += [{"role": "assistant", "content": text}]
                if self.tts_worker is not None and TTS_ENABLED:
                    emo = get_sentiment(target)
                    self.tts_worker.enqueue(sanitize_for_tts(text), emo)
                self.memory["mentioned_ids"].append(target)
                self.memory["mentioned_ids"] = self.memory["mentioned_ids"][-50:]
                save_memory(MEMORY_FILE, self.memory)
                self.shared.bump()
            except Exception:
                continue


def trigger_controls_from_comment(user_text: str, memory: dict) -> None:
    if any(kw in user_text for kw in ASK_SUPPRESS_KEYWORDS):
        memory["ask_suppress_until"] = time.time() + ASK_SUPPRESS_MIN * 60
        memory["ask_suppress_hits"] = float(memory.get("ask_suppress_hits", 0.0)) + 1.0
    if any(kw in user_text for kw in ASK_ALLOW_KEYWORDS):
        memory["ask_suppress_until"] = 0.0


def compute_auto_self_pct(memory: dict) -> float:
    pct = AUTO_SELF_TALE_PCT_BASE
    now = time.time()
    if float(memory.get("ask_suppress_until", 0.0)) > now:
        return 1.0
    if ADAPTIVE_AUTO_SELF:
        hits = float(memory.get("ask_suppress_hits", 0.0))
        pct = min(0.95, pct + min(0.4, 0.15 + 0.05 * hits))
    if too_many_tales(memory):
        pct = 0.05
    return max(0.05, min(0.95, pct))


def perform_laugh_burst(tts_worker: Optional["TTSWorker"], count: int) -> None:
    lines_big = ["あははは！", "あっはっは！", "わはは！"]
    lines_soft = ["ふふっ…！", "くすくす…！", "えへへ…！"]
    lines_shy = ["えへへ…照れるね。", "にへへ…", "うふふ…"]
    for _ in range(max(1, count)):
        text = random.choice(lines_big + lines_soft + lines_shy)
        print(f"もも: {text}")
        if tts_worker is not None and TTS_ENABLED:
            tts_worker.enqueue(sanitize_for_tts(text), "laugh")
    endcap = "えへへ…ちょっと照れちゃった。もう笑わなくて平気？"
    print(f"もも: {endcap}")
    if tts_worker is not None and TTS_ENABLED:
        tts_worker.enqueue(sanitize_for_tts(endcap), "tender")


def want_laugh_sequence(user_text: str) -> bool:
    return any(k in user_text for k in ["笑って", "笑わせて", "草", "www", "🤣", "😂"])


def run_chat_loop() -> None:
    ensure_env_present()
    client = openai_client()
    vts = VTSClient(VTS_URL, VTS_PLUGIN_NAME, VTS_PLUGIN_DEV, VTS_PARAM_MOUTH) if VTS_ENABLED else None
    engine = MotionEngine(vts) if VTS_ENABLED else None
    state = StateManager()
    tts_worker = TTSWorker(client if TTS_PROVIDER == "OPENAI" else None, vts, engine, state) if TTS_ENABLED else None
    planner = ActionPlanner(vts) if (PLANNER_ENABLED and VTS_ENABLED) else None
    memory = load_memory(MEMORY_FILE) if MEMORY_ENABLED else {
        "profile": "",
        "topic_counts": {},
        "recent_auto_texts": [],
        "user_samples": [],
        "mentioned_ids": [],
        "last_auto_ts": 0.0,
        "last_auto_kind": "",
        "ask_suppress_until": 0.0,
        "ask_suppress_hits": 0.0,
        "last_decay_ts": time.time(),
        "topic_ring": [],
        "motion_history": [],
    }
    chat_history: List[Dict[str, str]] = []
    print("可愛い音声チャットを開始するよ。終了するには q / !scan / !params / !vts / !hklist / !plan / pause / resume")
    eye_ctrl = EyeController(vts, state, engine) if VTS_ENABLED else None
    if eye_ctrl:
        eye_ctrl.start()
    last_comment_time = time.time()
    in_q: "queue.Queue[str]" = queue.Queue(maxsize=64)
    in_thread = InputThread(in_q)
    in_thread.start()
    shared = SharedState()
    mention_worker = MentionWorker(client, memory, chat_history, tts_worker, shared)
    mention_worker.start()
    hotkey_delay_ms = HOTKEY_DELAY_MS
    if VTS_HK_HELLO and VTS_ENABLED:
        try:
            vts.trigger_hotkey(VTS_HK_HELLO)
        except Exception:
            pass
    try:
        while True:
            try:
                user_text = in_q.get(timeout=0.5)
                if not user_text:
                    continue
                low = user_text.lower()
                state.set(state="listen")
                if low in {"q", "quit", "exit"} or user_text in {"ｑ", "終了", "終わり"}:
                    if VTS_HK_BYE and VTS_ENABLED:
                        try:
                            vts.trigger_hotkey(VTS_HK_BYE)
                        except Exception:
                            pass
                    print("またね！")
                    break
                if low in {"pause", "auto off", "自動停止", "停止"}:
                    shared.set_auto(False)
                    print("自動発話を停止したよ。")
                    continue
                if low in {"resume", "auto on", "自動開始", "再開"}:
                    shared.set_auto(True)
                    print("自動発話を再開するね！")
                    continue
                if low in {"!scan"}:
                    if vts is None:
                        print("[SCAN] VTS無効です(.envのVTS_ENABLED=true)")
                        continue
                    path = vts.write_env_auto()
                    print(f"[SCAN] 推奨マッピングを書き出しました: {path}（.envに反映して再起動してください）")
                    continue
                if low in {"!params"}:
                    if vts is None:
                        print("[PARAMS] VTS無効です (.envのVTS_ENABLED=true にして、VTube StudioのAPIを有効にしてね)")
                        continue
                    names = vts.list_params()
                    if names:
                        print("[PARAMS] " + ", ".join(names))
                    else:
                        st = vts.status()
                        print(f"[PARAMS] 取得できませんでした / connected={st['connected']} url={st['url']} err={st['last_error']}")
                    continue
                if low in {"!inparams"}:
                    if vts is None:
                        print("[IN] VTS無効です (.envのVTS_ENABLED=true)")
                        continue
                    names = vts.list_in_params()
                    if names:
                        print("[IN] " + ", ".join(names))
                    else:
                        st = vts.status()
                        msg = st['last_error'] or "InputParameterList の取得に失敗しました"
                        print(f"[IN] 取得できませんでした / connected={st['connected']} url={st['url']} err={msg}")
                    continue
                if low.startswith("!hotkey ") or low.startswith("!hk "):
                    if vts is None:
                        print("[VTS] 無効です (.envでVTS_ENABLED=true)")
                        continue
                    hk = user_text.split(" ", 1)[1].strip()
                    if not hk:
                        print("[VTS] usage: !hotkey <name>")
                        continue
                    ok = vts.trigger_hotkey(hk)
                    print(f"[VTS] hotkey '{hk}' -> {ok}")
                    continue
                if low in {"!hklist", "!hotkeys"}:
                    if vts is None:
                        print("[VTS] 無効です (.envでVTS_ENABLED=true)")
                        continue
                    scanned = vts.list_hotkeys()
                    if scanned:
                        print("[VTS] hotkeys:")
                        for h in scanned:
                            print(f"  - {h.get('name','')}  id={h.get('id','')}  type={h.get('type','')}")
                    else:
                        if VTS_HOTKEY_LIST:
                            print("[VTS] hotkeys (env): " + ", ".join(VTS_HOTKEY_LIST))
                        else:
                            print("[VTS] hotkeys: なし（モデルにホットキー未設定）")
                    continue
                if low.startswith("!macro "):
                    if vts is None:
                        print("[VTS] 無効です (.envでVTS_ENABLED=true)")
                        continue
                    args = user_text.split(" ", 1)[1].strip()
                    names = [s.strip() for s in args.split(",") if s.strip()] if args else VTS_HOTKEY_LIST
                    if not names:
                        print("[VTS] usage: !macro HK1,HK2,...  または .env の VTS_HOTKEY_LIST を設定")
                        continue
                    ok = vts.trigger_hotkeys_sequence(names, hotkey_delay_ms)
                    print(f"[VTS] macro {names} delay={hotkey_delay_ms}ms -> {ok}")
                    continue
                if low.startswith("!hkdelay "):
                    arg = user_text.split(" ", 1)[1].strip()
                    try:
                        ms = int(arg)
                        hotkey_delay_ms = max(0, ms)
                        print(f"[VTS] hotkey delay set to {hotkey_delay_ms}ms")
                    except Exception:
                        print("[VTS] usage: !hkdelay <milliseconds>")
                    continue
                if low.startswith("!set "):
                    if vts is None:
                        print("[VTS] 無効です (.envでVTS_ENABLED=true)")
                        continue
                    try:
                        _, rest = user_text.split(" ", 1)
                        pid, tail = _parse_name_and_tail(rest)
                        sval = tail.split(" ", 1)[0]
                        ok = vts.set_param(pid, float(sval))
                        print(f"[VTS] set {pid}={sval} -> {ok}")
                    except Exception:
                        print("[VTS] usage: !set <ParamName> <0..1>")
                    continue
                if low.startswith("!anim "):
                    if vts is None:
                        print("[VTS] 無効です (.envでVTS_ENABLED=true)")
                        continue
                    try:
                        _, rest = user_text.split(" ", 1)
                        pid, tail = _parse_name_and_tail(rest)
                        nums = tail.split()
                        if not pid or len(nums) < 4:
                            print("[VTS] usage: !anim <ParamName> <start> <end> <duration_ms> [fps]")
                            continue
                        s = float(nums[0])
                        e = float(nums[1])
                        dur = int(nums[2])
                        fps = int(nums[3]) if len(nums) > 3 else 60
                        ok = vts.animate_param(pid, s, e, dur, fps)
                        print(f"[VTS] anim {pid} {s}->{e} {dur}ms {fps}fps -> {ok}")
                    except Exception:
                        print("[VTS] usage: !anim <ParamName> <start> <end> <duration_ms> [fps]")
                    continue
                if low.startswith("!vts"):
                    if vts is None:
                        print("[VTS] 無効です (.envでVTS_ENABLED=true)")
                        continue
                    parts = low.split()
                    if len(parts) == 1 or parts[1] == "status":
                        st = vts.status()
                        print(f"[VTS] connected={st['connected']} url={st['url']} token={st['token_file']} err={st['last_error']}")
                    elif parts[1] == "reconnect":
                        vts.stop()
                        time.sleep(0.5)
                        vts = VTSClient(VTS_URL, VTS_PLUGIN_NAME, VTS_PLUGIN_DEV, VTS_PARAM_MOUTH)
                        print("[VTS] 再接続を試みています…")
                    else:
                        print("[VTS] usage: !vts [status|reconnect]")
                    continue
                if low.startswith("!plan "):
                    if planner is None or engine is None:
                        print("[PLAN] プランナー無効です(.envでPLANNER_ENABLED=true & VTS_ENABLED=true)")
                        continue
                    comment = user_text.split(" ", 1)[1].strip()
                    plan = planner.plan(comment, use_llm=PLANNER_USE_LLM)
                    planner.execute(plan, engine)
                    continue

                state.set(state="think")
                emo_user = get_sentiment(user_text)
                trigger_controls_from_comment(user_text, memory)
                if LAUGH_SEQUENCE_ENABLED and want_laugh_sequence(user_text):
                    perform_laugh_burst(tts_worker, random.randint(2, LAUGH_BURST_MAX))
                msgs = build_messages(SYSTEM_PROMPT, memory, chat_history)
                msgs.append({"role": "user", "content": user_text})
                try:
                    reply = generate_reply(OpenAI(), msgs)
                except Exception as e:
                    print(f"[ERROR] 返答生成に失敗しました: {e}")
                    continue
                reply = reply.strip()
                print(f"もも: {reply}")
                chat_history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]
                if MEMORY_ENABLED:
                    try:
                        update_topics(memory, extract_topics(user_text))
                        us = memory.get("user_samples", [])
                        us.append(user_text)
                        memory["user_samples"] = us[-12:]
                        memory = update_memory_summary(OpenAI(), memory, user_text, reply, MEMORY_MAX_CHARS)
                        save_memory(MEMORY_FILE, memory)
                    except Exception:
                        pass
                tempo = BASE_TEMPO
                hour = time.localtime().tm_hour
                night = hour >= 22 or hour <= 6
                if night:
                    tempo *= NIGHT_SLOW
                state.set(tempo=tempo, night=night, emotion=emo_user)
                if TTS_ENABLED and tts_worker is not None:
                    tts_worker.enqueue(sanitize_for_tts(reply), emo_user)
                if PLANNER_ENABLED and planner is not None and engine is not None:
                    try:
                        plan = planner.plan(user_text, use_llm=PLANNER_USE_LLM)
                        planner.execute(plan, engine)
                    except Exception:
                        pass
                last_comment_time = time.time()
                memory["last_auto_ts"] = last_comment_time
                shared.bump()
            except queue.Empty:
                if shared.auto_enabled and (time.time() - last_comment_time) >= IDLE_AUTO_TALK_SEC:
                    auto_self_pct = compute_auto_self_pct(memory)
                    do_self = random.random() < auto_self_pct
                    if not do_self and memory.get("last_auto_kind") == "question":
                        do_self = True
                    if time.time() < float(memory.get("ask_suppress_until", 0.0)):
                        do_self = True
                    state.set(state="idle")
                    force_question = False
                    if do_self:
                        reply = generate_self_tale(OpenAI(), memory)
                        if SELF_TALE_BLACKLIST and is_blacklisted_tale(reply):
                            reply = generate_self_tale(OpenAI(), memory)
                            if SELF_TALE_BLACKLIST and is_blacklisted_tale(reply):
                                force_question = True
                        memory["last_auto_kind"] = "tale"
                    if force_question or not do_self:
                        reply = generate_auto_question(OpenAI(), memory)
                        memory["last_auto_kind"] = "question"
                    reply = reply.strip()
                    print(f"もも: {reply}")
                    chat_history += [{"role": "assistant", "content": reply}]
                    if MEMORY_ENABLED:
                        try:
                            push_recent_auto(memory, memory["last_auto_kind"], reply)
                            memory = update_memory_summary(OpenAI(), memory, "[AUTO]", reply, MEMORY_MAX_CHARS)
                            save_memory(MEMORY_FILE, memory)
                        except Exception:
                            pass
                    if TTS_ENABLED and tts_worker is not None:
                        emo = "happy"
                        if memory.get("user_samples"):
                            emo = get_sentiment(" ".join(memory["user_samples"][-3:])) or "happy"
                        tts_worker.enqueue(sanitize_for_tts(reply), emo)
                    last_comment_time = time.time()
                    memory["last_auto_ts"] = last_comment_time
                    shared.bump()
    finally:
        if tts_worker is not None:
            tts_worker.stop()
        if eye_ctrl is not None:
            eye_ctrl.stop()
        in_thread.stop()
        shared.stop = True
        if VTS_ENABLED and vts is not None:
            vts.stop()


if __name__ == "__main__":
    run_chat_loop()