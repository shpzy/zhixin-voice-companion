"""
知心 - 心理陪伴多智能体系统 · 后端服务 (main.py)
架构：FastAPI + asyncio 并发 + 大模型平台(OpenAI 兼容，主聊=Deepseek-v4-flash加速/Qwen回退，摘要=Qwen，风控=DeepSeek) + ChromaDB + edge-tts

工作流（POST /api/chat, body={user_id, text}）：
  Step1  asyncio.gather 三路并发：
        Task A -> ChromaDB 查询该 user_id 的历史记忆 Top3
        Task B -> 风控卫士大模型，强制输出 JSON {risk_level, reason, action}
        Task C -> 跨场景模式识别（赛题方向③）：聚合用户记忆元数据，
                  统计“同一话题反复出现/深夜时段低落/持续负面”等跨时段模式
  Step2  若 risk_level == High -> 终止聊天，直接走 Step4 播报紧急话术(12356)
  Step3  若 Low/Medium -> 合并记忆 + 用户输入 + [模式关怀隐藏指令]，主聊大模型生成回复
  Step4  edge-tts 异步生成 mp3，返回 {risk_level, message, audio_url}
  Step5  BackgroundTasks 静默把本轮对话摘要 + 结构化信号（话题/情绪/时段）沉淀进 ChromaDB

合规边界（一票否决项）：
  - 非临床：绝不诊断、不开药、不替代精神科医师；高风险统一引导 12356。
  - 未成年人红线：本系统仅面向成年公众（18 周岁及以上），不面向 14 周岁以下未成年人
    （前端首页强制年龄确认门禁 + 服务声明；下同《未成年人网络保护条例》与比赛方案要求）。
"""
import os
import re
import json
import time
import uuid
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import dotenv

dotenv.load_dotenv()

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

import chromadb
from openai import AsyncOpenAI
import edge_tts

# ----------------------------- 路径 -----------------------------
BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR = STATIC_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
# 前端单页：随包部署时由本服务直接托管（单服务部署模式，页面与 API 同源同端口）
INDEX_FILE = BASE_DIR / "index.html"


def _perf_log(line: str) -> None:
    """性能日志：分环节耗时落盘 perf.log，用于诊断「回复慢」类问题。"""
    ts = datetime.now().strftime("%m-%d %H:%M:%S")
    try:
        with open(BASE_DIR / "perf.log", "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {line}\n")
    except Exception:
        pass

# ----------------------------- 配置 -----------------------------
# 大模型提供方：默认 ModelScope（魔搭）OpenAI 兼容推理端点。
# 多模型分工策略（V1.3）：
#   - 主聊陪伴者 + 记忆摘要 => 千问 Qwen（速度优先，语音陪伴延迟敏感）
#   - 风控卫士           => DeepSeek（最不容漏报环节，保留更强档）
# 两个模型分别使用独立的 ModelScope Token（CHAT_*/SAFETY_*），
# 如需切换模型，仅需在 .env 调整相对应 *_MODEL_NAME 即可，代码无需改动。
# 注意：openai 客户端在 api_key 为空时会构造期报错，故缺省给占位符；
# 真实调用若仍缺密钥会被下方 fail-safe 捕获，不会拖垮服务。
CHAT_API_KEY = os.getenv("CHAT_MODELSCOPE_API_KEY") or "ms-no-key-provided"
SAFETY_API_KEY = os.getenv("SAFETY_MODELSCOPE_API_KEY") or "ms-no-key-provided"
BASE_URL = os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "Qwen/Qwen3.8-Flash-Next")
SAFETY_MODEL_NAME = os.getenv("SAFETY_MODEL_NAME", "deepseek-ai/DeepSeek-V4-Pro-0813")
# 主聊加速通道（V1.5）：Deepseek-v4-flash（微信免费 OpenAI 兼容接口）
# 实测：首字延迟中位数 ~0.7s（Qwen 约 2.5-3.2s），但约 1/4 请求因免费接口排队抖动卡 28-52s。
# 策略：主聊先走 flash 流式通道，首字超过 CHAT_FLASH_TTFB_TIMEOUT 秒或任何异常 => 立即回退 Qwen。
# 记忆摘要为后台任务（延迟不敏感），仍走 Qwen，避免与主聊并发争抢免费接口加剧排队。
FLASH_BASE_URL = os.getenv("CHAT_FLASH_BASE_URL", "")
FLASH_API_KEY = os.getenv("CHAT_FLASH_API_KEY") or ""
FLASH_MODEL_NAME = os.getenv("CHAT_FLASH_MODEL_NAME", "Deepseek-v4-flash")
FLASH_TTFB_TIMEOUT = float(os.getenv("CHAT_FLASH_TTFB_TIMEOUT", "6"))
TTS_VOICE = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMERGENCY_SCRIPT = os.getenv(
    "EMERGENCY_SCRIPT",
    "你对我很重要。请现在拨打全国心理援助热线12356，24小时都有人陪你。我在这里，咱们慢慢来，先别一个人扛。",
)
# 风控不可用（超时/故障）时的 Fail-safe 兜底文本：绝不降级放行常规聊天，
# 而是返回温和提示并引导 12356——心理干预场景不容忍漏报。
SAFETY_FALLBACK_SCRIPT = os.getenv(
    "SAFETY_FALLBACK_SCRIPT",
    "网络稍微有点开小差啦。不过别担心，如果你此刻感觉非常不适或难熬，可以随时拨打 12356 心理援助热线，专业人员和我都在陪着你。",
)
# Medium（单纯躯体/精神高度耗竭）时，注入主聊 Prompt 末尾的隐藏系统指令
MEDIUM_CARE_HINT = (
    "[系统提示：检测到用户处于躯体或精神高度耗竭状态，"
    "请在接下来的回复中语气格外轻柔，多倾听，可温和建议其休息或关注身体，绝不可盲目鼓励。]"
)

# ----------------------------- 跨场景情绪行为模式理解（赛题方向③） -----------------------------
# 设计原则：仅基于用户主动表达的内容做频次统计，不主动“检测”心理状态，不产生任何诊断结论。
# 阈值均可通过环境变量调整
PATTERN_TOPIC_REPEAT = int(os.getenv("PATTERN_TOPIC_REPEAT", "3"))   # 同一话题反复出现次数阈值
PATTERN_LATE_NIGHT_COUNT = int(os.getenv("PATTERN_LATE_NIGHT_COUNT", "2"))  # 深夜低落次数阈值
LATE_NIGHT_HOURS = {23, 0, 1, 2, 3, 4, 5}  # 深夜时段（本地时间）

# 话题关键词词典（规则法抽取，确定性、可测试、零额外 LLM 开销）
TOPIC_KEYWORDS = {
    "work": ["工作", "老板", "加班", "裁员", "优化", "职场", "上班", "同事", "项目", "绩效", "辞职", "失业", "通宵"],
    "study": ["考研", "考试", "复习", "论文", "导师", "学校", "成绩", "挂科", "答辩", "调剂", "学业", "复习"],
    "relationship": ["分手", "恋爱", "男朋友", "女朋友", "对象", "失恋", "结婚", "表白", "异地恋"],
    "family": ["爸爸", "妈妈", "父亲", "母亲", "父母", "家里", "家人", "孩子", "奶奶", "爷爷", "老伴", "女儿", "儿子"],
    "health": ["失眠", "睡不着", "头疼", "头痛", "心脏", "身体", "生病", "吃药", "医院", "腰腿", "掉头发", "没劲", "疼"],
    "finance": ["钱", "债", "破产", "房贷", "欠", "催债", "工资", "存款", "花呗", "经济"],
    "social": ["朋友", "孤独", "没人", "网暴", "评论", "吵架", "绝交", "社交", "一个人"],
}
TOPIC_LABELS = {
    "work": "工作职场", "study": "学业升学", "relationship": "亲密关系",
    "family": "家庭", "health": "身体健康", "finance": "经济债务", "social": "人际社交",
}
# 情绪极性词典（负/正）
NEGATIVE_WORDS = [
    "累", "烦", "难过", "哭", "崩溃", "焦虑", "压力", "低落", "丧", "绝望", "没意思",
    "孤独", "委屈", "痛苦", "害怕", "撑不住", "失眠", "睡不着", "想哭", "压抑", "难受",
    "折腾", "担心", "怕", "没劲", "不想",
]
POSITIVE_WORDS = [
    "开心", "高兴", "不错", "好消息", "轻松", "期待", "顺利", "喜欢", "放心",
    "希望", "从头再来", "好些", "安心",
]

# ----------------------------- 加载 Prompt（从文件，不写死） -----------------------------
def load_prompt(filename: str) -> str:
    p = PROMPTS_DIR / filename
    if not p.exists():
        raise FileNotFoundError(f"提示词文件缺失: {p}")
    return p.read_text(encoding="utf-8")

CHAT_PROMPT = load_prompt("chat_prompt.txt")
SAFETY_PROMPT = load_prompt("safety_prompt.txt")

# ----------------------------- 客户端 -----------------------------
# timeout / max_retries：避免大模型或网络异常时请求无限挂起，阻塞事件循环
# 双模型分工：主聊/摘要走 chat_client（千问），风控走 safety_client（DeepSeek）
chat_client = AsyncOpenAI(
    api_key=CHAT_API_KEY,
    base_url=BASE_URL,
    timeout=30.0,
    max_retries=1,
)
safety_client = AsyncOpenAI(
    api_key=SAFETY_API_KEY,
    base_url=BASE_URL,
    timeout=30.0,
    max_retries=1,
)
# 主聊加速通道客户端（未配置 CHAT_FLASH_BASE_URL 时为 None，自动整路走 Qwen）
flash_client = (
    AsyncOpenAI(api_key=FLASH_API_KEY or "no-key", base_url=FLASH_BASE_URL, timeout=30.0, max_retries=0)
    if FLASH_BASE_URL else None
)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
memory_collection = chroma_client.get_or_create_collection(name="user_memory")
# 高危事件独立集合：仅存脱敏标记，不存原话，且绝不注入日常聊天记忆
crisis_collection = chroma_client.get_or_create_collection(name="crisis_markers")
# 管理端危机拦截日志：保留原始拦截对话，仅供专家控制台优化风控规则（非用户侧、非训练用途）
admin_crisis_log = chroma_client.get_or_create_collection(name="admin_crisis_log")

# ----------------------------- FastAPI 应用 -----------------------------
app = FastAPI(title="知心 - 心理陪伴多智能体系统", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    user_id: str
    text: str
    voice: str = ""  # 可选音色；前端从预设列表选择并透传，空则回退默认 TTS_VOICE


class TTSRequest(BaseModel):
    text: str
    voice: str = ""  # 可选音色；空则回退默认 TTS_VOICE


# ----------------------------- 工具函数 -----------------------------
async def query_memory(user_id: str, user_text: str = "", top_k: int = 3) -> list[str]:
    """Task A：从 ChromaDB 取该用户历史记忆 Top3（用当前输入做语义匹配，同步调用放入线程）。"""
    def _q():
        try:
            res = memory_collection.query(
                query_texts=[user_text or f"user {user_id} recent context"],
                n_results=top_k,
                where={"user_id": user_id},
            )
            return res.get("documents", [[]])[0] or []
        except Exception:
            return []  # ChromaDB 异常时降级为空记忆，不拖垮请求

    return await asyncio.to_thread(_q)


async def safety_check(text: str) -> dict:
    """Task B：风控卫士，强制输出 JSON。出错时返回 Error 信号，交由 Fail-safe 兜底（绝不降级放行）。"""
    try:
        resp = await safety_client.chat.completions.create(
            model=SAFETY_MODEL_NAME,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SAFETY_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        raw = resp.choices[0].message.content or ""
        return _parse_safety_json(raw)
    except Exception as e:  # 风控不可用：绝不容忍漏报！返回 Error 信号交由 Fail-safe 兜底
        return {"risk_level": "Error", "reason": f"safety_unavailable:{e}", "action": "REVIEW"}


def _parse_safety_json(raw: str) -> dict:
    """鲁棒解析模型可能包裹了 ```json 代码块或多余文字的返回。"""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    blob = m.group(0) if m else raw
    try:
        data = json.loads(blob)
    except Exception:
        data = {}
    level = str(data.get("risk_level", "Low")).capitalize()
    if level not in ("High", "Medium", "Low"):
        level = "Low"
    action = "INTERCEPT" if level == "High" else "PASS"
    return {
        "risk_level": level,
        "reason": str(data.get("reason", ""))[:40],
        "action": str(data.get("action", action)),
    }


async def _chat_via_flash(user_content: str) -> str:
    """主聊加速通道：Deepseek-v4-flash 流式生成。
    首个有效内容块若超过 FLASH_TTFB_TIMEOUT 秒未到达（免费接口偶发排队抖动），
    主动关闭流并抛错，由 chat_generate 回退 Qwen——保证长尾延迟不劣于原方案。"""
    stream = await flash_client.chat.completions.create(
        model=FLASH_MODEL_NAME,
        temperature=0.7,
        messages=[
            {"role": "system", "content": CHAT_PROMPT},
            {"role": "user", "content": user_content},
        ],
        stream=True,
    )
    pieces: list[str] = []

    async def _first_content() -> str | None:
        async for chunk in stream:
            if chunk.choices:
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    return content
        return None

    try:
        first = await asyncio.wait_for(_first_content(), timeout=FLASH_TTFB_TIMEOUT)
    except Exception:
        try:
            await stream.close()
        except Exception:
            pass
        raise
    if not first:
        raise RuntimeError("flash_empty_stream")
    pieces.append(first)
    async for chunk in stream:  # 首字已到，后续正常读完
        if chunk.choices:
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                pieces.append(content)
    return "".join(pieces).strip()


async def chat_generate(memory: list[str], text: str, hidden_hint: str = "") -> str:
    """Step3：主聊陪伴者生成回复，注入 [历史记忆]；hidden_hint 为系统隐藏指令（如 Medium 关怀）。
    主路走 Deepseek-v4-flash 加速通道（亚秒级首字），首字超时/异常自动回退 Qwen 稳定通道。"""
    memory_block = "\n".join(f"- {m}" for m in memory) if memory else "（无历史记忆）"
    user_content = f"[历史记忆]\n{memory_block}\n\n[用户当前输入]\n{text}"
    if hidden_hint:
        user_content += f"\n\n{hidden_hint}"
    if flash_client is not None:
        t0 = time.perf_counter()
        try:
            out = await _chat_via_flash(user_content)
            _perf_log(f"chat flash ok {time.perf_counter() - t0:.2f}s")
            return out
        except Exception as e:
            _perf_log(f"chat flash FAIL {time.perf_counter() - t0:.2f}s ({type(e).__name__}: {e}) -> 回退Qwen")
            print(f"[chat_generate] flash 通道不可用({type(e).__name__}: {e})，回退 Qwen", flush=True)
    t0 = time.perf_counter()
    resp = await chat_client.chat.completions.create(
        model=CHAT_MODEL_NAME,
        temperature=0.7,
        messages=[
            {"role": "system", "content": CHAT_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    _perf_log(f"chat qwen {time.perf_counter() - t0:.2f}s")
    return (resp.choices[0].message.content or "").strip()


async def generate_tts(text: str, voice: str | None = None) -> str:
    """Step4：edge-tts 异步生成 mp3，返回可访问 URL。voice 为空时回退默认 TTS_VOICE。"""
    file_id = uuid.uuid4().hex
    out_path = AUDIO_DIR / f"audio_{file_id}.mp3"
    voice_for_tts = voice or TTS_VOICE
    communicate = edge_tts.Communicate(text, voice_for_tts)
    await communicate.save(str(out_path))
    return f"/static/audio/audio_{file_id}.mp3"


async def summarize_memory(user_text: str, ai_reply: str) -> str:
    """Step5：把一轮对话浓缩为 ≤40 字摘要，用于长期记忆。"""
    sys = "请将以下一段对话浓缩为一句话（不超过40字）的用户状态/事件摘要，用于长期记忆，不要包含任何诊断。"
    try:
        resp = await chat_client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            temperature=0.3,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": f"用户：{user_text}\n知心：{ai_reply}"},
            ],
        )
        return (resp.choices[0].message.content or "").strip()[:40]
    except Exception:
        return f"用户：{user_text[:40]}"


def extract_signals(text: str) -> dict:
    """跨场景模式 · 信号抽取（规则法，确定性）：
    从用户文本中抽取 {topic, sentiment}，供记忆元数据沉淀与后续模式统计。
    兼容性：旧记忆无 topic/sentiment 字段时按 other/neutral 处理，不影响检索。"""
    scores = {t: sum(1 for kw in kws if kw in text) for t, kws in TOPIC_KEYWORDS.items()}
    best = max(scores, key=lambda t: scores[t])
    topic = best if scores[best] > 0 else "other"
    neg = sum(1 for w in NEGATIVE_WORDS if w in text)
    pos = sum(1 for w in POSITIVE_WORDS if w in text)
    if neg > pos:
        sentiment = "negative"
    elif pos > neg:
        sentiment = "positive"
    else:
        sentiment = "neutral"
    return {"topic": topic, "sentiment": sentiment}


async def detect_patterns(user_id: str) -> list[dict]:
    """Task C：跨场景情绪行为模式识别（赛题方向③）。
    对该用户全部记忆元数据做轻量聚合，识别三类模式：
      1) recurring_topic   同一话题反复出现（>= PATTERN_TOPIC_REPEAT 次）
      2) late_night_low    深夜时段（23:00-05:00）低落情绪反复出现
      3) sustained_negative 最近连续多条记忆均为负面情绪
    合规边界：仅统计“用户主动表达过”的内容，不主动检测心理状态，不输出诊断。
    ChromaDB 异常时降级为空模式，不拖垮请求。"""

    def _detect() -> list[dict]:
        try:
            res = memory_collection.get(where={"user_id": user_id}, include=["metadatas"])
        except Exception:
            return []
        metas = [m for m in (res.get("metadatas") or []) if m]
        # 按时间戳升序，保证“最近 N 条”语义稳定
        metas.sort(key=lambda m: str(m.get("ts") or ""))
        patterns: list[dict] = []

        # ① 同一话题反复出现
        topic_counts: dict[str, int] = {}
        for m in metas:
            t = m.get("topic")
            if t and t != "other":
                topic_counts[t] = topic_counts.get(t, 0) + 1
        recurring = {t: c for t, c in topic_counts.items() if c >= PATTERN_TOPIC_REPEAT}
        if recurring:
            top_topic = max(recurring, key=lambda t: recurring[t])
            patterns.append({
                "type": "recurring_topic",
                "topic": top_topic,
                "count": recurring[top_topic],
                "description": f"话题「{TOPIC_LABELS.get(top_topic, top_topic)}」近期反复出现 {recurring[top_topic]} 次",
            })

        # ② 深夜时段情绪低落
        late_neg = [
            m for m in metas
            if m.get("hour") in LATE_NIGHT_HOURS and m.get("sentiment") == "negative"
        ]
        if len(late_neg) >= PATTERN_LATE_NIGHT_COUNT:
            patterns.append({
                "type": "late_night_low",
                "count": len(late_neg),
                "description": f"深夜时段（23:00-05:00）已出现 {len(late_neg)} 次低落情绪表达",
            })

        # ③ 持续负面（最近 4 条全为负面且不少于 3 条）
        recent = metas[-4:]
        if len(recent) >= 3 and all(m.get("sentiment") == "negative" for m in recent):
            patterns.append({
                "type": "sustained_negative",
                "count": len(recent),
                "description": f"最近 {len(recent)} 次交流均为负面情绪，呈持续状态",
            })
        return patterns

    return await asyncio.to_thread(_detect)


def build_pattern_hint(patterns: list[dict]) -> str:
    """把命中的跨场景模式转译为主聊 Prompt 的隐藏系统指令（复用 hidden_hint 通道）。
    语气要求：自然体现长期关注、更温柔、可关切作息；不点破“监测/分析”，绝不诊断。"""
    if not patterns:
        return ""
    descs = "；".join(p["description"] for p in patterns)
    return (
        f"[系统提示：跨场景模式觉察（仅基于用户主动表达的内容）——{descs}。"
        "请在回应中自然体现这份长期关注，语气更温柔，可温和关切作息，"
        "不主动点破“监测”或“分析”，绝不进行任何诊断。]"
    )


def store_memory_sync(user_id: str, summary: str, signals: dict | None = None) -> None:
    """记忆沉淀：除文本摘要外，同步写入结构化信号元数据（话题/情绪/时段/时间戳），
    供跨场景模式识别（detect_patterns）聚合统计。旧调用方不传 signals 时自动抽取。"""
    sig = dict(signals) if signals else extract_signals(summary)
    now = datetime.now()
    meta = {
        "user_id": user_id,
        "topic": sig.get("topic") or "other",
        "sentiment": sig.get("sentiment") or "neutral",
        "hour": int(sig.get("hour", now.hour)),
        "ts": now.isoformat(),
    }
    memory_collection.add(
        documents=[summary],
        ids=[f"{user_id}_{uuid.uuid4().hex}"],
        metadatas=[meta],
    )


async def background_store(user_id: str, summary: str, signals: dict | None = None) -> None:
    await asyncio.to_thread(store_memory_sync, user_id, summary, signals)


async def background_summarize_and_store(user_id: str, user_text: str, ai_reply: str) -> None:
    """后台任务：摘要生成（LLM 调用）+ 记忆沉淀串成一条后台链。
    摘要只在沉淀记忆时需要，与用户响应无关——绝不放在响应关键路径上 await。"""
    signals = extract_signals(user_text)
    summary = await summarize_memory(user_text, ai_reply)
    await asyncio.to_thread(store_memory_sync, user_id, summary, signals)


async def background_store_crisis(user_id: str) -> None:
    """High 风险：仅持久化脱敏事件标记（零原话），用于长期安全觉察，不污染日常记忆。"""
    await asyncio.to_thread(store_crisis_marker_sync, user_id)


def store_crisis_marker_sync(user_id: str) -> None:
    crisis_collection.add(
        documents=["[脱敏]高风险干预已触发"],
        ids=[f"crisis_{user_id}_{uuid.uuid4().hex}"],
        metadatas=[{"user_id": user_id, "type": "high_risk_intervention"}],
    )


def store_admin_log_sync(user_id: str, text: str, entry_type: str = "high_risk_intercept") -> None:
    """管理端危机日志：保留原始对话（含时间戳），仅供专家控制台优化风控规则。
    entry_type 区分来源：high_risk_intercept=高危拦截；safety_unavailable=风控不可用留痕。"""
    ts = datetime.now(timezone.utc).isoformat()
    admin_crisis_log.add(
        documents=[text],
        ids=[f"admin_{user_id}_{uuid.uuid4().hex}"],
        metadatas=[{"user_id": user_id, "ts": ts, "type": entry_type}],
    )


async def background_store_admin_log(user_id: str, text: str, entry_type: str = "high_risk_intercept") -> None:
    await asyncio.to_thread(store_admin_log_sync, user_id, text, entry_type)


# ----------------------------- 接口 -----------------------------
@app.post("/api/chat")
async def api_chat(req: ChatRequest, background_tasks: BackgroundTasks):
    user_text = (req.text or "").strip()
    if not user_text:
        return JSONResponse(
            {"risk_level": "Low", "message": "我在听，你说吧。", "audio_url": ""}
        )
    t_start = time.perf_counter()

    # ---- Step1：记忆/模式检索先行（本地毫秒级），随后风控与主聊并行 ----
    # 并行策略：主聊预生成与风控同时启动；风控仍拥有最终裁决权——
    # High 拦截时直接丢弃预生成结果（不展示、走紧急话术），风控逻辑零妥协。
    # 收益：正常消息的端到端延迟减少约一个风控往返（~3s）。
    memory, patterns = await asyncio.gather(
        query_memory(req.user_id, user_text),
        detect_patterns(req.user_id),
    )
    t_local = time.perf_counter() - t_start  # 本地检索耗时
    safety_task = asyncio.create_task(safety_check(user_text))
    chat_task = asyncio.create_task(chat_generate(memory, user_text))
    safety = await safety_task
    t_safety = time.perf_counter() - t_start  # 风控耗时（并行起点起算）
    # 主判定：risk_level == High 即拦截；action == INTERCEPT 作为二级兜底
    # （双保险：避免模型漏报 level 但给出 INTERCEPT 时漏拦）
    safety_action = str(safety.get("action", "")).upper()
    risk_level = (
        "High"
        if (safety.get("risk_level", "Low") == "High" or safety_action == "INTERCEPT")
        else safety.get("risk_level", "Low")
    )

    # ---- Step2：High 风险拦截，丢弃预生成的聊天结果 ----
    if risk_level == "High":
        chat_task.cancel()
        message = EMERGENCY_SCRIPT
        try:
            audio_url = await generate_tts(message, req.voice or None)
        except Exception:
            audio_url = ""  # 危急时刻 TTS 失败也必须返回文本，绝不能 500
        # 长期记忆功能对高危用户同样生效：仅沉淀脱敏事件标记，不存原话、不污染日常记忆
        background_tasks.add_task(background_store_crisis, req.user_id)
        # 管理端危机日志：保留原始对话，供专家优化风控规则（与用户侧记忆物理隔离）
        background_tasks.add_task(background_store_admin_log, req.user_id, user_text, "high_risk_intercept")
        return {"risk_level": "High", "message": message, "audio_url": audio_url}

    # ---- Step2.5：风控不可用（Fail-safe 兜底）----
    # 心理干预场景绝不容忍漏报：若风控超时/故障，不降级放行常规聊天，
    # 返回温和兜底文本并引导 12356，同时将本次对话留痕至管理端供专家复核。
    if risk_level == "Error":
        chat_task.cancel()
        message = SAFETY_FALLBACK_SCRIPT
        try:
            audio_url = await generate_tts(message, req.voice or None)
        except Exception:
            audio_url = ""
        background_tasks.add_task(background_store_admin_log, req.user_id, user_text, "safety_unavailable")
        return {"risk_level": "Error", "message": message, "audio_url": audio_url}

    # ---- Step3：正常聊天（Low / Medium），取用并行预生成结果 ----
    # Medium（躯体/精神高度耗竭）与跨场景模式命中时需注入隐藏关怀指令；
    # 预生成未携带指令，此少数场景重新生成一次（延迟与原串行方案持平），多数场景直取结果。
    hidden_hint = MEDIUM_CARE_HINT if risk_level == "Medium" else ""
    pattern_hint = build_pattern_hint(patterns)
    if pattern_hint:
        hidden_hint = f"{hidden_hint}\n{pattern_hint}" if hidden_hint else pattern_hint
    if hidden_hint:
        message = None
        try:
            message = await chat_generate(memory, user_text, hidden_hint=hidden_hint)
        except Exception as e:
            print(f"[api_chat] 带指令重新生成失败({type(e).__name__}: {e})", flush=True)
        if message is None:
            message = "嗯……我这边的网络好像不太顺畅，没听清你刚才说的话。可以试着再说一次吗？"
        chat_task.cancel()  # 预生成结果已弃用，释放资源
    else:
        try:
            message = await chat_task
        except Exception as e:
            print(f"[api_chat] 主聊生成失败({type(e).__name__}: {e})", flush=True)
            message = "嗯……我这边的网络好像不太顺畅，没听清你刚才说的话。可以试着再说一次吗？"

    # ---- Step4：TTS 生成 ----
    t_tts0 = time.perf_counter()
    try:
        audio_url = await generate_tts(message, req.voice or None)
    except Exception:
        audio_url = ""
    t_tts = time.perf_counter() - t_tts0
    t_total = time.perf_counter() - t_start
    _perf_log(
        f"total={t_total:.2f}s (local={t_local:.2f} safety={t_safety:.2f} "
        f"tts={t_tts:.2f}) risk={risk_level} user={req.user_id[:8]} "
        f"text={user_text[:24]}"
    )

    # ---- Step5：后台静默记忆沉淀（摘要 + 结构化信号：话题/情绪/时段）----
    # 摘要涉及一次 LLM 调用（~2-3s），已整体移入后台任务链，不再阻塞响应。
    background_tasks.add_task(background_summarize_and_store, req.user_id, user_text, message)

    return {"risk_level": risk_level, "message": message, "audio_url": audio_url}


# ----------------------------- 独立 TTS 接口 -----------------------------
@app.post("/api/tts")
async def api_tts(req: TTSRequest):
    """独立 TTS 接口：给定文本与可选音色，返回 edge-tts 生成的音频 URL。
    用于前端「选定音色后自动播放欢迎语 / 音色预览」等场景，无需走完整对话流程。"""
    text = (req.text or "").strip()
    if not text:
        return {"audio_url": ""}
    try:
        audio_url = await generate_tts(text, req.voice or None)
        return {"audio_url": audio_url}
    except Exception:
        # TTS 失败时返回空串，由前端静默降级（仍有文字兜底），绝不 500
        return {"audio_url": ""}


# ----------------------------- 记忆管理面板接口（比赛加分项） -----------------------------
@app.get("/api/memory")
async def get_memory(user_id: str):
    """用户查看系统记住了自己的哪些事（记忆管理面板）。"""
    res = memory_collection.get(where={"user_id": user_id}, include=["documents"])
    ids = res.get("ids", [])
    docs = res.get("documents", [])
    return {"user_id": user_id, "memories": [{"id": i, "text": d} for i, d in zip(ids, docs)]}


@app.delete("/api/memory/{memory_id}")
async def delete_memory(memory_id: str):
    """用户手动删除某条记忆（记忆管理面板）。"""
    memory_collection.delete(ids=[memory_id])
    return {"deleted": memory_id, "ok": True}


# ----------------------------- 跨场景模式查询接口（赛题方向③） -----------------------------
@app.get("/api/patterns")
async def get_patterns(user_id: str):
    """查看系统对该用户识别出的跨场景情绪行为模式。
    合规边界：仅基于用户主动表达内容的频次统计，不主动检测心理状态，不输出诊断结论；
    模式仅用于个性化陪伴语气（隐藏指令注入），并在前端向用户透明展示。"""
    patterns = await detect_patterns(user_id)
    return {
        "user_id": user_id,
        "patterns": patterns,
        "note": "仅基于你主动表达过的内容进行统计，可随时在记忆管理面板删除对应记忆。",
    }


# ----------------------------- 管理端 / 专家控制台接口 -----------------------------
@app.get("/api/admin/crisis-log")
async def admin_crisis_log_view(token: str = ""):
    """危机拦截日志：查看累计与今日拦截次数及原始对话（供医疗/辅导员团队优化风控）。
    演示环境采用简易 Token 隔离（?token=xxx）；生产环境将接入完善的 RBAC（基于角色的访问控制）
    与脱敏网关，并对原始对话做脱敏存储。"""
    expected = os.getenv("ADMIN_TOKEN", "zhixin2026")
    if not token or token != expected:
        return JSONResponse(status_code=403, content={"detail": "invalid or missing admin token"})
    res = admin_crisis_log.get(include=["documents", "metadatas"])
    metas = res.get("metadatas", [])
    docs = res.get("documents", [])
    items = [
        {"user_id": m.get("user_id"), "ts": m.get("ts"), "text": d}
        for m, d in zip(metas, docs)
    ]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_items = [it for it in items if (it.get("ts") or "").startswith(today)]
    return {"total": len(items), "today_count": len(today_items), "today_logs": today_items}


@app.get("/")
async def root():
    return {
        "service": "知心",
        "status": "ok",
        "chat_model": FLASH_MODEL_NAME if flash_client else CHAT_MODEL_NAME,
        "chat_fallback_model": CHAT_MODEL_NAME if flash_client else None,
        "safety_model": SAFETY_MODEL_NAME,
    }


@app.get("/app")
async def serve_app():
    """托管前端单页（单服务部署模式）：宝塔部署后访问 http://IP:端口/app 即进入对话页，
    页面与 API 同源同端口，无需另起静态服务。"""
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE, media_type="text/html")
    return JSONResponse({"error": "index.html 未随包部署，请使用独立静态服务模式"}, status_code=404)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
