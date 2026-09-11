# 知心 · 非临床心理健康语音陪伴多智能体系统

> 基于 FastAPI + ChromaDB + edge-tts 的多智能体语音陪伴后端，搭配 Vue3 单页前端，提供长期记忆、情绪安抚、安全风控三件套。

## ✨ 核心特性

- 🎙️ **文字 + 语音一体化**：edge-tts 流式合成，自动播放浏览器兼容、iOS 静音拨片兼容
- 🧠 **长期记忆**：基于 ChromaDB 向量数据库，按 `uid` 隔离，支持跨天跨设备召回
- 🛡️ **三段式安全风控**：用户消息先过安全模型 → 主陪伴者回应 → 自动危机干预话术
- ⚡ **并行加速**：风控与主聊并行 + 摘要后台生成，首字延迟 < 8 秒
- 🌐 **零依赖前端**：单文件 `index.html`，双击即可在浏览器打开（指向本地 / 远程后端）

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────────┐
│                       index.html (Vue3 SPA)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ 文本输入  │  │ 音色选择  │  │ 语音播放  │  │ 历史记录  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
└───────┼─────────────┼─────────────┼─────────────┼──────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI (main.py)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ /api/chat│  │ /api/tts │  │/api/memory│  │/api/health│   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼─────────────┼─────────────┼─────────────┼──────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
   ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ LLM 主聊 │  │ 安全模型  │  │  ChromaDB │  │ edge-tts │
   │ (OpenAI │  │ (Reasoner│  │ (向量数据库│  │ (微软    │
   │ 兼容端点)│  │  强档)   │  │  本地持久) │  │  免费)   │
   └─────────┘  └──────────┘  └──────────┘  └──────────┘
```

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/<your-account>/zhixin-voice-companion.git
cd zhixin-voice-companion
```

### 2. 安装依赖

需要 Python 3.12+。

```bash
# 推荐：创建虚拟环境
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate             # Windows

# 安装依赖
pip install -r requirements.txt
```

> 📌 阿里云轻量服务器等老系统可能需要 `pysqlite3-binary` 补丁（`run.py` 已自动处理）。

### 3. 配置环境变量

```bash
cp .env.example .env
# 用编辑器打开 .env，至少填入三个 API KEY：
#   CHAT_FLASH_API_KEY   (主聊)
#   CHAT_API_KEY         (回退)
#   SAFETY_API_KEY       (安全)
```

### 4. 启动后端

```bash
python run.py
```

看到 `Uvicorn running on http://0.0.0.0:8000` 即启动成功。

### 5. 打开前端

- **本地**：浏览器访问 `http://localhost:8000/app`
- **带长期记忆的演示账号**：访问 `http://localhost:8000/app?uid=demo_lin`

## 📂 目录结构

```
.
├── main.py                # FastAPI 后端入口（所有 API 路由）
├── run.py                 # 启动器（兼容 venv / 系统 Python / Docker）
├── index.html             # 前端单页应用（Vue3 + Tailwind 风格）
├── seed_memories.py       # 演示账号（demo_lin）种子记忆灌入脚本
├── requirements.txt       # Python 依赖清单
├── .env.example           # 环境变量模板
├── .gitignore
├── LICENSE
├── prompts/               # 大模型提示词
│   ├── chat_prompt.txt       # 主陪伴者系统提示词
│   └── safety_prompt.txt     # 安全风控提示词
└── static/
    └── audio/             # edge-tts 生成的语音文件缓存（运行时自动生成）
```

## 🔧 关键配置说明

| 环境变量 | 必填 | 说明 |
|---------|------|------|
| `CHAT_FLASH_*` | ✅ | 主聊陪伴者，4 个变量必须同源（BASE_URL/API_KEY/MODEL_NAME/TTFB_TIMEOUT） |
| `CHAT_*` | ✅ | 回退模型，主聊失败时使用；建议与主聊不同厂商避免同一故障 |
| `SAFETY_*` | ✅ | 安全风控模型，**强烈建议用更强档**（deepseek-reasoner / gpt-4o 等） |
| `TTS_VOICE` | ❌ | edge-tts 音色，默认 `zh-CN-XiaoxiaoNeural` |
| `CHROMA_PATH` | ❌ | 向量库路径，默认 `./chroma_db` |
| `PORT` | ❌ | 监听端口，默认 8000 |

## 📊 性能基准

| 指标 | 数值 | 测试环境 |
|------|------|---------|
| 首字延迟 (TTFT) | 0.8-2.5s | Deepseek-V4-flash (免费层) |
| 完整回复延迟 | 4-8s | 文本 + 流式 TTS |
| 并发支持 | 10-15 路 | 2GB 内存 / 单核 |
| 向量召回 | <100ms | ChromaDB PersistentClient (10K 条) |
| TTS 流式 | 200-400ms/句 | edge-tts 微软免费端点 |

## 🛡️ 安全设计

1. **三段式风险拦截**：
   - 用户消息 → 安全模型打分（low/medium/high）
   - high → 立即返回危机干预话术（不调用主聊）
   - low/medium → 并行生成主聊回复
2. **危机干预话术**：内置 6 种（自杀倾向 / 自伤 / 暴力 / 药物滥用 / 性侵 / 严重抑郁），对接 24h 心理援助热线
3. **数据隔离**：ChromaDB 按 `uid` 分 collection，不同用户互不可见
4. **非临床声明**：界面顶部固定免责声明，本系统**不替代专业医疗**

## 📜 许可证

本项目采用 [MIT License](LICENSE) 开源。

## 🙏 致谢

- [FastAPI](https://fastapi.tiangolo.com/) - 异步 Web 框架
- [ChromaDB](https://www.trychroma.com/) - 嵌入式向量数据库
- [edge-tts](https://github.com/rany2/edge-tts) - 微软免费 TTS
- [Vue3](https://vuejs.org/) - 前端框架

## ⚠️ 免责声明

本系统为非临床心理陪伴工具，**不提供医学诊断或治疗**。如有严重心理困扰，请联系专业医疗机构或拨打 24 小时心理援助热线 **400-161-9995**。