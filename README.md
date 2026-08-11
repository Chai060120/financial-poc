# Financial PoC

PDF 财报 + 财经新闻 → 自动处理 → 向量索引 → 规则估值 / 对话分析 / 网页 Agent。

**不配置任何 LLM / Cursor API 也能用**：检索、指标抽取、高估/合理/低估结论均基于本地规则与模型。  
LLM / Cursor 仅用于可选的自然语言问答与解读。

---

## 安全须知（必读）

**千万不要把 API Key（如 OpenAI、DeepSeek、Cursor 等）硬编码进代码或提交到 GitHub。**

正确做法：从环境变量 / `.env` 读取：

```python
import os

api_key = os.getenv("OPENAI_API_KEY")
```

本仓库已通过 `python-dotenv` 加载项目根目录的 `.env`（见 `config.py`）。  
**运行时请在 `.env` 文件中配置你的 API Key**（可复制 `.env.example` 后填写）。

- `.env` 已加入 `.gitignore`，**不要** `git add .env`
- 仓库只提供 `.env.example` 作为模板（不含真实密钥）

---

## 功能一览

| 能力 | 说明 |
|------|------|
| 网页 Agent | 浏览器对话分析，一键启动 |
| 自动下年报 | 输入「公司名 + 年份」从巨潮资讯下载年报 |
| 网络新闻 | 分析时抓取个股相关新闻 |
| 手动上传 PDF | 网页上传或放入 `data/raw/pdf/` |
| 五段式报告 | 估值结论 / 核心指标 / 关键词 / 行情对比 / 资讯依据 |
| 多轮追问 | 为什么？ / 和同业比呢？ / 净利润多少？ |
| 可选 Cursor 解读 | 分析后追加自然语言段落 |
| 可选 LLM 问答 | `ask` 命令 / RAG 对话 |

---

## 快速开始

### 1. 安装

```powershell
cd financial-poc
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

按需编辑 `.env`（无 Key 也可先跑规则分析）。

### 2. 网页 Agent（推荐）

双击 `启动网页Agent.bat`，或：

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

浏览器打开：**http://127.0.0.1:8000/agent**

在对话框输入例如：

- `美的集团`
- `美的集团 2024`
- `贵州茅台2024年报`

也可点击「上传 PDF 财报」。

### 3. 终端 Agent

```powershell
python scripts/agent.py
```

单次分析：

```powershell
python scripts/agent.py analyze 中国平安
python scripts/agent.py analyze 美的集团
```

---

## 环境变量（`.env`）

复制模板：

```powershell
copy .env.example .env
```

常用项：

```env
# —— 规则分析即可，无需 Key ——
FINANCIAL_POC_AGENT_ENABLE_LLM=false

# —— 可选：Cursor 自然语言解读 ——
CURSOR_API_KEY=
FINANCIAL_POC_CURSOR_NARRATIVE=true
FINANCIAL_POC_CURSOR_MODEL=composer-2.5

# —— 可选：LLM 问答（任选一家）——
LLM_PROVIDER=deepseek
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
QWEN_API_KEY=
DASHSCOPE_API_KEY=

# —— 可选：Tushare ——
TUSHARE_TOKEN=
```

说明：

- **不填 Key**：网页/终端 Agent 的五段式分析、检索、估值仍可用
- **填 Cursor Key**：分析末尾可追加第 6 段自然语言解读
- **填 LLM Key**：可用于 `python scripts/agent.py ask -i` 等问答能力

---

## 常用命令

| 命令 | 作用 |
|------|------|
| `启动网页Agent.bat` / `uvicorn api.main:app --port 8000` | 网页 Agent |
| `python scripts/agent.py` | 终端对话 Agent |
| `python scripts/agent.py analyze 公司名` | 单次全量分析 |
| `python scripts/agent.py query "..."` | 单次财报检索 |
| `python scripts/agent.py pdf` | 仅处理新 PDF |
| `python scripts/agent.py sync --skip-fetch` | 同步本地 PDF/新闻到索引 |
| `python scripts/agent.py ask -i` | LLM 连续对话（需 API Key） |
| `python scripts/agent.py serve --time 08:00` | 定时日报 |
| `python scripts/check_code.py` | 代码检测 |

Swagger（调试接口）：http://127.0.0.1:8000/docs

---

## 使用说明

### 自动下载年报

输入公司名（可带年份）后，Agent 会：

1. 从巨潮资讯检索对应年报 PDF  
2. 保存到 `data/raw/pdf/` 并入库建索引  
3. 抓取个股网络新闻  
4. 输出高估 / 合理 / 低估等五段报告  

若自动下载失败，会尽量用已有索引继续分析；也可手动上传 PDF。

### 手动上传 PDF

- 网页：左侧「上传 PDF 财报」  
- 或把文件放到 `data/raw/pdf/` 后执行 `python scripts/agent.py pdf`

建议文件名：`公司名+年份+年报.pdf`（例如 `美的集团2024年年报.pdf`），便于识别公司。

### 无 LLM 模式（零费用）

```env
FINANCIAL_POC_AGENT_ENABLE_LLM=false
```

---

## 目录结构

```text
api/                    FastAPI（网页 Agent + REST）
web/agent.html          网页 Agent 前端
src/agent/              对话 Agent / 意图路由 / Cursor 解读
src/analysis/           全量分析、估值、行情对比
src/financial/          指标抽取 / PE·PB / 数据校验
src/collectors/         PDF 扫描、新闻 RSS、巨潮年报下载
src/pipelines/          文档处理与索引流水线
scripts/agent.py        统一 CLI 入口
data/raw/pdf/           原始财报 PDF（本地，默认不提交大文件）
data/raw/news/          新闻缓存
docs/analysis/          分析报告输出
.env.example            环境变量模板（无密钥）
启动网页Agent.bat       Windows 一键启动
```

---

## 说明与限制

- 本项目为 **PoC**，输出不构成投资建议  
- 首次运行会下载 Embedding / Rerank 模型，较慢，之后会缓存  
- 巨潮下载依赖网络；个别公司/年份若未披露或接口变更，可能失败  
- 请勿将含真实密钥的 `.env`、私有财报或隐私数据推送到公开仓库  

---

## License

PoC / 学习用途。按需自行补充许可证。
