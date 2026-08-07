# Financial PoC

PDF 财报 + 财经新闻 → 自动处理 → 向量索引 → 智能检索 / 日报。

**不配置 LLM API 也能用**（检索、日报均离线可用）。LLM 仅用于 `ask` 问答和公司简报，可选。

---

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

PDF 放入 `data/raw/pdf/`，然后：

```powershell
python scripts/agent.py sync
python scripts/agent.py          # 进入交互检索（演示用这一条即可）
```

---

## 一键全分析（推荐）

把财报 PDF 放入 `data/raw/pdf/`，然后：

```powershell
python scripts/agent.py analyze
```

自动完成：**解析财报 → 抽取指标/关键词 → 爬取网络新闻与股价 → 同业对比 → 输出高估/合理/低估**。

报告保存至 `docs/analysis/`。

```powershell
python scripts/agent.py analyze 贵州茅台2024年报.pdf   # 指定 PDF
python scripts/agent.py analyze 贵州茅台                 # 已入库公司，跳过 PDF 处理
```

---

## 常用命令

| 命令 | 作用 |
|------|------|
| **`python scripts/agent.py analyze`** | **一键全分析（推荐）** |
| `python scripts/agent.py` | 交互检索演示 |
| `python scripts/agent.py sync` | 抓新闻 + 处理 PDF + 建索引 |
| `python scripts/agent.py query "贵州茅台2024年净利润"` | 单次检索 |
| `python scripts/agent.py pdf --valuate` | 导入 PDF 后自动输出估值结论 |
| `python scripts/agent.py compare 贵州茅台` | 爬取网络行情/新闻，实时同业对比 |
| `python scripts/agent.py compare --watchlist` | 监控列表批量实时对比 |
| `python scripts/agent.py valuate 贵州茅台 --compare` | 财报估值 + 实时网络对比 |
| `python scripts/agent.py daily` | 生成规则化日报 → `docs/daily/` |
| `python scripts/agent.py pdf` | 仅处理新 PDF |
| `python scripts/agent.py ask "..."` | RAG 问答（需 API Key） |
| `python scripts/agent.py run` | 自主跑一轮（含 LLM 简报，需 API Key） |

```powershell
python scripts/agent.py sync --skip-fetch    # 跳过抓新闻，用本地数据
python scripts/check_code.py                 # 代码检测（快速）
python scripts/check_code.py --full          # 含检索+估值（较慢）
python scripts/agent.py ask -i               # 连续对话
python scripts/agent.py serve --time 08:00   # 每天定时自主运行
```

---

## 无 LLM 模式（推荐，零费用）

`.env` 中关闭 LLM，避免无意义的 Preview 输出：

```env
FINANCIAL_POC_AGENT_ENABLE_LLM=false
```

日常使用：`sync` → `query` → `daily` 即可。

---

## 可选：接入 LLM

`.env` 任选一家：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
```

---

## 目录

```text
data/raw/pdf/       原始 PDF
data/raw/news/      新闻（自动抓取）
data/chroma/        向量索引
docs/daily/         日报输出
scripts/agent.py    统一入口
```

---

## API（可选）

```powershell
uvicorn api.main:app --reload --port 8000
```

Swagger：`http://127.0.0.1:8000/docs`（调试接口用，不是 Agent 本体）

---

## 说明

- 新增 PDF **不用改代码**，丢进 `data/raw/pdf/` 后跑 `pdf` 或 `sync`
- 首次运行会下载 Embedding / Rerank 模型，较慢，之后会缓存
- 问句带上**公司名 + 指标 + 年份**检索更准
