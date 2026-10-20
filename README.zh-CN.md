# classic-nlp-mcp

**用 MCP 提供确定性传统 NLP 能力 —— 给 LLM Agent 装上便宜、可复现、离线的"语言传感器"。**

[English](README.md) | [中文文档](README.zh-CN.md)

**以智谱 Z.ai GLM-5.3 作为驱动 Agent 完成开发与端到端实测。** 兼容所有 MCP 宿主：Claude Desktop、Z Code（GLM Coding Plan）、WorkBuddy、Cursor、Cline、自研 Agent。

大模型时代，传统 NLP 库没有过时，只是换了岗位：它们不再是"NLP 引擎"，而是引擎底下的**确定性工具层**——批量预处理近乎零成本、字符偏移永远精确、结果可复现、完全离线、零幻觉。本项目把这层能力暴露给所有支持 MCP 的客户端（Claude Desktop、WorkBuddy、Cursor、自研 Agent……），按**能力（capability）**而非"一个库一个壳"来组织。

## 为什么还要传统 NLP

| 需求 | 只用 LLM | classic-nlp-mcp |
|---|---|---|
| 给 10 万篇文档做分词/切句（RAG 预处理） | 又贵又慢 | 快、≈0 成本、离线 |
| 引用需要精确字符偏移 | 经常漂移 | 永远精确 |
| 知识图谱构建要 (S,P,O) 候选三元组 | 质量好但贵 | OpenIE 打底，LLM 只做校验 |
| 省 token 的两级漏斗（粗筛 → 精判） | — | 天生为此设计 |
| 可解释语义（WordNet、知网义原） | 没有原生接口 | 一等公民工具 |
| 可复现的科研流水线 | 输出会变 | 确定性 |

### 这个项目是给谁用的

- **RAG / 检索工程师**——`sent_split` + `tokenize` 是确定性分块的地基；`keyword_extract` 在跑 embedding 之前就给每个块挂上几乎零成本的元数据。
- **知识图谱 / GraphRAG 建设者**——`openie_triples` 以≈0 成本挖出带来源 span 的 (S,P,O) 候选三元组；LLM 只在最后做本体归一化与校验。
- **心疼 token 账单的 Agent 开发者**——两级漏斗：传统 NER/关键词粗筛，只把候选送 LLM 精判。同样的答案，上下文少一个量级。
- **隐私与合规敏感的部署**——所有工具本地离线运行，文本不出进程，结果逐字节可复现。
- **科研人员**——精确偏移、带版本号的引擎归属、确定性输出，流水线可审计、论文可辩护。

**一句话价值**：LLM 时代传统 NLP 的价值不在于"比大模型更懂语言"，而在于**做大模型底下那层便宜、精确、可复现的传感器与预处理器**。本项目让这层能力通过 MCP 被任何 Agent 直接调用。

## 设计原则

1. **包装能力，不包装库**。工具面向任务（`tokenize`、`openie_triples`、`seme_lookup`），库只是适配器背后可替换的引擎。
2. **单网关 + 可选引擎**。一个 MCP 进程搞定；重引擎（spaCy、NLTK、OpenHowNet）都是可选依赖，`jieba` 是唯一的必装依赖——中文开箱即用。
3. **统一输出 Schema**。所有工具返回同一信封结构，引擎归属诚实标注，字符偏移精确对齐原文。
4. **优雅降级，绝不撒谎**。缺可选依赖 → 给出可操作的安装提示（不是崩）；降级引擎明确标注（`regex-fallback`）。
5. **mcp SDK 1.x / 2.x 双兼容**。`mcp` 2.0 把 `FastMCP` 改名为 `MCPServer` 并删除了旧导入路径；兼容垫片见 [`src/classic_nlp_mcp/_compat.py`](src/classic_nlp_mcp/_compat.py)。

## 架构

```
 MCP 客户端（Claude / WorkBuddy / Cursor / ...）
        │
        ▼
 classic-nlp-mcp 网关  ── 单进程 · 能力级路由 · 统一输出 Schema
        │
 ┌──────┼───────────┬──────────────┐
 ▼      ▼           ▼              ▼
spaCy  jieba       NLTK        OpenHowNet
(可选) (必装)      (可选)        (可选)
```

## 工具一览

| 工具 | 能力 | 引擎（中 / 英） |
|---|---|---|
| `list_capabilities` | 报告已装引擎与各语言能力矩阵 | — |
| `tokenize` | 分词，精确字符偏移 | jieba / spaCy → stanza → 正则兜底 |
| `pos_tag` | 词性标注（stanza 附带词元） | jieba.posseg / spaCy → stanza |
| `ner` | 命名实体识别（带 span） | stanza 统计 + jieba 启发式 + spaCy / spaCy → stanza |
| `keyword_extract` | 关键词（带权重） | jieba TF-IDF/TextRank / 词频兜底 |
| `sent_split` | 句子切分（RAG 分块地基） | jieba+规则 / spaCy → stanza → 正则兜底 |
| `openie_triples` | 开放三元组（S,P,O），带来源 span 与置信度 | jieba 规则（中文，草稿级）/ **Stanford CoreNLP**（正统）→ spaCy 依存规则 |
| `constituency_parse` | 成分句法树 + 节点标签 | stanza（斯坦福独占） |
| `seme_lookup` | 知网义原查询 | OpenHowNet（中文） |
| `wordnet_lookup` | 同义词组、定义、上位词 | NLTK WordNet（英文） |
| `wordnet_similarity` | WordNet 路径相似度 | NLTK WordNet（英文） |
| `annotate_full` | **一次调用跑全部能力**，按能力隔离报错 | 编排 |

**多语言**：stanza 支撑 `tokenize` / `pos_tag` / `ner` / `sent_split` / `constituency_parse` 的**任意 stanza 语言代码**（`fr`、`de`、`ja`、`es`……）——下载对应语言模型后把代码当 `lang` 传即可。

### 统一输出 Schema

所有工具返回同一信封：

```json
{
  "tool": "tokenize",
  "lang": "zh",
  "engine": "jieba",
  "engine_version": "0.42.1",
  "tokens": [{"text": "知识", "start": 2, "end": 4, "pos": "n"}]
}
```

`openie_triples` 额外返回 `confidence` 和来源 `span` + `sentence`——三元组是"带出处的候选"，下游接本体对齐或 LLM 校验都方便。

## 兼容性与版本

所有被包装的工具包都按**当前版本 API** 编写，版本约束刻意覆盖最新发布：

| 包 | 支持范围 | 成文时最新版 | 说明 |
|---|---|---|---|
| Python | >= 3.10 | 3.13 / 3.14 | spaCy wheel 覆盖 3.9–3.14 |
| [mcp](https://pypi.org/project/mcp/) SDK | `>=1.2, <3` | 2.0.x | **2.0 把 `FastMCP` 改名为 `MCPServer`** 并删除旧导入路径；兼容垫片（`src/classic_nlp_mcp/_compat.py`）双版本通吃 |
| [jieba](https://pypi.org/project/jieba/) | `>=0.42.1` | 0.42.1（多年稳定，纯 Python） | 唯一必装依赖——中文开箱即用 |
| [spaCy](https://pypi.org/project/spacy/) | `>=3.7`（extras `en`） | 3.8.x（2026-08） | 当前 `spacy.load` / `Doc` / `Span` API；`en_core_web_sm`、`zh_core_web_sm` 均支持 |
| [stanza](https://pypi.org/project/stanza/) | `>=1.8`（extras `stanford`） | 1.15.x | 按 1.15 编写（模型在平台缓存目录）；下载某语言模型后即可用该语言代码。**任何已发布版本都不带 OpenIE**——正统 OpenIE 走 CoreNLP 桥 |
| Stanford [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) | 任意 4.x 服务端 | 4.5.10 | 可选：通过本地 `StanfordCoreNLPServer`（Java，`CORENLP_URL`，默认 `localhost:9000`）提供正统 OpenIE；服务未起时优雅降级 |
| [NLTK](https://pypi.org/project/nltk/) | `>=3.8`（extras `semantic`） | 3.10.x | `nltk.corpus.wordnet` API 在 3.x 全线稳定 |
| [OpenHowNet](https://pypi.org/project/OpenHowNet/) | 任意（extras `semantic`） | 0.0.x（2020 年发布，数据内置） | 上游许久未发版；已做防御式包装（`try_import` + 优雅降级），绝不会拖垮服务 |

所有可选依赖都带可操作的安装提示优雅降级——只装 `jieba` 服务也能完整启动并保持可用。

## 安装

```bash
# 基础安装（中文能力立即可用）
pip install classic-nlp-mcp

# 可选引擎
pip install "classic-nlp-mcp[en]"         # spaCy
pip install "classic-nlp-mcp[semantic]"   # NLTK + OpenHowNet
pip install "classic-nlp-mcp[all]"

python -m spacy download en_core_web_sm   # spaCy 英文模型
# 可选：python -m spacy download zh_core_web_sm  （中文统计 NER）
# 可选：python -c "import nltk; nltk.download('wordnet')"
```

从源码：

```bash
git clone https://github.com/liftkkkk/classic-nlp-mcp.git
cd classic-nlp-mcp
pip install -e ".[all,dev]"
pytest
```

## 接入你的 Agent 宿主（MCP 配置）

classic-nlp-mcp 是标准 **stdio** MCP 服务，任何讲 MCP 的客户端都能接。本项目以**智谱 Z.ai GLM-5.3** 作为驱动 Agent 完成开发与端到端实测。

### Claude Desktop / Cursor / Cline / 通用 `mcpServers` 客户端

```json
{
  "mcpServers": {
    "classic-nlp": {
      "command": "python",
      "args": ["-m", "classic_nlp_mcp"]
    }
  }
}
```

### Z Code（智谱 GLM Coding Plan / GLM-5.3）

[Z Code](https://docs.z.ai/) 使用同款标准 `mcpServers` 配置块——加进 Z Code 的 MCP 设置后，GLM-5.3 Agent 就能在编程会话里直接调用 `tokenize`、`openie_triples`、`annotate_full` 等工具：

```json
{
  "mcpServers": {
    "classic-nlp": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "classic_nlp_mcp"]
    }
  }
}
```

Z Code 典型用法：对模块文档字符串跑 `annotate_full` → 留下 `confidence ≥ 0.35` 的 `openie_triples` → 让 GLM-5.3 按你的本体做归一化。便宜且确定性的抽取先行，昂贵的模型推理只花在值得的地方。

### WorkBuddy

把同样的配置块加进 `~/.workbuddy/mcp.json`，然后在连接器管理页信任该服务。

### 或使用安装后的命令行入口

```json
{
  "mcpServers": {
    "classic-nlp": { "command": "classic-nlp-mcp" }
  }
}
```

## 典型 Agent 工作流

- **RAG 入库**：`sent_split` 按句子切块 → `keyword_extract` 给每块挂元数据。
- **知识图谱流水线**：语料跑 `annotate_full` → 留下 `confidence ≥ 0.35` 的 `openie_triples` → LLM 按本体归一化 S/P/O → 入图。
- **省 token 漏斗**：先 `ner` 粗筛，只把候选实体送 LLM 做类型判定/链接。
- **可解释相似度**：先试 `seme_lookup`（义原）或 `wordnet_similarity`（路径相似度），再考虑向量相似度。

## v0.2 ——斯坦福全家桶到位

两个新成员，因为斯坦福工具链在特定领域**仍然不可替代**：

- **[stanza](https://github.com/stanfordnlp/stanza)** 适配器（可选 `[stanford]` extra）：神经统计 NER、词性+词元、**60+ 语言**的句子切分（把任意 stanza 语言代码当 `lang` 传即可），以及 `constituency_parse` 成分句法树——其他引擎都没有这块。注意：**任何已发布的 stanza 版本都不带 OpenIE**——OpenIE 一直在 CoreNLP 里，所以才有下一条。
- **Stanford CoreNLP OpenIE 桥**：*正统* OpenIE 实现。本地起一个 `StanfordCoreNLPServer`（需要 Java），本 MCP 自动把它用作英文三元组的最高优先级引擎——全项目质量最高的候选三元组，同样带来源 span。服务没起时英文自动降级到 spaCy 依存规则，`list_capabilities` 如实上报状态。

```bash
pip install "classic-nlp-mcp[stanford]"
python -c "import stanza; stanza.download('en')"
# 可选，正统 OpenIE（需要 Java）：
#   从 https://stanfordnlp.github.io/CoreNLP/download.html 下载 CoreNLP 并启动：
#   java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

## 本项目依赖的上游开源项目

- [spaCy](https://github.com/explosion/spaCy) — 工业级多语言 NLP · [文档](https://spacy.io/)
- [jieba](https://github.com/fxsjy/jieba) — 经典中文分词库
- [NLTK](https://github.com/nltk/nltk) · [文档](https://www.nltk.org/) — WordNet 与 NLP 时代的标准库
- [OpenHowNet](https://github.com/thunlp/OpenHowNet) · [知网官网 (keenage.com)](https://www.keenage.com/) — THUNLP 义原词典
- [stanza](https://github.com/stanfordnlp/stanza) · [文档](https://stanfordnlp.github.io/stanza/) — 斯坦福 NLP 组的 Python 库
- [Stanford CoreNLP](https://github.com/stanfordnlp/CoreNLP) · [官网](https://stanfordnlp.github.io/CoreNLP/) — 经典 OpenIE 背后的 Java 工具包（JVM；可用同一适配器模式接入）
- [Model Context Protocol](https://github.com/modelcontextprotocol) · [规范与文档](https://modelcontextprotocol.io/) — 本服务所讲的协议

> 说明：stanza 以 `[stanford]` extra 提供（模型按语言单独下载）；CoreNLP 通过其 HTTP 服务接入——见上方 v0.2 小节。

## 路线图

- [ ] CoreNLP 适配器加固：自动拉起服务端、中文 OpenIE 规则
- [ ] stanza 中文模型 CI 冒烟测试
- [ ] 按 `(text, lang, capability, engine_version)` 键的 LRU/磁盘结果缓存
- [ ] 对齐自定义本体的 RDF / JSON-LD 导出
- [ ] `compare_engines` 工具（同一能力多引擎对齐输出）

## 请作者喝杯咖啡

classic-nlp-mcp 是我在业余时间开发和维护的开源项目，永久免费。

如果它给你的 Agent 提供了确定性语言传感器、便宜的知识图谱预处理，或者帮你省掉了拼接五个 NLP 库的麻烦——请我喝杯咖啡吧（9.9 元就够）。每一份支持都会直接变成新功能和 bug 修复。

<p align="center">
  <img src="donate.jpg" alt="请作者喝杯咖啡" width="280">
</p>

<p align="center"><i>打赏时留言告诉我你最想要的功能，我会优先做 ;)</i></p>

**不想打赏？** 点个 Star、提个 Issue、转发给需要的人，同样是对项目的巨大支持！

## 许可证

[MIT](LICENSE) © 2026 liftkkkk
