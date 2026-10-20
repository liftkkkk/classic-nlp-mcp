# classic-nlp-mcp

**Deterministic classical NLP, served over MCP — give your LLM agent cheap, reproducible, offline language sensors.**

[English](README.md) | [中文文档](README.zh-CN.md)

**Developed and end-to-end tested with Z.ai GLM-5.3 as the driving agent.** Works with any MCP host: Claude Desktop, Z Code (GLM Coding Plan), WorkBuddy, Cursor, Cline, or your own agent.

In the LLM era, classical NLP libraries did not become useless — they changed jobs. They are no longer the *engine*; they are the **deterministic tool layer** underneath one: batch preprocessing at near-zero cost, exact character spans, reproducible results, fully offline, zero hallucination. This project exposes that layer to any MCP-compatible agent (Claude Desktop, WorkBuddy, Cursor, your own agent...) as **capability-level tools**, not as one-wrapper-per-library glue.

## Why

| You need | LLM alone | classic-nlp-mcp |
|---|---|---|
| Tokenize/parse 100k docs for RAG chunking | expensive, slow | fast, ≈0 cost, offline |
| Exact character offsets for citations | drifts | always exact |
| Candidate (S, P, O) triples for KG construction | good but costly | OpenIE first, LLM verifies later |
| Cheap two-stage funnel (coarse NER → LLM refine) | — | built for it |
| Interpretable semantics (WordNet, HowNet sememes 义原) | no native interface | first-class tools |
| Reproducible research pipelines | varies | deterministic, cached logic |

### Who is this for?

- **RAG / search engineers** — `sent_split` + `tokenize` are the foundation of deterministic chunking; `keyword_extract` attaches cheap metadata to every chunk before embeddings ever run.
- **Knowledge-graph / GraphRAG builders** — `openie_triples` mines candidate (S, P, O) triples with provenance spans at ≈0 cost; the LLM only normalizes and verifies against your ontology afterwards.
- **Agent developers watching their token bill** — the two-stage funnel: classical NER/keyword pre-filter, then send only the candidates to the LLM. Same answer, order-of-magnitude less context.
- **Privacy- and compliance-sensitive deployments** — every tool runs locally and offline. Text never leaves the process, and results are reproducible byte-for-byte.
- **Researchers** — exact spans, versioned engine attribution and deterministic output make pipelines auditable and papers defensible.

**The one-line value**: classical NLP in the LLM era is not about "understanding language better than an LLM" — it is about being the **cheap, exact, reproducible sensor and preprocessor layer** underneath one. This project makes that layer callable by any agent over MCP.

## Design principles

1. **Wrap capabilities, not libraries.** Tools are task-oriented (`tokenize`, `openie_triples`, `seme_lookup`); libraries are swappable backends behind adapters.
2. **One gateway, optional engines.** A single MCP server process; heavy engines (spaCy, NLTK, OpenHowNet) are optional extras. `jieba` is the only mandatory dependency, so Chinese works out of the box.
3. **Unified output schema.** Every tool returns the same envelope with honest engine attribution and exact character offsets.
4. **Graceful degradation, never silent lies.** Missing optional dependency → an actionable install hint (not a crash); fallback engines are labeled as such (`regex-fallback`).
5. **mcp SDK 1.x and 2.x compatible.** The `mcp` 2.0 release renamed `FastMCP` → `MCPServer` and removed the old import path; a compat shim ([`src/classic_nlp_mcp/_compat.py`](src/classic_nlp_mcp/_compat.py)) supports both lines.

## Architecture

```
 MCP clients (Claude / WorkBuddy / Cursor / ...)
        │
        ▼
 classic-nlp-mcp gateway  ── one process, capability routing, unified schema
        │
 ┌──────┼───────────┬──────────────┐
 ▼      ▼           ▼              ▼
spaCy  jieba       NLTK        OpenHowNet
(optional, base) (optional)   (optional)
```

## Tools

| Tool | Capability | Engine (zh / en) |
|---|---|---|
| `list_capabilities` | What is installed, per language | — |
| `tokenize` | Segmentation with exact offsets | jieba / spaCy → stanza → regex fallback |
| `pos_tag` | Part-of-speech tagging (+lemma) | jieba.posseg / spaCy → stanza |
| `ner` | Named entities with spans | stanza statistical + jieba heuristic + spaCy / spaCy → stanza |
| `keyword_extract` | TF-IDF / TextRank with weights | jieba.analyse / frequency fallback |
| `sent_split` | Sentence segmentation (RAG chunking base) | jieba+regex / spaCy → stanza → regex fallback |
| `openie_triples` | OpenIE (S, P, O) triples with source span & confidence | jieba rule (zh, draft quality) / **Stanford CoreNLP** (canonical) → spaCy dep rules |
| `constituency_parse` | Phrase-structure trees + node labels | stanza (Stanford stack exclusive) |
| `seme_lookup` | HowNet sememes (义原) lookup | OpenHowNet (zh) |
| `wordnet_lookup` | Synsets, definitions, hypernyms | NLTK WordNet (en) |
| `wordnet_similarity` | WordNet path similarity | NLTK WordNet (en) |
| `annotate_full` | All of the above in **one call**, per-capability error isolation | orchestration |

**Multilingual**: stanza powers `tokenize` / `pos_tag` / `ner` / `sent_split` / `constituency_parse` for **any stanza language code** (`fr`, `de`, `ja`, `es`, ...) - just pass it as `lang` once that language's models are downloaded.

### Unified output schema

Every tool returns the same envelope:

```json
{
  "tool": "tokenize",
  "lang": "zh",
  "engine": "jieba",
  "engine_version": "0.42.1",
  "tokens": [{"text": "知识", "start": 2, "end": 4, "pos": "n"}]
}
```

`openie_triples` additionally returns `confidence` and the source `span` + `sentence`, so triples are *candidates with provenance* — ready for ontology alignment or LLM verification downstream.

## Compatibility & versions

Written against the **current** APIs of every wrapped toolkit; the pinned ranges deliberately include the latest releases:

| Package | Supported range | Latest at time of writing | Notes |
|---|---|---|---|
| Python | >= 3.10 | 3.13 / 3.14 | spaCy wheels cover 3.9–3.14 |
| [mcp](https://pypi.org/project/mcp/) SDK | `>=1.2, <3` | 2.0.x | **2.0 renamed `FastMCP` → `MCPServer`** and removed the old import path; the compat shim (`src/classic_nlp_mcp/_compat.py`) supports both majors |
| [jieba](https://pypi.org/project/jieba/) | `>=0.42.1` | 0.42.1 (stable, pure Python) | base dependency — Chinese works out of the box |
| [spaCy](https://pypi.org/project/spacy/) | `>=3.7` (extras `en`) | 3.8.x (2026-08) | current `spacy.load` / `Doc` / `Span` API; `en_core_web_sm`, `zh_core_web_sm` both supported |
| [stanza](https://pypi.org/project/stanza/) | `>=1.8` (extras `stanford`) | 1.15.x | written against 1.15 (models under the platform cache dir); any language code works once its models are downloaded. **No released stanza version ships OpenIE** - use the CoreNLP bridge for that |
| Stanford [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) | any 4.x server | 4.5.10 | optional: canonical OpenIE via local `StanfordCoreNLPServer` (Java, `CORENLP_URL`, default `localhost:9000`); degrades gracefully when the server is down |
| [NLTK](https://pypi.org/project/nltk/) | `>=3.8` (extras `semantic`) | 3.10.x | `nltk.corpus.wordnet` API is stable across 3.x |
| [OpenHowNet](https://pypi.org/project/OpenHowNet/) | any (extras `semantic`) | 0.0.x (2020 release, data bundled) | last upstream release is old; wrapped defensively (`try_import` + graceful degradation) so it can never break the server |

Everything optional degrades with an actionable install hint — the server starts and stays useful even with only `jieba` installed.

## Installation

```bash
# base (Chinese works immediately)
pip install classic-nlp-mcp

# optional engines
pip install "classic-nlp-mcp[en]"         # spaCy
pip install "classic-nlp-mcp[semantic]"   # NLTK + OpenHowNet
pip install "classic-nlp-mcp[all]"

python -m spacy download en_core_web_sm   # spaCy English model
# optional: python -m spacy download zh_core_web_sm  (statistical Chinese NER)
# optional: python -c "import nltk; nltk.download('wordnet')"
```

Or from source:

```bash
git clone https://github.com/liftkkkk/classic-nlp-mcp.git
cd classic-nlp-mcp
pip install -e ".[all,dev]"
pytest
```

## Plug it into your agent host (MCP configuration)

classic-nlp-mcp is a plain **stdio** MCP server, so any host that speaks MCP works. Development and end-to-end testing were done with **Z.ai GLM-5.3** as the driving agent.

### Claude Desktop / Cursor / Cline / any `mcpServers` client

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

### Z Code (Z.ai, GLM Coding Plan / GLM-5.3)

[Z Code](https://docs.z.ai/) consumes the same standard `mcpServers` block — add it to Z Code's MCP settings and the GLM-5.3 agent can call `tokenize`, `openie_triples`, `annotate_full` etc. directly from your coding sessions:

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

Typical Z Code loop: `annotate_full` on a module's docstrings → keep `openie_triples` with `confidence ≥ 0.35` → let GLM-5.3 normalize them against your ontology. Cheap deterministic extraction first, expensive model reasoning only where it pays.

### WorkBuddy

Add the same block to `~/.workbuddy/mcp.json`, then trust the server in the connector management page.

### Or use the installed console script

```json
{
  "mcpServers": {
    "classic-nlp": { "command": "classic-nlp-mcp" }
  }
}
```

## Example agent workflows

- **RAG ingestion**: `sent_split` → chunk by sentence spans → `keyword_extract` for per-chunk metadata.
- **KG construction pipeline**: `annotate_full` on a corpus → keep `openie_triples` with `confidence ≥ 0.35` → LLM normalizes S/P/O against your ontology → graph.
- **Token-saving funnel**: `ner` first; only send candidate entities to the LLM for typing/linking.
- **Interpretable similarity**: `seme_lookup` (sememes) or `wordnet_similarity` (path similarity) before embedding-based similarity.

## v0.2 - the Stanford stack is in

Two additions, because the Stanford toolchain is still **irreplaceable** in specific niches:

- **[stanza](https://github.com/stanfordnlp/stanza)** adapter (optional `[stanford]` extra): neural statistical NER, POS + lemma, sentence splitting for **60+ languages** (pass any stanza language code as `lang`), and `constituency_parse` - phrase-structure trees that no other wrapped engine provides. Note: no released stanza version ships OpenIE - it lives in CoreNLP, which is why the next line exists.
- **Stanford CoreNLP OpenIE bridge**: the *canonical* OpenIE implementation. Run a local `StanfordCoreNLPServer` (Java) and this MCP uses it automatically for English triples - highest-quality candidate triples in the project, still with provenance spans. When the server is down, English falls back to spaCy dependency rules and `list_capabilities` reports the true status.

```bash
pip install "classic-nlp-mcp[stanford]"
python -c "import stanza; stanza.download('en')"
# optional, for canonical OpenIE (Java required):
#   download CoreNLP from https://stanfordnlp.github.io/CoreNLP/download.html and start:
#   java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

## Upstream projects this stands on

- [spaCy](https://github.com/explosion/spaCy) — industrial-strength multilingual NLP · [docs](https://spacy.io/)
- [jieba](https://github.com/fxsjy/jieba) — the classic Chinese segmentation library
- [NLTK](https://github.com/nltk/nltk) · [docs](https://www.nltk.org/) — WordNet & the historical standard library of NLP
- [OpenHowNet](https://github.com/thunlp/OpenHowNet) · [HowNet homepage (keenage.com)](https://www.keenage.com/) — THUNLP's sememe dictionary (义原词典)
- [stanza](https://github.com/stanfordnlp/stanza) · [docs](https://stanfordnlp.github.io/stanza/) — Stanford NLP group's Python library
- [Stanford CoreNLP](https://github.com/stanfordnlp/CoreNLP) · [site](https://stanfordnlp.github.io/CoreNLP/) — the Java toolkit behind classic OpenIE (JVM; can be wrapped with the same adapter pattern)
- [Model Context Protocol](https://github.com/modelcontextprotocol) · [spec & docs](https://modelcontextprotocol.io/) — the protocol this server speaks

> Note: stanza ships as the `[stanford]` extra (models downloaded separately per language); CoreNLP integrates through its HTTP server - see the v0.2 section above.

## Roadmap

- [ ] CoreNLP adapter hardening: auto-start the server, Chinese OpenIE rules
- [ ] stanza zh model CI smoke test
- [ ] LRU/disk result cache keyed by `(text, lang, capability, engine_version)`
- [ ] RDF / JSON-LD export aligned with custom ontologies
- [ ] `compare_engines` tool (same capability, multiple engines, aligned output)

## Buy me a coffee

classic-nlp-mcp is an open-source project I develop and maintain in my spare time, free forever.

If it gave your agent deterministic language sensors, cheap KG preprocessing, or saved you the trouble of gluing five NLP libraries together — consider buying me a coffee (9.9 CNY is enough). Every bit of support goes straight into new features and bug fixes.

<p align="center">
  <img src="donate.jpg" alt="Buy me a coffee" width="280">
</p>

<p align="center"><i>Leave a note with your donation about the feature you want most - I prioritize those ;)</i></p>

**Can't donate?** Starring the repo, opening an Issue, or sharing it with someone who needs it helps just as much!

## License

[MIT](LICENSE) © 2026 liftkkkk
