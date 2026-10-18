# classic-nlp-mcp

**Deterministic classical NLP, served over MCP — give your LLM agent cheap, reproducible, offline language sensors.**

[English](README.md) | [中文文档](README.zh-CN.md)

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
| `tokenize` | Segmentation with exact offsets | jieba / spaCy → regex fallback |
| `pos_tag` | Part-of-speech tagging | jieba.posseg / spaCy |
| `ner` | Named entities with spans | jieba heuristic + spaCy / spaCy |
| `keyword_extract` | TF-IDF / TextRank with weights | jieba.analyse / frequency fallback |
| `sent_split` | Sentence segmentation (RAG chunking base) | jieba+regex / spaCy → regex fallback |
| `openie_triples` | OpenIE (S, P, O) triples with source span & confidence | jieba rule (zh, draft quality) / spaCy dep rules (en) |
| `seme_lookup` | HowNet sememes (义原) lookup | OpenHowNet (zh) |
| `wordnet_lookup` | Synsets, definitions, hypernyms | NLTK WordNet (en) |
| `wordnet_similarity` | WordNet path similarity | NLTK WordNet (en) |
| `annotate_full` | All of the above in **one call**, per-capability error isolation | orchestration |

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

## MCP client configuration

Generic `mcp.json` (Works with Claude Desktop, WorkBuddy, Cursor, Cline, ...):

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

Or use the installed console script:

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

## Upstream projects this stands on

- [spaCy](https://github.com/explosion/spaCy) — industrial-strength multilingual NLP · [docs](https://spacy.io/)
- [jieba](https://github.com/fxsjy/jieba) — the classic Chinese segmentation library
- [NLTK](https://github.com/nltk/nltk) · [docs](https://www.nltk.org/) — WordNet & the historical standard library of NLP
- [OpenHowNet](https://github.com/thunlp/OpenHowNet) · [HowNet homepage (keenage.com)](https://www.keenage.com/) — THUNLP's sememe dictionary (义原词典)
- [stanza](https://github.com/stanfordnlp/stanza) · [docs](https://stanfordnlp.github.io/stanza/) — Stanford NLP group's Python library
- [Stanford CoreNLP](https://github.com/stanfordnlp/CoreNLP) · [site](https://stanfordnlp.github.io/CoreNLP/) — the Java toolkit behind classic OpenIE (JVM; can be wrapped with the same adapter pattern)
- [Model Context Protocol](https://github.com/modelcontextprotocol) · [spec & docs](https://modelcontextprotocol.io/) — the protocol this server speaks

> Note: stanza / CoreNLP are not bundled as adapters yet — they are on the roadmap (see below). The links are acknowledgements of the toolkits this project's design builds upon.

## Roadmap

- [ ] stanza adapter (neural multilingual pipeline + native OpenIE)
- [ ] CoreNLP JVM bridge for production-grade English/Chinese OpenIE
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
