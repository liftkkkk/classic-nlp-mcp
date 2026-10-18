"""classic-nlp-mcp server: classical NLP toolkits exposed as MCP tools.

One gateway process, capability-level tools, pluggable engines.  Works with
the ``mcp`` SDK 1.x (FastMCP) and 2.x (MCPServer) via ``_compat``.
"""

from __future__ import annotations

import re
from typing import Any

from ._compat import FastMCP, MCP_SDK_MAJOR
from .adapters.base import (
    INSTALL_HINTS,
    MissingDependency,
    REGISTRY,
    detect_lang,
    make_result,
)
from . import fallback

mcp = FastMCP(
    "classic-nlp-mcp",
    instructions=(
        "Deterministic classical NLP (spaCy / jieba / NLTK / OpenHowNet) exposed as "
        "capability-level MCP tools. Use for cheap, reproducible, offline language "
        "analysis: tokenization, POS, NER, keywords, sentence splitting, OpenIE "
        "triples for KG construction, sememe/WordNet lookups. All results carry "
        "exact character offsets and an honest engine attribution."
    ),
)

_ENGLISH_TAG_RE = re.compile(r"^[a-zA-Z]")


def _resolve_lang(text: str, lang: str) -> str:
    return detect_lang(text) if lang in ("auto", "", None) else lang


def _engine_name(lang: str, engine: str | None, default: str) -> str:
    return engine or default


# ---------------------------------------------------------------------------
# list_capabilities
# ---------------------------------------------------------------------------


@mcp.tool()
def list_capabilities() -> dict[str, Any]:
    """Report which NLP engines are installed and which capabilities each
    language supports.  Always call this first if unsure what is available."""
    spacy_ok = REGISTRY.engine_available("spacy")
    nltk_ok = REGISTRY.engine_available("nltk")
    hownet_ok = REGISTRY.engine_available("openhownet")

    zh_stats_ner = bool(spacy_ok and REGISTRY.get("spacy", "zh"))
    en_pipeline = bool(spacy_ok and REGISTRY.get("spacy", "en"))

    caps = {
        "mcp_sdk_major": MCP_SDK_MAJOR,
        "engines": {
            "jieba": {"installed": True, "optional": False},
            "spacy": {"installed": spacy_ok, "optional": True,
                      "hint": INSTALL_HINTS["spacy"]},
            "nltk": {"installed": nltk_ok, "optional": True,
                     "hint": INSTALL_HINTS["nltk"]},
            "openhownet": {"installed": hownet_ok, "optional": True,
                           "hint": INSTALL_HINTS["OpenHowNet"]},
        },
        "capabilities": {
            "tokenize": {"zh": "jieba (regex fallback)", "en": "spacy or regex fallback"},
            "pos_tag": {"zh": "jieba.posseg", "en": "spacy" if en_pipeline else "unavailable"},
            "ner": {"zh": "jieba heuristic" + (" / spacy" if zh_stats_ner else ""),
                    "en": "spacy" if en_pipeline else "unavailable"},
            "keyword_extract": {"zh": "jieba TF-IDF/TextRank", "en": "frequency fallback"},
            "sent_split": {"zh": "jieba+regex", "en": "regex fallback"},
            "openie_triples": {"zh": "jieba rule-based (draft quality)",
                               "en": "spacy dependency rules" if en_pipeline else "unavailable"},
            "wordnet_lookup": {"en": "nltk" if (nltk_ok and REGISTRY.get("nltk", "en")) else "unavailable"},
            "wordnet_similarity": {"en": "nltk" if (nltk_ok and REGISTRY.get("nltk", "en")) else "unavailable"},
            "seme_lookup": {"zh": "OpenHowNet" if (hownet_ok and REGISTRY.get("openhownet", "zh")) else "unavailable"},
            "annotate_full": {"zh": "best available per capability",
                              "en": "best available per capability"},
        },
    }
    return caps


# ---------------------------------------------------------------------------
# tokenize / pos / ner
# ---------------------------------------------------------------------------


@mcp.tool()
def tokenize(text: str, lang: str = "auto") -> dict[str, Any]:
    """Segment text into tokens with exact character offsets.

    Chinese uses jieba; English uses spaCy when installed, otherwise a
    regex fallback.  ``lang``: "auto" | "zh" | "en".
    """
    lang = _resolve_lang(text, lang)
    if lang == "zh":
        return REGISTRY.get("jieba", "zh").tokenize(text)
    adapter = REGISTRY.get("spacy", "en")
    if adapter and adapter.available:
        return adapter.tokenize(text)
    return fallback.fallback_tokenize(text, "en")


@mcp.tool()
def pos_tag(text: str, lang: str = "auto") -> dict[str, Any]:
    """Part-of-speech tagging.  Chinese: jieba.posseg; English: spaCy
    (raises an actionable install hint when spaCy is missing)."""
    lang = _resolve_lang(text, lang)
    if lang == "zh":
        return REGISTRY.get("jieba", "zh").pos_tag(text)
    adapter = REGISTRY.get("spacy", "en")
    if adapter and adapter.available:
        return adapter.pos_tag(text)
    raise MissingDependency("spacy")


@mcp.tool()
def ner(text: str, lang: str = "auto") -> dict[str, Any]:
    """Named entity recognition with exact spans.

    Chinese: jieba POS-tag heuristic (nr/ns/nt -> PER/LOC/ORG); English:
    spaCy statistical NER.  Install the spaCy zh model for statistical
    Chinese NER."""
    lang = _resolve_lang(text, lang)
    if lang == "zh":
        zh_spacy = REGISTRY.get("spacy", "zh")
        if zh_spacy and zh_spacy.available:
            return zh_spacy.ner(text)
        return REGISTRY.get("jieba", "zh").ner(text)
    adapter = REGISTRY.get("spacy", "en")
    if adapter and adapter.available:
        return adapter.ner(text)
    raise MissingDependency("spacy")


# ---------------------------------------------------------------------------
# keywords / sentences
# ---------------------------------------------------------------------------


@mcp.tool()
def keyword_extract(text: str, top_k: int = 10, method: str = "tfidf") -> dict[str, Any]:
    """Keyword extraction with weights.  Chinese: jieba TF-IDF or TextRank.
    English: word-frequency fallback (honest, cheap, deterministic)."""
    lang = detect_lang(text)
    if lang == "zh":
        return REGISTRY.get("jieba", "zh").keyword_extract(text, top_k=top_k, method=method)
    return fallback.fallback_keyword_extract(text, "en", top_k)


@mcp.tool()
def sent_split(text: str, lang: str = "auto") -> dict[str, Any]:
    """Sentence segmentation with offsets - the foundation for RAG chunking.
    Chinese: jieba + rule regex; English: spaCy sents when available."""
    lang = _resolve_lang(text, lang)
    if lang == "zh":
        return REGISTRY.get("jieba", "zh").sent_split(text)
    adapter = REGISTRY.get("spacy", "en")
    if adapter and adapter.available:
        try:
            return adapter.sent_split(text)
        except Exception:
            pass
    return fallback.fallback_sent_split(text, "en")


# ---------------------------------------------------------------------------
# OpenIE
# ---------------------------------------------------------------------------


@mcp.tool()
def openie_triples(text: str, lang: str = "auto") -> dict[str, Any]:
    """Open Information Extraction: (subject, predicate, object) triples.

    English: dependency-parse rules over spaCy (confidence 0.6).
    Chinese: naive (noun, verb, noun) pattern mining over jieba POS
    (confidence 0.35 - draft quality by design, meant as candidate triples
    for knowledge-graph construction, to be verified by an LLM or ontology
    downstream).  Every triple carries the source character span."""
    lang = _resolve_lang(text, lang)
    if lang == "zh":
        return REGISTRY.get("jieba", "zh").openie_triples(text)
    adapter = REGISTRY.get("spacy", "en")
    if adapter and adapter.available:
        return adapter.openie_triples(text)
    raise MissingDependency("spacy")


# ---------------------------------------------------------------------------
# semantic resources
# ---------------------------------------------------------------------------


@mcp.tool()
def seme_lookup(word: str, lang: str = "auto") -> dict[str, Any]:
    """HowNet sememe (义原) lookup for a Chinese word via OpenHowNet.
    Fine-grained interpretable lexical semantics - useful for entity typing
    and sememe-based similarity."""
    lang = _resolve_lang(word, lang)
    adapter = REGISTRY.get("openhownet", "zh" if lang == "zh" else "zh")
    if adapter:
        return adapter.seme_lookup(word)
    raise MissingDependency("OpenHowNet")


@mcp.tool()
def wordnet_lookup(word: str, max_synsets: int = 5) -> dict[str, Any]:
    """WordNet synsets, definitions, synonyms and hypernyms for an English
    word via NLTK."""
    adapter = REGISTRY.get("nltk", "en")
    if adapter and _nltk_ready():
        return adapter.wordnet_lookup(word, max_synsets=max_synsets)
    raise MissingDependency("nltk")


@mcp.tool()
def wordnet_similarity(word_a: str, word_b: str) -> dict[str, Any]:
    """Best WordNet path similarity between two English words via NLTK."""
    adapter = REGISTRY.get("nltk", "en")
    if adapter and _nltk_ready():
        return adapter.wordnet_similarity(word_a, word_b)
    raise MissingDependency("nltk")


def _nltk_ready() -> bool:
    from .adapters import nltk_adapter

    return nltk_adapter.corpus_ready()


def _hownet_ready() -> bool:
    from .adapters import openhownet_adapter

    return openhownet_adapter.dict_ready()


# ---------------------------------------------------------------------------
# annotate_full - the one-call pipeline
# ---------------------------------------------------------------------------


@mcp.tool()
def annotate_full(text: str, lang: str = "auto", with_semantics: bool = False) -> dict[str, Any]:
    """Run every available capability in one call and merge the results -
    one round trip instead of seven.

    Includes tokenize/pos/ner/keywords/sent_split/openie; adds
    seme_lookup/wordnet lookups when ``with_semantics`` is true and the
    corresponding libraries are installed.  Failed sub-capabilities are
    reported per-key under ``errors`` with actionable install hints
    instead of failing the whole call."""
    lang = _resolve_lang(text, lang)
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}

    def _run(key: str, fn, *args, **kwargs) -> None:
        try:
            results[key] = fn(*args, **kwargs)
        except MissingDependency as exc:
            errors[key] = str(exc)
        except Exception as exc:  # keep the pipeline resilient
            errors[key] = f"{type(exc).__name__}: {exc}"

    _run("tokenize", tokenize, text, lang)
    _run("pos_tag", pos_tag, text, lang)
    _run("ner", ner, text, lang)
    _run("keyword_extract", keyword_extract, text)
    _run("sent_split", sent_split, text, lang)
    _run("openie_triples", openie_triples, text, lang)
    if with_semantics:
        if lang == "zh":
            _run("seme_lookup", seme_lookup, text)
        else:
            _run("wordnet_lookup", wordnet_lookup, text)

    engines = sorted({r.get("engine", "?") for r in results.values()})
    return make_result(
        "annotate_full", lang, "+".join(engines) or "none", f"mcp-sdk-{MCP_SDK_MAJOR}",
        results=results, errors=errors,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
