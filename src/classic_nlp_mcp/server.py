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
    """'auto'/empty -> detect zh/en; any other value is kept verbatim so that
    callers can pass a stanza language code (fr, de, ja, ...)."""
    if lang in ("auto", "", None):
        return detect_lang(text)
    return lang


def _stanza_for(lang: str):
    """Stanza adapter for *lang* or None (not installed / models missing)."""
    return REGISTRY.get("stanza", lang)


def _engine_name(lang: str, engine: str | None, default: str) -> str:
    return engine or default


_ALLOWED_ENGINES = ("jieba", "spacy", "stanza")


def _pick_engine(engine: str, tool: str) -> str:
    """Validate an engine request.  Unknown values raise instead of being
    silently ignored - the mcp SDK drops extra kwargs, so an unvalidated
    engine used to disappear without a trace."""
    engine = (engine or "auto").strip().lower()
    if engine != "auto" and engine not in _ALLOWED_ENGINES:
        raise ValueError(
            f"unknown engine {engine!r} for {tool}; allowed: auto, "
            + ", ".join(_ALLOWED_ENGINES))
    return engine


def _zh_adapter(name: str):
    if name == "jieba":
        return REGISTRY.get("jieba", "zh")
    if name == "spacy":
        adapter = REGISTRY.get("spacy", "zh")
        return adapter if adapter and adapter.available else None
    return _stanza_for("zh")


def _adapter_for(lang: str, engine: str, cap: str):
    """Resolve an adapter honoring an explicit engine request.

    ``engine='auto'`` keeps the documented per-capability priority.  An
    explicit engine is honored when installed and raises MissingDependency
    with an actionable install hint otherwise - a forced engine is never
    silently downgraded (fixed in v0.2.3; before that, tokenize/pos_tag/ner
    had no engine parameter at all and spaCy always shadowed stanza)."""
    if lang == "zh":
        priority = ("stanza", "spacy", "jieba") if cap == "ner" \
            else ("jieba", "spacy", "stanza")
        if engine == "auto":
            for name in priority:
                adapter = _zh_adapter(name)
                if adapter is not None:
                    return adapter
            return REGISTRY.get("jieba", "zh")
        if engine not in priority and engine != "jieba":
            raise ValueError(
                f"engine {engine!r} is not a Chinese engine for {cap}; "
                f"allowed: auto, {', '.join(priority)}")
        adapter = _zh_adapter(engine)
        if adapter is None:
            raise MissingDependency(engine)
        return adapter
    # non-Chinese
    if engine == "jieba":
        raise ValueError(
            "engine 'jieba' is Chinese-only; use auto | spacy | stanza here")
    if engine in ("auto", "spacy"):
        adapter = REGISTRY.get("spacy", "en")
        if adapter and adapter.available:
            return adapter
        if engine == "spacy":
            raise MissingDependency("spacy")
    stanza = _stanza_for(lang)
    if stanza is not None:
        return stanza
    if engine == "stanza":
        raise MissingDependency("stanza")
    return None


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
    stanza_ok = REGISTRY.engine_available("stanza")
    corenlp_ok = REGISTRY.engine_available("corenlp")
    en_stanza = bool(stanza_ok and REGISTRY.get("stanza", "en"))
    zh_stanza = bool(stanza_ok and REGISTRY.get("stanza", "zh"))
    en_spacy = bool(spacy_ok and REGISTRY.get("spacy", "en"))
    zh_spacy = bool(spacy_ok and REGISTRY.get("spacy", "zh"))

    caps = {
        "mcp_sdk_major": MCP_SDK_MAJOR,
        "positioning": (
            "Chinese-first bilingual: zh and en are equally first-class. The base "
            "install already runs the full Chinese pipeline (jieba); extras 'zh' "
            "(zh_core_web_sm) upgrades Chinese NER to spaCy statistics."
        ),
        "extras": {"zh": "spacy>=3.7 + zh_core_web_sm", "en": "spacy>=3.7 + en_core_web_sm",
                   "semantic": "nltk + OpenHowNet", "stanford": "stanza>=1.8"},
        "engines": {
            "jieba": {"installed": True, "optional": False,
                      "note": "base dependency - full Chinese pipeline out of the box"},
            "spacy": {"installed": spacy_ok, "optional": True,
                      "hint": INSTALL_HINTS["spacy"]},
            "stanza": {"installed": stanza_ok, "optional": True,
                       "note": "Stanford NLP neural pipeline: statistical NER, "
                               "constituency parsing, 60+ languages (no OpenIE - "
                               "OpenIE lives in CoreNLP)",
                       "hint": INSTALL_HINTS["stanza"]},
            "corenlp": {"installed": corenlp_ok, "optional": True,
                        "note": "canonical Stanford OpenIE via a local CoreNLP HTTP server",
                        "hint": INSTALL_HINTS.get(
                            "corenlp",
                            "Start StanfordCoreNLPServer on :9000 or set CORENLP_URL; "
                            "see https://stanfordnlp.github.io/CoreNLP/download.html")},
            "nltk": {"installed": nltk_ok, "optional": True,
                     "hint": INSTALL_HINTS["nltk"]},
            "openhownet": {"installed": hownet_ok, "optional": True,
                           "hint": INSTALL_HINTS["OpenHowNet"]},
        },
        "capabilities": {
            "tokenize": {"zh": "jieba (regex fallback)",
                         "en": "spacy / stanza / regex fallback",
                         "multilingual": "stanza (any downloaded lang code)",
                         "engine": "auto | jieba | spacy | stanza (forced engine raises a hint when unavailable)"},
            "pos_tag": {"zh": "jieba.posseg / stanza",
                        "en": "spacy / stanza",
                        "multilingual": "stanza",
                        "engine": "auto | jieba | spacy | stanza (engine='stanza' adds lemma)"},
            "ner": {"zh": ("stanza statistical" if zh_stanza
                           else "spacy statistical ([zh] extra)" if zh_spacy
                           else "jieba heuristic"),
                    "en": "spacy / stanza",
                    "multilingual": "stanza",
                    "engine": "auto | jieba | spacy | stanza"},
            "keyword_extract": {"zh": "jieba TF-IDF/TextRank", "en": "frequency fallback"},
            "sent_split": {"zh": "jieba+regex", "en": "spacy / stanza / regex fallback"},
            "openie_triples": {
                "zh": "jieba rule-based (draft quality)",
                "en": "stanford-corenlp server (canonical)" if corenlp_ok
                      else ("spacy dependency rules" if en_spacy else "unavailable"),
                "note": "English prefers the canonical CoreNLP OpenIE when a local "
                        "CoreNLP server is running; engine='corenlp' forces it",
            },
            "constituency_parse": {
                "zh": "stanza" if zh_stanza else "unavailable",
                "en": "stanza" if en_stanza else "unavailable",
                "note": "phrase-structure trees - exclusive to the Stanford stack here",
            },
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
def tokenize(text: str, lang: str = "auto", engine: str = "auto") -> dict[str, Any]:
    """Segment text into tokens with exact character offsets.

    Chinese: jieba by default; English: spaCy when installed, then stanza,
    then a regex fallback.  ``engine`` may force one: "jieba" | "spacy" |
    "stanza" - an unavailable forced engine raises an actionable install
    hint instead of being silently ignored.  ``lang``: "auto" | "zh" | "en"
    | any stanza language code (fr, de, ja, ...).
    """
    lang = _resolve_lang(text, lang)
    engine = _pick_engine(engine, "tokenize")
    adapter = _adapter_for(lang, engine, "tokenize")
    if adapter is not None:
        return adapter.tokenize(text)
    if lang != "en":
        raise MissingDependency("stanza")
    return fallback.fallback_tokenize(text, "en")


@mcp.tool()
def pos_tag(text: str, lang: str = "auto", engine: str = "auto") -> dict[str, Any]:
    """Part-of-speech tagging (+ lemma with stanza).  Chinese: jieba.posseg;
    English: spaCy then stanza; other languages: stanza.  ``engine`` may
    force one: "jieba" | "spacy" | "stanza" - use engine="stanza" to get
    UPOS + lemma even when spaCy is installed.  An unavailable forced
    engine raises an actionable install hint instead of being silently
    ignored."""
    lang = _resolve_lang(text, lang)
    engine = _pick_engine(engine, "pos_tag")
    adapter = _adapter_for(lang, engine, "pos_tag")
    if adapter is not None:
        return adapter.pos_tag(text)
    raise MissingDependency("stanza" if lang != "en" else "spacy")


@mcp.tool()
def ner(text: str, lang: str = "auto", engine: str = "auto") -> dict[str, Any]:
    """Named entity recognition with exact spans.

    Chinese: stanza statistical NER when installed (recommended), otherwise
    the jieba POS-tag heuristic (nr/ns/nt -> PER/LOC/ORG).  English: spaCy,
    then stanza.  Other languages: stanza.  ``engine`` may force one:
    "jieba" | "spacy" | "stanza" - an unavailable forced engine raises an
    actionable install hint instead of being silently ignored."""
    lang = _resolve_lang(text, lang)
    engine = _pick_engine(engine, "ner")
    adapter = _adapter_for(lang, engine, "ner")
    if adapter is not None:
        return adapter.ner(text)
    raise MissingDependency("stanza" if lang != "en" else "spacy")


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
def sent_split(text: str, lang: str = "auto", engine: str = "auto") -> dict[str, Any]:
    """Sentence segmentation with offsets - the foundation for RAG chunking.
    Chinese: jieba + rule regex; English: spaCy sents by default, or the
    zero-dependency regex fallback with ``engine="regex"`` - use the fast
    path for bulk preprocessing where neural segmentation is not worth the
    CPU time."""
    lang = _resolve_lang(text, lang)
    if engine == "regex":
        return fallback.fallback_sent_split(text, lang)
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
def openie_triples(text: str, lang: str = "auto", engine: str = "auto") -> dict[str, Any]:
    """Open Information Extraction: (subject, predicate, object) triples.

    Engine priority for English (``engine`` may force one):
    1. ``corenlp``  - the canonical Stanford CoreNLP OpenIE via a local
       CoreNLP server (env CORENLP_URL, default localhost:9000) when it is up;
    2. ``spacy``    - dependency-parse rules (confidence 0.6).

    Chinese: naive (noun, verb, noun) pattern mining over jieba POS
    (confidence 0.35 - draft quality by design).  CoreNLP's OpenIE is
    English-focused.  Every triple carries the source character span and
    is a *candidate* for downstream KG normalization - not a fact."""
    lang = _resolve_lang(text, lang)
    if lang != "zh" and engine in ("auto", "corenlp"):
        corenlp = REGISTRY.get("corenlp", "en")
        if corenlp:
            return corenlp.openie_triples(text)
        if engine == "corenlp":
            from .adapters import corenlp_adapter

            raise RuntimeError(corenlp_adapter.setup_hint())
    if lang == "zh":
        result = REGISTRY.get("jieba", "zh").openie_triples(text)
        if engine == "corenlp":
            # forced CoreNLP on Chinese: honest degradation, never silent
            result["requested_engine"] = "corenlp"
            result["fallback"] = True
            result["note"] = ("CoreNLP OpenIE is English-only - the forced "
                              "engine='corenlp' fell back to jieba rule "
                              "mining for Chinese.  " + result.get("note", ""))
        return result
    adapter = REGISTRY.get("spacy", "en")
    if adapter and adapter.available:
        return adapter.openie_triples(text)
    raise MissingDependency("spacy")


# ---------------------------------------------------------------------------
# constituency parsing (Stanford stack exclusive)
# ---------------------------------------------------------------------------


@mcp.tool()
def constituency_parse(text: str, lang: str = "auto", max_trees: int = 5) -> dict[str, Any]:
    """Phrase-structure (constituency) parses via stanza - exclusive to the
    Stanford stack among the wrapped engines.  Returns bracketed trees plus
    the set of node labels per sentence.  Requires the language's stanza
    models with the 'constituency' processor."""
    lang = _resolve_lang(text, lang)
    st = _stanza_for(lang)
    if st:
        return st.constituency_parse(text, max_trees=max_trees)
    raise MissingDependency("stanza")


# ---------------------------------------------------------------------------
# semantic resources
# ---------------------------------------------------------------------------


def _semantic_unavailable(tool: str, dep: str) -> dict[str, Any]:
    """Graceful envelope instead of raising through the tool boundary - the
    mcp SDK turns raw exceptions into opaque ``Error executing tool`` text."""
    return make_result(tool, "zh" if tool == "seme_lookup" else "en", "unavailable", "-",
                       supported=False, note=INSTALL_HINTS[dep])


@mcp.tool()
def seme_lookup(word: str, lang: str = "auto") -> dict[str, Any]:
    """HowNet sememe (义原) lookup for a Chinese word via OpenHowNet.
    Fine-grained interpretable lexical semantics - useful for entity typing
    and sememe-based similarity."""
    lang = _resolve_lang(word, lang)
    adapter = REGISTRY.get("openhownet", "zh" if lang == "zh" else "zh")
    if adapter and _hownet_ready():
        return adapter.seme_lookup(word)
    return _semantic_unavailable("seme_lookup", "OpenHowNet")


@mcp.tool()
def wordnet_lookup(word: str, max_synsets: int = 5) -> dict[str, Any]:
    """WordNet synsets, definitions, synonyms and hypernyms for an English
    word via NLTK."""
    adapter = REGISTRY.get("nltk", "en")
    if adapter and _nltk_ready():
        return adapter.wordnet_lookup(word, max_synsets=max_synsets)
    return _semantic_unavailable("wordnet_lookup", "nltk")


@mcp.tool()
def wordnet_similarity(word_a: str, word_b: str) -> dict[str, Any]:
    """Best WordNet path similarity between two English words via NLTK."""
    adapter = REGISTRY.get("nltk", "en")
    if adapter and _nltk_ready():
        return adapter.wordnet_similarity(word_a, word_b)
    return _semantic_unavailable("wordnet_similarity", "nltk")


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
