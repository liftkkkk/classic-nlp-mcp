import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest  # noqa: E402


# ---------------------------------------------------------------------------
# compat shim (the mcp 2.0 lesson learned the hard way)
# ---------------------------------------------------------------------------


def test_compat_shim_imports():
    from classic_nlp_mcp._compat import FastMCP, MCP_SDK_MAJOR

    assert FastMCP is not None
    assert MCP_SDK_MAJOR in (1, 2)


def test_server_module_registers_tools():
    from classic_nlp_mcp.server import MCP_SDK_MAJOR, mcp

    assert mcp is not None
    assert MCP_SDK_MAJOR in (1, 2)


# ---------------------------------------------------------------------------
# pure logic
# ---------------------------------------------------------------------------


def test_detect_lang():
    from classic_nlp_mcp.adapters.base import detect_lang

    assert detect_lang("这是一个中文句子") == "zh"
    assert detect_lang("This is English text.") == "en"
    assert detect_lang("") == "en"


def test_unified_schema():
    from classic_nlp_mcp.adapters.base import make_result

    r = make_result("tokenize", "zh", "jieba", "0.42", tokens=[])
    assert set(r) == {"tool", "lang", "engine", "engine_version", "tokens"}


# ---------------------------------------------------------------------------
# jieba (base dependency - must always work)
# ---------------------------------------------------------------------------

ZH_TEXT = "朱洪银在新华三负责AI网关的Token费用预测项目。"


def test_jieba_tokenize_offsets():
    from classic_nlp_mcp.server import tokenize

    r = tokenize(ZH_TEXT, "zh")
    assert r["engine"].startswith("jieba")
    for tok in r["tokens"]:
        assert ZH_TEXT[tok["start"]:tok["end"]] == tok["text"]


def test_jieba_pos_and_ner():
    from classic_nlp_mcp.server import ner, pos_tag

    pos = pos_tag(ZH_TEXT, "zh")
    assert any(t.get("pos") for t in pos["tokens"])

    ents = ner(ZH_TEXT, "zh")
    labels = {e["label"] for e in ents["entities"]}
    assert labels & {"PER", "ORG", "LOC", "MISC"}


def test_jieba_openie():
    from classic_nlp_mcp.server import openie_triples

    r = openie_triples("公司发布了新产品。", "zh")
    assert r["supported"] is True
    assert isinstance(r["triples"], list)


def test_keyword_and_split_zh():
    from classic_nlp_mcp.server import keyword_extract, sent_split

    kw = keyword_extract("知识图谱与大模型结合是未来的重要方向，图谱技术很有价值。", top_k=5)
    assert 0 < len(kw["keywords"]) <= 5
    sents = sent_split("第一句话。第二句话！第三句话？", "zh")
    assert len(sents["sentences"]) == 3


# ---------------------------------------------------------------------------
# English path (works with or without spaCy)
# ---------------------------------------------------------------------------

EN_TEXT = "Apple hired John as VP of Engineering in 2020."


def test_en_tokenize_never_fails():
    from classic_nlp_mcp.server import tokenize

    r = tokenize(EN_TEXT, "en")
    assert r["tokens"][0]["text"] in ("Apple", "This")  # spaCy or fallback
    for tok in r["tokens"]:
        assert EN_TEXT[tok["start"]:tok["end"]] == tok["text"]


def test_en_sent_split_never_fails():
    from classic_nlp_mcp.server import sent_split

    r = sent_split("First sentence. Second one! Third? ", "en")
    assert len(r["sentences"]) == 3


def test_en_missing_optional_raises_hint():
    from classic_nlp_mcp.adapters.base import MissingDependency
    from classic_nlp_mcp.server import REGISTRY

    en_spacy = REGISTRY.get("spacy", "en")
    if en_spacy and en_spacy.available:
        pytest.skip("spaCy installed")
    from classic_nlp_mcp.server import ner

    with pytest.raises(MissingDependency) as excinfo:
        ner(EN_TEXT, "en")
    assert "pip install" in str(excinfo.value)


# ---------------------------------------------------------------------------
# annotate_full resilience
# ---------------------------------------------------------------------------


def test_annotate_full_merges_and_reports_errors():
    from classic_nlp_mcp.server import annotate_full

    r = annotate_full(ZH_TEXT, "zh")
    assert "tokenize" in r["results"]
    assert "ner" in r["results"]
    # English-only or optional caps should degrade into errors, not crash
    combined = set(r["results"]) | set(r["errors"])
    assert {"pos_tag", "keyword_extract", "sent_split", "openie_triples"} <= combined


# ---------------------------------------------------------------------------
# MCP protocol end-to-end (in-memory client), mirroring how WorkBuddy loads it
# ---------------------------------------------------------------------------


def test_mcp_in_memory_handshake():
    pytest.importorskip("mcp")
    try:
        import anyio

        return anyio.run(_handshake)
    except ImportError:
        pytest.skip("anyio not available")


async def _handshake():
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "classic_nlp_mcp"], env=env
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            expected = {
                "list_capabilities", "tokenize", "pos_tag", "ner", "keyword_extract",
                "sent_split", "openie_triples", "seme_lookup", "wordnet_lookup",
                "wordnet_similarity", "annotate_full",
            }
            assert expected <= names
            res = await session.call_tool(
                "tokenize", {"text": "打开知识图谱的门。", "lang": "zh"}
            )
            payload = json.loads(res.content[0].text)
            assert payload["lang"] == "zh"
            assert payload["tokens"]
