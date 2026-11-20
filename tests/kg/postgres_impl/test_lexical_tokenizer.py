"""The lexical tokenizer's string contract.

Pins the rules in ``lightrag/kg/lexical_tokenizer.py``: CJK overlapping
bigrams, latin word tokens, NFKC + lowercase on both sides, tsquery
reserved characters neutralized by construction, bounded output, and
write/query token consistency. Expected values below are worked out by
hand from the rules, never recomputed with the code under test.

The live-PG proof that these strings parse as tsvector/tsquery belongs
to the requires_db phase; offline, the guarantee is the allowlist grammar:
emitted tokens are alphanumeric only, joined by spaces or `` & ``.
"""

import pytest

from lightrag.kg.lexical_tokenizer import (
    MAX_TOKEN_LENGTH,
    MAX_TOKENS,
    tokenize,
    tokenize_for_index,
    tokenize_for_query,
)

pytestmark = pytest.mark.offline

# Derived worst-case query length from the module's caps.
_MAX_QUERY_CHARS = MAX_TOKENS * MAX_TOKEN_LENGTH + 3 * (MAX_TOKENS - 1)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("数据库", "数据 据库"),
        ("数据库系统", "数据 据库 库系 系统"),
        ("中文分词", "中文 文分 分词"),
        ("数", "数"),
        ("数据", "数据"),
        ("错误", "错误"),
        ("数据库，系统", "数据 据库 系统"),
        ("中文。英文", "中文 英文"),
        ("数a据", "数 a 据"),
        ("人々", "人々"),
        ("㐀㐁㐂", "㐀㐁 㐁㐂"),
        ("データベース", "デー ータ タベ ベー ース"),
        ("カタカナ", "カタ タカ カナ"),
        ("데이터", "데이 이터"),
        ("한국어", "한국 국어"),
    ],
    ids=[
        "three-chars",
        "five-chars",
        "four-chars",
        "single-unigram",
        "two-chars-one-bigram",
        "error-word",
        "fullwidth-comma-separates",
        "ideographic-period-separates",
        "single-cjk-around-latin",
        "iteration-mark",
        "extension-a",
        "katakana",
        "katakana-short",
        "hangul",
        "hangul-short",
    ],
)
def test_index_cjk_runs_become_overlapping_bigrams(text, expected):
    assert tokenize_for_index(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Oracle Database", "oracle database"),
        ("ORA-00600", "ora 00600"),
        ("hello_world", "hello world"),
        ("ABC123def", "abc123def"),
        ("  spaced\tout\n", "spaced out"),
        ("café", "café"),
        ("Über", "über"),
        ("База данных", "база данных"),
        ("12345", "12345"),
        ("don't", "don t"),
    ],
)
def test_index_latin_runs_become_lowercase_word_tokens(text, expected):
    assert tokenize_for_index(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Oracle数据库ORA-00600错误", "oracle 数据 据库 ora 00600 错误"),
        ("使用Python3编写", "使用 python3 编写"),
        ("ABC中文DEF", "abc 中文 def"),
        ("中文abc", "中文 abc"),
        ("abc中文", "abc 中文"),
        ("ERROR错误", "error 错误"),
    ],
)
def test_index_mixed_cjk_and_latin(text, expected):
    assert tokenize_for_index(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "\t\n\r",
        "!!!???",
        "。。。、《》",
        "!|&()",
        "-_-",
        "\u3000",
        "🎉🎉",
        "\u0301",
    ],
    ids=[
        "empty",
        "spaces",
        "control-whitespace",
        "ascii-punctuation",
        "cjk-punctuation",
        "tsquery-operators",
        "hyphen-underscore",
        "ideographic-space",
        "emoji",
        "combining-mark",
    ],
)
def test_separator_only_input_yields_empty_on_both_sides(text):
    assert tokenize_for_index(text) == ""
    assert tokenize_for_query(text) == ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a & b", "a & b"),
        ("a | b", "a & b"),
        ("!a", "a"),
        ("(a)", "a"),
        ("a <-> b", "a & b"),
        ("it's", "it & s"),
        ("back\\slash", "back & slash"),
        ("a:b", "a & b"),
        ("a*b", "a & b"),
        ("a!b|c&d(e)<->f'g\\h:i*j", "a & b & c & d & e & f & g & h & i & j"),
        ("'; DROP TABLE users--", "drop & table & users"),
        ("' OR 1=1 --", "or & 1 & 1"),
        ("/*comment*/", "comment"),
        ("%%", ""),
        ("数据 & 库", "数据 & 库"),
        ("a@b.com", "a & b & com"),
        ("weight>5", "weight & 5"),
        ("\"quoted\"", "quoted"),
    ],
)
def test_query_neutralizes_reserved_characters(text, expected):
    assert tokenize_for_query(text) == expected


@pytest.mark.parametrize(
    "separator",
    [
        "&", "|", "!", "(", ")", "<->", "'", "\\", ":", "*",
        "<", ">", "~", "@", "#", "$", "%", "^", "`", "{", "}",
        "[", "]", ";", "=", "+", "/", ",", ".", "?", "-", "_",
    ],
)
def test_every_special_char_separates_tokens(separator):
    text = f"alpha{separator}beta"
    assert tokenize_for_query(text) == "alpha & beta"
    assert tokenize_for_index(text) == "alpha beta"


@pytest.mark.parametrize(
    "text",
    [
        "a!b|c&d(e)<->f'g\\h:i*j",
        "'; DROP TABLE users--",
        "数据 & 库",
        "Oracle数据库ORA-00600错误",
        "!|&()",
        "数据库",
        "a & b",
        "ＯＲＡＣＬＥ",
    ],
)
def test_query_output_is_alphanumeric_tokens_joined_by_and(text):
    expression = tokenize_for_query(text)
    parts = expression.split(" & ") if expression else []
    assert all(part and part.isalnum() for part in parts)
    assert parts == list(tokenize(text))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ＯＲＡＣＬＥ", "oracle"),
        ("Ｏｒａｃｌｅ数据库", "oracle 数据 据库"),
        ("００６００", "00600"),
        ("ﬁle", "file"),
        ("ﬀ", "ff"),
        ("①②", "12"),
        ("㈱", "株"),
        ("㍿", "株式 式会 会社"),
        ("½", "1 2"),
        ("ｽｹｰﾙ", "スケ ケー ール"),
        ("ᴬ", "a"),
    ],
)
def test_nfkc_normalization_before_tokenizing(text, expected):
    assert tokenize_for_index(text) == expected


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("ＯＲＡＣＬＥ", "ORACLE"),
        ("００６００", "00600"),
        ("ﬁle", "file"),
        ("ｽｹｰﾙ", "スケール"),
        ("㈱", "(株)"),
        ("ERROR", "error"),
        ("ＡＢＣ中文", "ABC中文"),
    ],
)
def test_nfkc_equivalent_inputs_token_identically(left, right):
    assert tokenize(left) == tokenize(right)
    assert tokenize_for_index(left) == tokenize_for_index(right)
    assert tokenize_for_query(left) == tokenize_for_query(right)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Straße", "straße"),
        ("İstanbul", "i stanbul"),
    ],
    ids=["sharp-s-kept", "dotted-capital-i-lowercases-to-combining-sequence"],
)
def test_lowercase_rules(text, expected):
    assert tokenize_for_index(text) == expected


@pytest.mark.parametrize(
    ("text", "expected_count", "uniform_token"),
    [
        ("数" * 100000, MAX_TOKENS, "数数"),
        ("word " * 100000, MAX_TOKENS, "word"),
        ("词" * 5000, MAX_TOKENS, "词词"),
        ("数据库abc " * 20000, MAX_TOKENS, None),
        ("词" * 1002, 1001, "词词"),
        ("词" * 4097, MAX_TOKENS, "词词"),
    ],
    ids=[
        "cjk-100k",
        "latin-100k-words",
        "cjk-5000-bigrams",
        "mixed-60k-tokens",
        "over-1000-tokens-not-capped",
        "cap-boundary-exact",
    ],
)
def test_long_input_is_bounded_to_max_tokens(text, expected_count, uniform_token):
    tokens = tokenize(text)
    assert len(tokens) == expected_count
    assert len(tokenize_for_index(text).split()) == expected_count
    query_parts = tokenize_for_query(text).split(" & ")
    assert len(query_parts) == expected_count
    if uniform_token is not None:
        assert set(tokens) == {uniform_token}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a" * 1000000, "a" * MAX_TOKEN_LENGTH),
        ("a" * 200 + "b", "a" * MAX_TOKEN_LENGTH),
        ("a" * MAX_TOKEN_LENGTH, "a" * MAX_TOKEN_LENGTH),
        ("a" * (MAX_TOKEN_LENGTH + 1), "a" * MAX_TOKEN_LENGTH),
    ],
    ids=["megabyte-run", "prefix-kept-suffix-dropped", "exact-cap", "one-over-cap"],
)
def test_word_run_truncated_to_prefix_cap(text, expected):
    assert tokenize_for_index(text) == expected
    assert tokenize_for_query(text) == expected


@pytest.mark.parametrize(
    ("text", "exact_length"),
    [
        ("数" * 100000, MAX_TOKENS * 2 + 3 * (MAX_TOKENS - 1)),
        ("word " * 100000, MAX_TOKENS * 4 + 3 * (MAX_TOKENS - 1)),
        ("a" * 1000000, MAX_TOKEN_LENGTH),
        (("a" * 300 + "中文 ") * 5000, None),
    ],
    ids=["cjk-bigrams", "latin-words", "single-truncated-token", "mixed"],
)
def test_query_length_bounded(text, exact_length):
    expression = tokenize_for_query(text)
    assert len(expression) <= _MAX_QUERY_CHARS
    if exact_length is not None:
        assert len(expression) == exact_length


@pytest.mark.parametrize(
    "text",
    [
        "",
        "数",
        "数据库",
        "Oracle数据库ORA-00600错误",
        "a & b",
        "!|&()",
        "ＯＲＡＣＬＥ",
        "café",
        "使用Python3编写",
        "数据，库",
        "a" * 300,
        "数" * 10000,
        "it's",
        "🎉",
        "人々",
        "ｽｹｰﾙ",
    ],
)
def test_write_and_query_sides_share_one_token_sequence(text):
    tokens = tokenize(text)
    assert tuple(tokenize_for_index(text).split()) == tokens
    query_parts = tokenize_for_query(text).split(" & ") if tokens else []
    assert query_parts == list(tokens)
    if not tokens:
        assert tokenize_for_index(text) == ""
        assert tokenize_for_query(text) == ""