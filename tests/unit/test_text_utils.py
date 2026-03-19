from job_intel.utils.text import normalize_spaces, slugify, tokenize


def test_slugify_basic():
    assert slugify("Square (Block)") == "square_block"


def test_normalize_spaces():
    assert normalize_spaces("a   b\n c") == "a b c"


def test_tokenize():
    tokens = tokenize("Python, SQL + ML")
    assert "python" in tokens
    assert "sql" in tokens
