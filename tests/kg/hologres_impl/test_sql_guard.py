import pytest

from lightrag.kg.hologres.client import (
    HologresSqlError,
    quote_identifier,
    quote_qualified_identifier,
    validate_identifier,
    validate_single_statement,
)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT $1::text",
        "SELECT 'it''s; BEGIN'",
        "SELECT E'it\\'s; COMMIT'",
        'SELECT "COMMIT;identifier"',
        "SELECT $$ROLLBACK; 'still data'$$",
        "SELECT $body$BEGIN; COMMIT;$body$",
        "SELECT 1 -- COMMIT;\n",
        "SELECT /* outer ; BEGIN /* nested COMMIT; */ still outer */ 1",
        "SELECT (SELECT 'SAVEPOINT x;')",
        "WITH value AS (SELECT 'SET TRANSACTION') SELECT * FROM value",
        "SELECT 'CALL arbitrary_proc()'",
        "SELECT \"ABORT\" FROM records",
        "SELECT /* END TRANSACTION */ 1",
        "SELECT (SELECT 'SET SESSION CHARACTERISTICS AS TRANSACTION')",
        "SELECT 'DO $$BEGIN CALL unsafe(); END$$'",
        'SELECT "DO" FROM records',
        "SELECT /* DO $$BEGIN COMMIT; END$$ */ 1",
        "SELECT (SELECT 'DO')",
    ],
)
def test_single_statement_guard_accepts_semicolon_and_keywords_only_in_lexical_regions(sql):
    validate_single_statement(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1;",
        "SELECT 1; SELECT 2",
        ";SELECT 1",
        "SELECT 'safe\\'; COMMIT -- '",
        "SELECT name$tag$; ROLLBACK; $tag$",
        "BEGIN",
        "START TRANSACTION",
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT checkpoint_one",
        "RELEASE checkpoint_one",
        "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE",
        "SET LOCAL TRANSACTION ISOLATION LEVEL SERIALIZABLE",
        "SET SESSION TRANSACTION ISOLATION LEVEL SERIALIZABLE",
        "PREPARE TRANSACTION 'transaction-id'",
        "CALL arbitrary_proc()",
        "DO $$BEGIN CALL arbitrary_proc(); END$$",
        "/* ownership is unknown */ DO $body$BEGIN COMMIT; END$body$",
        "ABORT",
        "ABORT WORK",
        "ABORT TRANSACTION",
        "END",
        "END WORK",
        "END TRANSACTION",
        "SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY",
    ],
)
def test_single_statement_guard_rejects_top_level_semicolons_and_transaction_commands(sql):
    with pytest.raises(HologresSqlError):
        validate_single_statement(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "   -- comment only",
        "SELECT 'unterminated",
        'SELECT "unterminated',
        "SELECT $tag$unterminated",
        "SELECT /* unterminated",
        "SELECT (1",
        "SELECT 1)",
    ],
)
def test_single_statement_guard_fails_closed_on_empty_or_unbalanced_sql(sql):
    with pytest.raises(HologresSqlError):
        validate_single_statement(sql)


@pytest.mark.parametrize("identifier", ["public", "LightRAG_1", "_private", "a9"])
def test_identifier_helpers_accept_and_quote_strict_identifiers(identifier):
    assert validate_identifier(identifier) == identifier
    assert quote_identifier(identifier) == f'"{identifier}"'


def test_qualified_identifier_quotes_each_validated_component():
    assert quote_qualified_identifier("LightRAG_1", "documents") == (
        '"LightRAG_1"."documents"'
    )


@pytest.mark.parametrize(
    "identifier",
    ["", "1table", "has-hyphen", "schema.table", 'has"quote', "name;drop", "a\x00b"],
)
def test_identifier_helpers_reject_unsafe_identifiers(identifier):
    with pytest.raises(HologresSqlError):
        validate_identifier(identifier)
