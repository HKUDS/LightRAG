import pytest
from lightrag.operate import extract_entities


@pytest.mark.asyncio
async def test_extract_entities_empty_chunks():
    """
    Locks out the bug where calling extract_entities with an empty chunks dict ({})
    caused an unhandled exception instead of returning an empty list [].
    """
    global_config = {
        "role_llm_funcs": {"extract": None},
        "entity_extract_max_gleaning": 0,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 100,
    }
    result = await extract_entities({}, global_config)
    assert result == []
