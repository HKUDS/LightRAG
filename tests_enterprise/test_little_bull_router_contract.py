from fastapi import FastAPI

from lightrag_enterprise.little_bull.router import create_little_bull_router


def test_little_bull_popular_labels_limit_matches_frontend_contract():
    app = FastAPI()
    app.include_router(create_little_bull_router(rag=object(), doc_manager=object()))

    openapi = app.openapi()
    parameters = openapi["paths"]["/little-bull/graph/label/popular"]["get"]["parameters"]
    limit = next(parameter for parameter in parameters if parameter["name"] == "limit")

    assert limit["schema"]["default"] == 300
    assert limit["schema"]["maximum"] >= 300


def test_little_bull_upload_requires_group_and_subgroup_contract():
    app = FastAPI()
    app.include_router(create_little_bull_router(rag=object(), doc_manager=object()))

    openapi = app.openapi()
    parameters = openapi["paths"]["/little-bull/documents/upload"]["post"]["parameters"]
    required = {
        parameter["name"]
        for parameter in parameters
        if parameter.get("in") == "query" and parameter.get("required") is True
    }

    assert {"workspace_id", "group_id", "subgroup_id"}.issubset(required)


def test_little_bull_group_routes_exist_contract():
    app = FastAPI()
    app.include_router(create_little_bull_router(rag=object(), doc_manager=object()))

    openapi = app.openapi()

    assert "/little-bull/knowledge-groups" in openapi["paths"]
    assert "/little-bull/knowledge-subgroups" in openapi["paths"]
    assert {"get", "post"}.issubset(openapi["paths"]["/little-bull/knowledge-groups"])
    assert {"get", "post"}.issubset(openapi["paths"]["/little-bull/knowledge-subgroups"])


def test_little_bull_note_routes_exist_contract():
    app = FastAPI()
    app.include_router(create_little_bull_router(rag=object(), doc_manager=object()))

    openapi = app.openapi()

    assert "/little-bull/notes" in openapi["paths"]
    assert "/little-bull/notes/markdown" in openapi["paths"]
    assert "/little-bull/notes/{note_id}/markdown" in openapi["paths"]
    assert "/little-bull/tags" in openapi["paths"]
    assert "get" in openapi["paths"]["/little-bull/notes"]
    assert "post" in openapi["paths"]["/little-bull/notes/markdown"]
    assert "get" in openapi["paths"]["/little-bull/notes/{note_id}/markdown"]
    assert "get" in openapi["paths"]["/little-bull/tags"]


def test_little_bull_backlink_and_provenance_routes_exist_contract():
    app = FastAPI()
    app.include_router(create_little_bull_router(rag=object(), doc_manager=object()))

    openapi = app.openapi()

    assert "/little-bull/backlinks" in openapi["paths"]
    assert "/little-bull/provenance/panel" in openapi["paths"]
    assert "/little-bull/source-provenance" in openapi["paths"]
    assert {"get", "post"}.issubset(openapi["paths"]["/little-bull/backlinks"])
    assert "get" in openapi["paths"]["/little-bull/provenance/panel"]
    assert {"get", "post"}.issubset(openapi["paths"]["/little-bull/source-provenance"])


def test_little_bull_canvas_routes_exist_contract():
    app = FastAPI()
    app.include_router(create_little_bull_router(rag=object(), doc_manager=object()))

    openapi = app.openapi()

    assert "/little-bull/canvas/boards" in openapi["paths"]
    assert "/little-bull/canvas/boards/{canvas_board_id}" in openapi["paths"]
    assert "/little-bull/canvas/boards/{canvas_board_id}/nodes" in openapi["paths"]
    assert "/little-bull/canvas/boards/{canvas_board_id}/edges" in openapi["paths"]
    assert "/little-bull/canvas/boards/{canvas_board_id}/analysis" in openapi["paths"]
    assert "/little-bull/canvas/boards/{canvas_board_id}/dossier" in openapi["paths"]
    assert {"get", "post"}.issubset(openapi["paths"]["/little-bull/canvas/boards"])
    assert "get" in openapi["paths"]["/little-bull/canvas/boards/{canvas_board_id}"]
    assert "post" in openapi["paths"]["/little-bull/canvas/boards/{canvas_board_id}/nodes"]
    assert "post" in openapi["paths"]["/little-bull/canvas/boards/{canvas_board_id}/edges"]
    assert "get" in openapi["paths"]["/little-bull/canvas/boards/{canvas_board_id}/analysis"]
    assert "post" in openapi["paths"]["/little-bull/canvas/boards/{canvas_board_id}/dossier"]


def test_little_bull_content_map_and_trail_routes_exist_contract():
    app = FastAPI()
    app.include_router(create_little_bull_router(rag=object(), doc_manager=object()))

    openapi = app.openapi()

    assert "/little-bull/content-maps" in openapi["paths"]
    assert "/little-bull/knowledge-trails" in openapi["paths"]
    assert "/little-bull/knowledge-trails/{knowledge_trail_id}" in openapi["paths"]
    assert "/little-bull/knowledge-trails/{knowledge_trail_id}/steps" in openapi["paths"]
    assert {"get", "post"}.issubset(openapi["paths"]["/little-bull/content-maps"])
    assert {"get", "post"}.issubset(openapi["paths"]["/little-bull/knowledge-trails"])
    assert "get" in openapi["paths"]["/little-bull/knowledge-trails/{knowledge_trail_id}"]
    assert "post" in openapi["paths"]["/little-bull/knowledge-trails/{knowledge_trail_id}/steps"]


def test_little_bull_inbox_and_daily_note_routes_exist_contract():
    app = FastAPI()
    app.include_router(create_little_bull_router(rag=object(), doc_manager=object()))

    openapi = app.openapi()

    assert "/little-bull/inbox" in openapi["paths"]
    assert "/little-bull/inbox/{inbox_item_id}/status" in openapi["paths"]
    assert "/little-bull/curator/suggestions" in openapi["paths"]
    assert "/little-bull/curator/suggestions/{inbox_item_id}/apply" in openapi["paths"]
    assert "/little-bull/daily-notes" in openapi["paths"]
    assert "/little-bull/daily-notes/ensure" in openapi["paths"]
    assert {"get", "post"}.issubset(openapi["paths"]["/little-bull/inbox"])
    assert "post" in openapi["paths"]["/little-bull/inbox/{inbox_item_id}/status"]
    assert {"get", "post"}.issubset(openapi["paths"]["/little-bull/curator/suggestions"])
    assert "post" in openapi["paths"]["/little-bull/curator/suggestions/{inbox_item_id}/apply"]
    assert "get" in openapi["paths"]["/little-bull/daily-notes"]
    assert "post" in openapi["paths"]["/little-bull/daily-notes/ensure"]

    schemas = openapi["components"]["schemas"]
    curator_request_props = schemas["LittleBullCuratorSuggestionRequest"]["properties"]
    curator_response_props = schemas["LittleBullCuratorSuggestionResponse"]["properties"]

    assert {
        "suggestion_kind",
        "group_id",
        "subgroup_id",
        "source_kind",
        "source_id",
        "target_kind",
        "target_id",
    }.issubset(curator_request_props)
    assert {"inbox_item", "requires_approval", "allowed_actions"}.issubset(curator_response_props)


def test_little_bull_agent_builder_and_context_budget_routes_exist_contract():
    app = FastAPI()
    app.include_router(create_little_bull_router(rag=object(), doc_manager=object()))

    openapi = app.openapi()

    assert "/little-bull/context/estimate" in openapi["paths"]
    assert "/little-bull/costs/summary" in openapi["paths"]
    assert "/little-bull/graph/obsidian" in openapi["paths"]
    assert "/little-bull/chat/operational" in openapi["paths"]
    assert "/little-bull/operational-chat" in openapi["paths"]
    assert "/little-bull/admin/agent-builder/sessions" in openapi["paths"]
    assert "/little-bull/admin/agent-builder/sessions/{agent_builder_session_id}/publish" in openapi["paths"]
    assert "/little-bull/admin/agents/context-budgets" in openapi["paths"]
    assert "/little-bull/admin/models" in openapi["paths"]
    assert "post" in openapi["paths"]["/little-bull/context/estimate"]
    assert "get" in openapi["paths"]["/little-bull/costs/summary"]
    assert "get" in openapi["paths"]["/little-bull/graph/obsidian"]
    assert "post" in openapi["paths"]["/little-bull/chat/operational"]
    assert "post" in openapi["paths"]["/little-bull/operational-chat"]
    cost_params = {
        parameter["name"]
        for parameter in openapi["paths"]["/little-bull/costs/summary"]["get"]["parameters"]
        if parameter.get("in") == "query"
    }
    assert {"workspace_id", "user_id", "agent_id", "model_id", "operation", "group_id", "subgroup_id"}.issubset(
        cost_params
    )
    assert {"get", "post"}.issubset(openapi["paths"]["/little-bull/admin/agent-builder/sessions"])
    assert "post" in openapi["paths"][
        "/little-bull/admin/agent-builder/sessions/{agent_builder_session_id}/publish"
    ]
    assert {"get", "post"}.issubset(openapi["paths"]["/little-bull/admin/agents/context-budgets"])
    assert {"get", "post"}.issubset(openapi["paths"]["/little-bull/admin/models"])

    schemas = openapi["components"]["schemas"]
    query_props = schemas["LittleBullQueryRequest"]["properties"]
    estimate_request_props = schemas["LittleBullContextEstimateRequest"]["properties"]
    estimate_response_props = schemas["LittleBullContextEstimateResponse"]["properties"]
    cost_response_props = schemas["LittleBullCostSummaryResponse"]["properties"]
    graph_response_props = schemas["LittleBullObsidianGraphResponse"]["properties"]
    operational_request_props = schemas["LittleBullOperationalChatRequest"]["properties"]
    operational_response_props = schemas["LittleBullOperationalChatResponse"]["properties"]
    graph_params = {
        parameter["name"]
        for parameter in openapi["paths"]["/little-bull/graph/obsidian"]["get"]["parameters"]
        if parameter.get("in") == "query"
    }

    assert {"group_id", "subgroup_id", "document_ids", "top_k"}.issubset(query_props)
    assert {"document_ids", "top_k", "reserved_response_tokens"}.issubset(estimate_request_props)
    assert {
        "available_context_tokens",
        "overflow_tokens",
        "retrieval_chunk_limit",
        "reserved_response_tokens",
    }.issubset(estimate_response_props)
    assert {"periods", "by_user", "by_agent", "by_model", "by_group_subgroup", "by_operation"}.issubset(
        cost_response_props
    )
    assert {"workspace_id", "scope", "group_id", "subgroup_id", "central_node_id", "origin_type", "max_nodes"}.issubset(
        graph_params
    )
    assert {"nodes", "edges", "clusters", "trails", "chat_context"}.issubset(graph_response_props)
    assert {
        "agent_id",
        "group_id",
        "subgroup_id",
        "document_ids",
        "conversation_id",
        "save_conversation",
        "transform_to",
    }.issubset(operational_request_props)
    assert {"sources", "context", "cost_estimate", "conversation", "note", "suggestion"}.issubset(
        operational_response_props
    )
