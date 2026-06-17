from types import SimpleNamespace
import sys

import pytest
import yaml
from dotenv import dotenv_values
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(tmp_path, monkeypatch, *, addon_params=None):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.chdir(tmp_path)
    prompt_dir = tmp_path / "prompts"
    entity_dir = prompt_dir / "entity_type"
    entity_dir.mkdir(parents=True)
    monkeypatch.setenv("PROMPT_DIR", str(prompt_dir))

    rag = SimpleNamespace(
        workspace="project_a",
        addon_params=addon_params or {"entity_type_prompt_file": "finance.yml"},
    )
    args = SimpleNamespace(
        workspace="project_a",
        llm_binding="openai",
        llm_model="gpt-4o-mini",
        llm_binding_host="https://api.openai.com/v1",
        embedding_binding="openai",
        embedding_model="text-embedding-3-small",
        embedding_binding_host="https://api.openai.com/v1",
        top_k=40,
        chunk_top_k=20,
        max_total_tokens=30000,
        max_parallel_insert=3,
    )
    from lightrag.api.routers.config_routes import create_config_routes

    app = FastAPI()
    app.include_router(create_config_routes(rag, args, api_key=None))
    return TestClient(app), entity_dir


def test_get_workbench_returns_env_fields_and_prompt_profiles(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "WORKSPACE=project_a",
                "LLM_MODEL=gpt-5-mini",
                "LLM_BINDING_API_KEY=sk-secret",
                "ENTITY_TYPE_PROMPT_FILE=finance.yml",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "env.example").write_text("LLM_MODEL=gpt-4o-mini\n", encoding="utf-8")

    client, entity_dir = _client(tmp_path, monkeypatch)
    (entity_dir / "finance.yml").write_text(
        yaml.safe_dump({"entity_types_guidance": "- Company\n- Person"}, sort_keys=False),
        encoding="utf-8",
    )

    response = client.get("/config/workbench")

    assert response.status_code == 200
    payload = response.json()
    assert payload["workspace"]["current"] == "project_a"
    assert ".env" in [profile["name"] for profile in payload["env"]["profiles"]]
    assert "finance.yml" in [
        profile["name"] for profile in payload["prompts"]["entity_type_profiles"]
    ]

    fields = {
        field["key"]: field
        for section in payload["env"]["sections"]
        for field in section["fields"]
    }
    assert fields["LLM_MODEL"]["value"] == "gpt-5-mini"
    assert fields["LLM_BINDING_API_KEY"]["value"] == ""
    assert fields["LLM_BINDING_API_KEY"]["configured"] is True
    assert fields["LLM_BINDING_API_KEY"]["sensitive"] is True
    assert fields["ENTITY_TYPE_PROMPT_FILE"]["value"] == "finance.yml"

    prompt = payload["prompts"]["stages"][0]
    assert prompt["key"] == "entity_type"
    assert prompt["editable"] is True
    assert prompt["content"] == "- Company\n- Person"


def test_workspace_list_can_remove_entries_without_deleting_directories(
    tmp_path, monkeypatch
):
    working_dir = tmp_path / "rag_storage"
    input_dir = tmp_path / "inputs"
    beta_storage = working_dir / "project_b"
    beta_input = input_dir / "project_b"
    beta_storage.mkdir(parents=True)
    beta_input.mkdir(parents=True)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "WORKSPACE=project_a",
                f"WORKING_DIR={working_dir}",
                f"INPUT_DIR={input_dir}",
                "LIGHTRAG_WEBUI_WORKSPACES=project_a,project_b",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client, _ = _client(tmp_path, monkeypatch)

    initial = client.get("/config/workbench")

    assert initial.status_code == 200
    assert [item["name"] for item in initial.json()["workspace"]["available"]] == [
        "project_a",
        "project_b",
    ]

    response = client.put(
        "/config/workbench/env",
        json={
            "values": {
                "LIGHTRAG_WEBUI_WORKSPACES": "project_a",
                "WORKSPACE": "project_a",
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()["workbench"]
    assert [item["name"] for item in payload["workspace"]["available"]] == [
        "project_a"
    ]
    assert beta_storage.is_dir()
    assert beta_input.is_dir()


def test_workbench_exposes_chunking_strategy_fields(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "LIGHTRAG_PARSER=*:native-teP,*:legacy-R",
                "CHUNK_P_SIZE=2200",
                "CHUNK_P_OVERLAP_SIZE=120",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get("/config/workbench")

    assert response.status_code == 200
    payload = response.json()
    assert payload["chunking"]["active_strategy"] == "P"
    fields = {
        field["key"]: field
        for section in payload["env"]["sections"]
        for field in section["fields"]
    }
    assert fields["LIGHTRAG_PARSER"]["value"] == "*:native-teP,*:legacy-R"
    assert fields["CHUNK_P_SIZE"]["value"] == "2200"
    assert fields["CHUNK_P_OVERLAP_SIZE"]["value"] == "120"


def test_workbench_exposes_parser_plugin_settings(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "LIGHTRAG_PARSER=*:raganything-R,*:legacy-R",
                r"RAGANYTHING_PATH=D:\RAG-Anything",
                "RAGANYTHING_PARSER=mineru",
                "RAGANYTHING_PARSE_METHOD=auto",
                "RAGANYTHING_LANG=ch",
                'RAGANYTHING_PARSE_KWARGS={"backend":"pipeline"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get("/config/workbench")

    assert response.status_code == 200
    payload = response.json()
    sections = {section["id"]: section for section in payload["env"]["sections"]}
    parser_fields = {
        field["key"]: field
        for field in sections["parser"]["fields"]
    }
    assert parser_fields["LIGHTRAG_PARSER"]["value"] == "*:raganything-R,*:legacy-R"
    assert parser_fields["RAGANYTHING_PATH"]["value"] == r"D:\RAG-Anything"
    assert parser_fields["RAGANYTHING_PARSER"]["value"] == "mineru"
    assert parser_fields["RAGANYTHING_PARSE_METHOD"]["value"] == "auto"
    assert parser_fields["RAGANYTHING_LANG"]["value"] == "ch"
    assert parser_fields["RAGANYTHING_PARSE_KWARGS"]["value"] == '{"backend":"pipeline"}'


def test_put_env_accepts_chunking_settings(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("LLM_MODEL=old-model\n", encoding="utf-8")
    client, _ = _client(tmp_path, monkeypatch)

    response = client.put(
        "/config/workbench/env",
        json={
            "values": {
                "LIGHTRAG_PARSER": "*:native-teV,*:legacy-V",
                "CHUNK_V_SIZE": "900",
                "CHUNK_V_BREAKPOINT_THRESHOLD_TYPE": "percentile",
            }
        },
    )

    assert response.status_code == 200
    values = dotenv_values(tmp_path / ".env")
    assert values["LIGHTRAG_PARSER"] == "*:native-teV,*:legacy-V"
    assert values["CHUNK_V_SIZE"] == "900"
    assert values["CHUNK_V_BREAKPOINT_THRESHOLD_TYPE"] == "percentile"


def test_put_env_accepts_parser_plugin_settings(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("LLM_MODEL=old-model\n", encoding="utf-8")
    client, _ = _client(tmp_path, monkeypatch)

    response = client.put(
        "/config/workbench/env",
        json={
            "values": {
                "LIGHTRAG_PARSER": "*:raganything-P,*:legacy-P",
                "RAGANYTHING_PATH": r"D:\RAG-Anything",
                "RAGANYTHING_PARSER": "mineru",
                "RAGANYTHING_PARSE_METHOD": "auto",
                "RAGANYTHING_LANG": "ch",
                "RAGANYTHING_PARSE_KWARGS": '{"backend":"pipeline"}',
                "LIGHTRAG_FORCE_REPARSE_RAGANYTHING": "false",
            }
        },
    )

    assert response.status_code == 200
    values = dotenv_values(tmp_path / ".env")
    assert values["LIGHTRAG_PARSER"] == "*:raganything-P,*:legacy-P"
    assert values["RAGANYTHING_PATH"] == r"D:\RAG-Anything"
    assert values["RAGANYTHING_PARSER"] == "mineru"
    assert values["RAGANYTHING_PARSE_METHOD"] == "auto"
    assert values["RAGANYTHING_LANG"] == "ch"
    assert values["RAGANYTHING_PARSE_KWARGS"] == '{"backend":"pipeline"}'
    assert values["LIGHTRAG_FORCE_REPARSE_RAGANYTHING"] == "false"


def test_pick_workspace_folder_returns_workspace_name_and_parent(
    tmp_path, monkeypatch
):
    selected = tmp_path / "inputs" / "project_c"
    selected.mkdir(parents=True)

    import lightrag.api.routers.config_routes as config_routes

    monkeypatch.setattr(
        config_routes,
        "_pick_directory_with_shell_dialog",
        lambda initial_dir=None: str(selected),
    )
    client, _ = _client(tmp_path, monkeypatch)

    response = client.post(
        "/config/workbench/folders/pick",
        json={"initial_dir": str(tmp_path / "inputs")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_path"] == str(selected)
    assert payload["workspace"] == "project_c"
    assert payload["input_dir"] == str(selected.parent)


def test_pick_workspace_folder_can_be_cancelled(tmp_path, monkeypatch):
    import lightrag.api.routers.config_routes as config_routes

    monkeypatch.setattr(
        config_routes,
        "_pick_directory_with_shell_dialog",
        lambda initial_dir=None: None,
    )
    client, _ = _client(tmp_path, monkeypatch)

    response = client.post("/config/workbench/folders/pick", json={})

    assert response.status_code == 200
    assert response.json() == {"selected_path": None}


def test_get_workbench_can_select_entity_prompt_profile(tmp_path, monkeypatch):
    client, entity_dir = _client(tmp_path, monkeypatch)
    (entity_dir / "finance.yml").write_text(
        yaml.safe_dump({"entity_types_guidance": "- Company"}, sort_keys=False),
        encoding="utf-8",
    )
    (entity_dir / "medical.yml").write_text(
        yaml.safe_dump({"entity_types_guidance": "- Disease"}, sort_keys=False),
        encoding="utf-8",
    )

    response = client.get("/config/workbench", params={"prompt_profile": "medical.yml"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["prompts"]["entity_type_active_profile"] == "medical.yml"
    prompt = payload["prompts"]["stages"][0]
    assert prompt["profile"] == "medical.yml"
    assert prompt["content"] == "- Disease"


def test_put_env_updates_whitelisted_values_and_preserves_blank_secret(
    tmp_path, monkeypatch
):
    (tmp_path / ".env").write_text(
        "LLM_MODEL=old-model\nLLM_BINDING_API_KEY=sk-secret\n", encoding="utf-8"
    )
    client, _ = _client(tmp_path, monkeypatch)

    response = client.put(
        "/config/workbench/env",
        json={"values": {"LLM_MODEL": "new-model", "LLM_BINDING_API_KEY": ""}},
    )

    assert response.status_code == 200
    values = dotenv_values(tmp_path / ".env")
    assert values["LLM_MODEL"] == "new-model"
    assert values["LLM_BINDING_API_KEY"] == "sk-secret"
    assert response.json()["requires_restart"] is True


def test_put_env_rejects_unknown_keys(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("LLM_MODEL=old-model\n", encoding="utf-8")
    client, _ = _client(tmp_path, monkeypatch)

    response = client.put(
        "/config/workbench/env",
        json={"values": {"NOT_A_LIGHTRAG_SETTING": "bad"}},
    )

    assert response.status_code == 400
    assert "NOT_A_LIGHTRAG_SETTING" in response.json()["detail"]


def test_put_entity_prompt_writes_profile_in_prompt_dir(tmp_path, monkeypatch):
    client, entity_dir = _client(tmp_path, monkeypatch)

    response = client.put(
        "/config/workbench/prompts/entity-type",
        json={
            "profile": "custom.yml",
            "entity_types_guidance": "- Company\n- Person\n- Metric",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"] == "custom.yml"
    assert payload["requires_restart"] is True
    written = yaml.safe_load((entity_dir / "custom.yml").read_text(encoding="utf-8"))
    assert written["entity_types_guidance"] == "- Company\n- Person\n- Metric"


@pytest.mark.parametrize("profile", ["../escape.yml", "nested/profile.yml", "bad.txt"])
def test_put_entity_prompt_rejects_unsafe_profile_names(tmp_path, monkeypatch, profile):
    client, _ = _client(tmp_path, monkeypatch)

    response = client.put(
        "/config/workbench/prompts/entity-type",
        json={"profile": profile, "entity_types_guidance": "- Person"},
    )

    assert response.status_code == 400
