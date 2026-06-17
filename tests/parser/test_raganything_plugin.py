from lightrag.parser import registry


def test_raganything_plugin_registers_parser_engine(tmp_path, monkeypatch):
    from lightrag_raganything_parser.plugin import register

    monkeypatch.setenv("RAGANYTHING_PATH", str(tmp_path))
    registry._REGISTRY.pop("raganything", None)
    try:
        register()
        assert "raganything" in registry.supported_parser_engines()
        assert "pdf" in registry.suffix_capabilities("raganything")
        spec = registry.parser_specs_snapshot()["raganything"]
        assert spec.queue_group == "raganything"
        assert spec.impl == "lightrag_raganything_parser.parser:RAGAnythingParser"
        assert spec.endpoint_configured() is True
    finally:
        registry._REGISTRY.pop("raganything", None)
