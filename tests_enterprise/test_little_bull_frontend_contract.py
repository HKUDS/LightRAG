from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_little_bull_graph_properties_are_read_only_for_workspace_context():
    graph_viewer = (REPO_ROOT / "lightrag_webui/src/features/GraphViewer.tsx").read_text()
    properties_view = (REPO_ROOT / "lightrag_webui/src/components/graph/PropertiesView.tsx").read_text()

    assert "<PropertiesView readOnly={Boolean(workspaceId)} />" in graph_viewer
    assert "const PropertiesView = ({ readOnly = false }" in properties_view
    assert "isEditable={!readOnly &&" in properties_view


def test_little_bull_graph_labels_do_not_seed_generic_legacy_fallbacks():
    graph_labels = (REPO_ROOT / "lightrag_webui/src/components/graph/GraphLabels.tsx").read_text()

    assert "fallbackLabels" not in graph_labels
    assert "['entity', 'relationship', 'document'" not in graph_labels
