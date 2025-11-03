from apolo_apps_lightrag import (
    LightRAGAppInputs,
    LightRAGAppOutputs,
    LightRAGInputsProcessor,
    LightRAGOutputsProcessor,
)


def test_package_exports() -> None:
    assert LightRAGAppInputs.__name__ == "LightRAGAppInputs"
    assert LightRAGAppOutputs.__name__ == "LightRAGAppOutputs"
    assert LightRAGInputsProcessor.__name__ == "LightRAGInputsProcessor"
    assert LightRAGOutputsProcessor.__name__ == "LightRAGOutputsProcessor"
