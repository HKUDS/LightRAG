import json

from lightrag.parser.video.media import AudioSegment, MediaInfo
from lightrag.parser.video.parser import VideoParser
from lightrag.parser.video.scene import SceneWindow
from lightrag.sidecar import write_sidecar


def test_video_parser_writes_manifest_and_timecoded_drawing_sidecar(tmp_path, monkeypatch):
    source = tmp_path / "demo.mp4"
    source.write_bytes(b"test video placeholder")
    monkeypatch.setenv("VIDEO_ASR_BINDING_HOST", "http://asr/v1")

    monkeypatch.setattr(
        "lightrag.parser.video.parser.probe_media",
        lambda *_args, **_kwargs: MediaInfo(4.0, 1280, 720, 30.0, True, "h264", "aac"),
    )
    monkeypatch.setattr(
        "lightrag.parser.video.parser.detect_windows",
        lambda *_args, **_kwargs: [
            SceneWindow("scene-0001", 1, 0.0, 4.0, (0.0, 2.0, 4.0))
        ],
    )

    def fake_sheet(_source, _window, destination, **_kwargs):
        destination.write_bytes(b"jpeg")
        return [0, 2000, 4000]

    monkeypatch.setattr("lightrag.parser.video.parser.create_contact_sheet", fake_sheet)
    monkeypatch.setattr(
        "lightrag.parser.video.parser.extract_audio_segment",
        lambda *_args, **_kwargs: AudioSegment(0, 4, b"wav"),
    )

    class FakeAsr:
        def __init__(self, *_args, **_kwargs):
            pass

        def transcribe(self, _data, **_kwargs):
            return "spoken words"

    monkeypatch.setattr("lightrag.parser.video.parser.VideoAsrClient", FakeAsr)
    parser = VideoParser()
    parsed_dir = tmp_path / "demo.parsed"
    asset_dir = parsed_dir / "demo.blocks.assets"
    asset_dir.mkdir(parents=True)
    blocks, warnings, metadata = parser.extract(
        source, parsed_dir=parsed_dir, asset_dir=asset_dir, base_name="demo"
    )
    ir = parser.build_ir(
        blocks,
        document_name="demo.mp4",
        asset_dir_name=asset_dir.name,
        metadata=metadata,
    )
    result = write_sidecar(ir, parsed_dir=parsed_dir, doc_id="doc-" + "a" * 32, engine="video", clean_parsed_dir=False)

    assert warnings == {"warnings": []}
    assert result["content"].count("spoken words") == 1
    drawing_payload = json.loads((parsed_dir / "demo.drawings.json").read_text())
    drawing = next(iter(drawing_payload["drawings"].values()))
    assert drawing["extras"]["media_type"] == "video_contact_sheet"
    assert drawing["extras"]["start_ms"] == 0
    assert (asset_dir / "scene-0001.jpg").is_file()
    manifest = json.loads((parsed_dir / "demo.video.manifest.json").read_text())
    assert manifest["scenes"][0]["scene_id"] == "scene-0001"
