from lightrag.parser.video.scene import build_scene_windows


def test_scene_windows_split_long_ranges_and_keep_frame_timestamps():
    windows = build_scene_windows(
        100,
        [20, 60],
        max_scene_seconds=25,
        min_scene_seconds=2,
        frames_per_scene=3,
    )

    assert [(round(item.start_seconds), round(item.end_seconds)) for item in windows] == [
        (0, 20),
        (20, 40),
        (40, 60),
        (60, 80),
        (80, 100),
    ]
    assert windows[0].scene_id == "scene-0001"
    assert windows[0].frame_timestamps == (0.0, 10.0, 20.0)


def test_scene_cap_is_deterministic():
    windows = build_scene_windows(100, list(range(1, 100)), max_scenes=4)
    assert len(windows) == 4
    assert [item.scene_id for item in windows] == [
        "scene-0001",
        "scene-0002",
        "scene-0003",
        "scene-0004",
    ]
