from filament_seg.weights_store import ensure_weight, split_weight


def test_split_and_join_restore_the_file(tmp_path):
    data = bytes(range(256)) * 41  # 10,496 bytes -> 3 parts of at most 4,000 bytes
    src = tmp_path / "model.pt"
    src.write_bytes(data)
    parts = split_weight(src, part_size=4000)
    assert [p.name for p in parts] == ["model.pt.part0", "model.pt.part1", "model.pt.part2"]
    src.unlink()  # a fresh clone holds only the parts
    out = ensure_weight(src, download=False)
    assert out.read_bytes() == data
    assert not (tmp_path / "model.pt.tmp").exists()


def test_existing_file_is_left_alone(tmp_path):
    src = tmp_path / "model.pt"
    src.write_bytes(b"weights")
    assert ensure_weight(src, download=False).read_bytes() == b"weights"
