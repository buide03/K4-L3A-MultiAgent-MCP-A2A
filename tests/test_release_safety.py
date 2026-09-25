from pathlib import Path


def test_repository_contains_only_released_input_payload() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "case-set.json").is_file()
    assert len(list((root / "inputs").glob("*.json"))) == 100
    forbidden = {"oracles", "reference-outputs", "private-partitions.json", "mcp-access.json"}
    assert not any(path.name in forbidden for path in root.rglob("*"))


def test_example_environment_has_no_real_key() -> None:
    root = Path(__file__).resolve().parents[1]
    content = (root / ".env.example").read_text(encoding="utf-8")
    assert "sk-team-replace_me" in content
    assert content.count("sk-team-") == 1
