from cttr_vps.config import _split_keys, load_yaml_config


def test_split_keys():
    assert _split_keys("a,b, c") == ["a", "b", "c"]
    assert _split_keys(None) == []


def test_yaml_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("method: single_pass\nsamples: 2\n", encoding="utf-8")
    assert load_yaml_config(path) == {"method": "single_pass", "samples": 2}
